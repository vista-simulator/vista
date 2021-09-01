import os
import re
import argparse
import copy
import shutil
from importlib import import_module
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import tools.utils as utils
import tools.objectives as objectives

from vista.utils import logging
logging.setLevel(logging.ERROR)


def main():
    # Parse arguments and config
    parser = argparse.ArgumentParser(description='IL/GPL in Vista')
    parser.add_argument('--config',
                        type=str,
                        default=None,
                        help='Path to .yaml config file. Overwrites default')
    parser.add_argument('--logdir',
                        type=str,
                        required=True,
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--num-workers',
                        type=int,
                        default=0,
                        help='Number of workers used in dataloader')
    parser.add_argument('--val-num-workers',
                        type=int,
                        default=0,
                        help='Number of workers used in validation dataloader')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='Disable gpu')
    parser.add_argument('--restore',
                        type=str,
                        default=None,
                        help='Path to checkpoint to be restored from')
    args = parser.parse_args()

    args.logdir = utils.validate_path(args.logdir)
    args.config = utils.validate_path(args.config)

    default_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'config/default.yaml')
    config = utils.load_yaml(default_config_path)
    utils.update_dict(config, utils.load_yaml(args.config))

    device = torch.device('cuda' if not args.no_cuda else 'cpu')

    # Set up output directory
    if not os.path.isdir(args.logdir):
        os.makedirs(args.logdir)
    ckpt_dir = os.path.join(args.logdir, 'ckpt')
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    logger = utils.Logger(args.logdir, write_mode='a' if args.restore else 'w')
    logger.create_group('train')
    logger.create_group('val')

    shutil.copy(
        default_config_path,
        os.path.join(args.logdir, os.path.basename(default_config_path)))
    shutil.copy(args.config,
                os.path.join(args.logdir, os.path.basename(args.config)))
    with open(os.path.join(args.logdir, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)

    utils.preprocess_config(config)  # validate paths
    logger.print(config)

    torch.multiprocessing.set_start_method(
        config.get('mp_start_method', 'fork'))

    # Define data loader
    dataset_mod = import_module('.' + config.dataset.type, 'datasets')

    train_dataset = dataset_mod.VistaDataset(**config.dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.dataset.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              worker_init_fn=dataset_mod.worker_init_fn)
    train_batch_iter = iter(train_loader)

    val_dataset_config = copy.deepcopy(config.dataset)
    config.val_dataset = utils.update_dict(val_dataset_config,
                                           config.val_dataset)
    val_dataset = dataset_mod.VistaDataset(**config.val_dataset, train=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=config.val_dataset.batch_size,
                            num_workers=args.val_num_workers,
                            pin_memory=True,
                            worker_init_fn=dataset_mod.worker_init_fn)
    val_batch_iter = iter(val_loader)

    # Define model
    extractors = nn.ModuleDict()
    for modal, cfg in config.model.extractors.items():
        modal = 'camera' if modal in ['fcamera'] else modal
        extractor_mod = import_module('.' + cfg['name'],
                                      f'models.extractors.{modal}')
        extractors[modal] = extractor_mod.Net()

    estimator_mod = import_module('.' + config.model.estimator.name,
                                  'models.estimators')
    model = estimator_mod.Net(extractors).to(device)

    # Define optimizer and objective function
    optimizer = getattr(optim, config.optimizer.name)(model.parameters(),
                                                      **config.optimizer.cfg)
    objective = getattr(objectives, config.objective.name)(
        **(dict() if config.objective.cfg is None else config.objective.cfg))

    # Restore from checkpoint
    iter_i = 0
    if args.restore:
        logger.print(f'Restore from {args.restore}')
        args.restore = utils.validate_path(args.restore)
        iter_i = utils.load_checkpoint(args.restore, model, optimizer)

    # Run training
    loss_buf = []
    while iter_i < config.n_iters:
        loss = train_iter(config, device, train_batch_iter, model, objective,
                          optimizer, logger)
        loss_buf.append(loss)

        if iter_i % config.log_every_iters == 0:
            avg_loss = np.mean(loss_buf)
            logger.scalar('avg_loss', avg_loss, group='train')
            logger.write(iter_i, group='train')
            loss_buf = []

        if iter_i % config.save_every_iters == 0 and iter_i > 0:
            iter_str = '-'.join([_v[::-1] for _v in \
                re.findall('.{1,3}', f'{iter_i:07d}'[::-1])[::-1]])
            fpath = os.path.join(ckpt_dir, f'iter-{iter_str}.pt')
            logger.print(f'Save checkpoint to {fpath}')
            utils.save_checkpoint(fpath, iter_i, model, optimizer)

        if iter_i % config.val_every_iters == 0 and iter_i > 0:
            avg_loss = val(config, device, val_batch_iter, model, objective,
                           logger)
            logger.scalar('avg_loss', avg_loss, group='val')
            logger.write(iter_i, group='val')

        iter_i += 1


def train_iter(config, device, batch_iter, model, objective, optimizer,
               logger):
    logger.tic('train_iter', group='train')

    model.train()
    logger.tic('data', group='train')
    batch = next(batch_iter)
    target = batch.pop('target').to(device)
    data = {k: v.to(device) for k, v in batch.items()}
    logger.toc('data', group='train')

    optimizer.zero_grad()

    logger.tic('model', group='train')
    output = model(data)
    loss = objective(output, target) / len(target)
    logger.toc('model', group='train')

    logger.tic('optim', group='train')
    loss.backward()
    optimizer.step()
    logger.toc('optim', group='train')

    logger.toc('train_iter', group='train')

    return loss.item()


def val(config, device, batch_iter, model, objective, logger):
    logger.tic('val', group='val')

    model.eval()
    sample_i = 0
    loss = 0.
    while sample_i < config.val_total_iters:
        batch = next(batch_iter)
        target = batch.pop('target').to(device)
        data = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            output = model(data)
            loss += objective(output, target)
        sample_i += len(target)
    loss /= sample_i

    logger.toc('val', group='val')
    return loss.item()


if __name__ == '__main__':
    main()
