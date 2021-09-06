import os
import cv2
import argparse
import pickle
import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsparse.utils.collate import sparse_collate_fn
from importlib import import_module
from skvideo.io import FFmpegWriter

import tools.utils as utils
import tools.objectives as objectives
import datasets.utils as d_utils

import vista
from vista.tasks import LaneFollowing
from vista.entities.sensors import Camera, Lidar, EventCamera
from vista.utils import logging
logging.setLevel(logging.ERROR)


def main():
    # Parse arguments and config
    parser = argparse.ArgumentParser(description='IL/GPL in Vista')
    parser.add_argument('--ckpt',
                        type=str,
                        default=None,
                        required=True,
                        help='Path to checkpoint')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='Disable gpu')
    parser.add_argument('--mode',
                        type=str,
                        default='simulation',
                        choices=['simulation', 'passive'],
                        help='Different testing mode')
    parser.add_argument('--n-episodes',
                        type=int,
                        default=1,
                        help='[Simulation] Number of episodes to be tested')
    parser.add_argument('--trace-paths',
                        type=str,
                        nargs='+',
                        default=None,
                        help='[Simulation] Path to the traces')
    parser.add_argument('--save-video',
                        action='store_true',
                        default=False,
                        help='[Simulation] Whether to save video')
    parser.add_argument('--test-config',
                        type=str,
                        default=None,
                        help='[Passive] Configuration for testing')
    parser.add_argument('--num-workers',
                        type=int,
                        default=0,
                        help='[Passive] Number of workers used in dataloader')
    parser.add_argument(
        '--out-dir',
        type=str,
        default=None,
        help='Output directory; default set to checkpoint directory')
    args = parser.parse_args()

    args.ckpt = utils.validate_path(args.ckpt)
    config_path = os.path.join(os.path.dirname(args.ckpt), '../config.pkl')
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    utils.preprocess_config(config)
    config['dataset']['trace_config']['road_width'] = 6

    device = torch.device('cuda' if not args.no_cuda else 'cpu')

    if args.out_dir is None:
        args.out_dir = os.path.join(
            os.path.dirname(os.path.dirname(args.ckpt)), 'results',
            args.ckpt.split('/')[-1].split('.')[0])
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

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
    utils.load_checkpoint(args.ckpt, model, load_optim=False)

    if args.mode == 'simulation':
        # Define task in vista
        if config.dataset.type in ['il_rgb_dataset', 'gpl_rgb_dataset']:
            utils.set_dict_value_by_str(config, 'dataset:camera_config:type',
                                        'camera')
            utils.set_dict_value_by_str(
                config, 'dataset:camera_config:use_synthesizer', True)
            sensors_configs = [config.dataset.camera_config]
        elif config.dataset.type in ['gpl_lidar_dataset']:
            utils.set_dict_value_by_str(config, 'dataset:lidar_config:type',
                                        'lidar')
            sensors_configs = [config.dataset.lidar_config]
        elif config.dataset.type in ['gpl_event_dataset']:
            utils.set_dict_value_by_str(config,
                                        'dataset:event_camera_config:type',
                                        'event_camera')
            sensors_configs = [config.dataset.event_camera_config]
        else:
            raise NotImplementedError(
                f'Unrecognized dataset type {config.dataset.type}')
        env = LaneFollowing(trace_paths=args.trace_paths,
                            trace_config=config.dataset.trace_config,
                            car_config=config.dataset.car_config,
                            sensors_configs=sensors_configs,
                            logging_level='ERROR')
        display = vista.Display(env.world, display_config=dict(gui_scale=2))

        # Run testing
        agent_id = env.world.agents[0].id
        sensors = env.world.agents[0].sensors
        for ep_i in range(args.n_episodes):
            obs = env.reset()
            if args.save_video:
                video_path = os.path.join(args.out_dir, f'ep-{ep_i:03d}.mp4')
                video_writer = FFmpegWriter(video_path)
                display.reset()
            done = False
            while not done:
                # prepare data
                data = {}
                for sensor in sensors:
                    v = obs[agent_id][sensor.name]
                    if isinstance(sensor, Camera):
                        v = d_utils.transform_rgb(v, sensor, False)
                        data['camera'] = v[None, ...].to(device)
                    elif isinstance(sensor, Lidar):
                        v = d_utils.transform_lidar(v, sensor, False)
                        zero_coords = torch.zeros(v.coords.shape[0], 1)
                        v.coords = torch.cat([v.coords, zero_coords], 1).int()
                        data['lidar'] = v.to(device)
                    elif isinstance(sensor, EventCamera):
                        v = d_utils.transform_events(v, sensor, False)
                        data['event_camera'] = v[None, ...].to(device)
                    else:
                        raise NotImplementedError(
                            f'Cannot handle sensor data {type(sensor)}')

                with torch.no_grad():
                    curvature = model(data)
                    print(curvature[0, 0])

                action = {agent_id: [curvature.item()]}
                obs, rew, done, _ = env.step(action, dt=1 / 10.)
                done = done[agent_id]

                if args.save_video:
                    vis_img = display.render()
                    cv2.imshow('vis', vis_img[:,:,::-1])
                    cv2.waitKey(1)
                    video_writer.writeFrame(vis_img)
            video_writer.close()
    elif args.mode == 'passive':
        utils.update_dict(config, utils.load_yaml(args.test_config))
        test_dataset_config = copy.deepcopy(config.dataset)
        config.test_dataset = utils.update_dict(test_dataset_config,
                                                config.test_dataset)
        utils.preprocess_config(config, ['test_dataset:trace_paths'])

        # torch.multiprocessing.set_start_method(
        #     config.get('mp_start_method', 'fork'))

        dataset_mod = import_module('.' + config.test_dataset.type, 'datasets')
        test_dataset = dataset_mod.VistaDataset(**config.test_dataset,
                                                train=False)
        collate_fn = (sparse_collate_fn if
                      ("lidar" in config.test_dataset.type) else None)
        test_loader = DataLoader(test_dataset,
                                 batch_size=config.test_dataset.batch_size,
                                 num_workers=args.num_workers,
                                 pin_memory=True,
                                 worker_init_fn=dataset_mod.worker_init_fn,
                                 collate_fn=collate_fn)
        loader_iter = iter(test_loader)

        objective = getattr(objectives, config.objective.name)(**(
            dict() if config.objective.cfg is None else config.objective.cfg))
        objective.reduction = 'none'

        stop = False
        i = 0
        all_test_loss = []
        while not stop:
            batch = next(loader_iter)
            target = batch.pop('target').to(device)
            data = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                output = model(data)
                loss = objective(output, target).cpu().numpy()[:, 0]

            bsize = len(target)
            i += bsize
            if 'total_samples' not in locals().keys():
                # need to iterate through loader first to instantiate world object
                total_samples = np.sum(
                    [v.num_of_frames for v in test_dataset._world.traces])
                pbar = tqdm(total=total_samples)
            pbar.update(bsize)
            stop = i >= total_samples - 1
            all_test_loss.append(loss)
        pbar.close()
        print('')

        all_test_loss = np.hstack(all_test_loss)
        results = {
            'config': config,
            'test_loss': all_test_loss,
        }

        logger = utils.Logger(args.out_dir, with_tensorboard=False)
        logger.print(config)
        logger.print('')
        logger.print(f'Test loss mean: {all_test_loss.mean()}')
        logger.print(f'Test loss std: {all_test_loss.std()}')
        logger.print(f'Test loss max: {all_test_loss.max()}')
        logger.print(f'Test loss min: {all_test_loss.min()}')

        with open(os.path.join(args.out_dir, 'results.pkl'), 'wb') as f:
            pickle.dump(results, f)
    else:
        raise NotImplementedError('Unrecognized testing mode {}'.format(
            args.mode))


if __name__ == '__main__':
    main()
