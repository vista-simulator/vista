""" Run in gpl root directory with python -m tools.inspect_dataset ... """
import os
import argparse
from importlib import import_module
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torchsparse.utils.collate import sparse_collate_fn

import tools.utils as utils
import vista
from vista.utils import logging
logging.setLevel(logging.ERROR)


def main():
    # Parse arguments and config
    parser = argparse.ArgumentParser(description='Inspect optimal control')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to .yaml config file. Will overwrite default config')
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['privileged_control', 'inspect_simulator', 'compute_stats'],
        help='Inspect mode')
    parser.add_argument('--outdir',
                        type=str,
                        default=os.environ.get('TMPDIR', '/tmp/vista/'),
                        help='Output directory')
    parser.add_argument('--n-trials',
                        type=int,
                        default=10,
                        help='Number of trials to be run')
    parser.add_argument('--num-workers',
                        type=int,
                        default=0,
                        help='Number of workers for dataloader (if any)')
    args = parser.parse_args()

    default_config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'config/default.yaml')
    config = utils.load_yaml(default_config_path)

    if args.config is not None:
        args.config = utils.validate_path(args.config)
        utils.update_dict(config, utils.load_yaml(args.config))
    utils.preprocess_config(config)  # validate paths

    # Instantiate dataset
    dataset_mod = import_module('.' + config.dataset.type, 'datasets')
    dataset = dataset_mod.VistaDataset(**config.dataset)
    dataset_iter = iter(dataset)

    # Initialization for different inspect mode
    args.outdir = utils.validate_path(args.outdir)
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    if args.mode == 'privileged_control':
        fig, ax = plt.subplots(1, 1)
    elif args.mode == 'inspect_simulator':
        from skvideo.io import FFmpegWriter
        display_config = dict(road_buffer_size=1000)
    elif args.mode == 'compute_stats':
        from torch.utils.data import DataLoader
        dataset.skip_step_sensors = True
        collate = (sparse_collate_fn if
                   ("lidar" in config.dataset.type) else None)
        loader = DataLoader(dataset,
                            batch_size=1,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            worker_init_fn=dataset_mod.worker_init_fn,
                            collate_fn=collate)
        loader_iter = iter(loader)
        target_list = []
    else:
        raise NotImplementedError(f'Unrecognized mode {args.mode}')

    # Run
    for trial_i in tqdm.tqdm(range(args.n_trials)):

        def _seq_reset():
            return agent.done or dataset._snippet_i >= dataset.snippet_size  # TODO: hacky

        if args.mode == 'privileged_control':
            data = next(
                dataset_iter
            )  # initialize some properties of dataset and sequence reset
            agent = dataset._agent
            human_traj, ego_traj = [], []
            while not _seq_reset():
                human_xy = agent.human_dynamics.numpy()[:2]
                ego_xy = agent.ego_dynamics.numpy()[:2]
                human_traj.append(human_xy)
                ego_traj.append(ego_xy)
                next(dataset_iter)
            human_traj = np.array(human_traj)
            ego_traj = np.array(ego_traj)

            ax.clear()
            ax.plot(human_traj[:, 0],
                    human_traj[:, 1],
                    c='r',
                    linewidth=2.,
                    label='human')
            ax.plot(ego_traj[:, 0],
                    ego_traj[:, 1],
                    c='b',
                    label='privileged control')
            xlim_abs_max = np.abs(ax.get_xlim()).max()
            xlim = max(xlim_abs_max, 1.0)
            ax.set_xlim(-xlim, xlim)
            ax.set_aspect("equal")
            ax.legend()
            fig.savefig(os.path.join(args.outdir, f'trial_{trial_i:02d}.jpg'))
        elif args.mode == 'inspect_simulator':
            data = next(
                dataset_iter
            )  # initialize some properties of dataset and sequence reset

            video_path = os.path.join(args.outdir, f'trial_{trial_i:02d}.mp4')
            video_writer = FFmpegWriter(video_path)

            if 'display' not in globals().keys():
                display = vista.Display(dataset._world,
                                        display_config=display_config)
            display.reset()
            while not _seq_reset():
                img = display.render()
                video_writer.writeFrame(img)
                data = next(dataset_iter)
            video_writer.close()
        elif args.mode == 'compute_stats':
            data = next(loader_iter)
            target_list.append(data['target'].cpu().numpy()[:, 0])
        else:
            raise NotImplementedError(f'Unrecognized mode {args.mode}')

    # Post processing
    if args.mode == 'compute_stats':
        target_list = np.hstack(target_list)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes[0].hist(target_list, bins=50)
        axes[0].set_title(f'Target Count#{args.n_trials}')
        axes[1].hist(target_list, bins=50)
        axes[1].set_title(f'Target Count#{args.n_trials} (log-scale)')
        axes[1].set_yscale('log')
        out_path = os.path.join(args.outdir, 'hist_curvature.jpg')
        fig.tight_layout()
        fig.savefig(out_path)
    elif args.mode in ['privileged_control', 'inspect_simulator']:
        pass
    else:
        raise NotImplementedError(f'Unrecognized mode {args.mode}')


if __name__ == '__main__':
    main()
