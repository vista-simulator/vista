""" Run in gpl root directory with python -m tools.inspect_dataset ... """
import os
import argparse
from importlib import import_module
import tqdm
import numpy as np
import matplotlib.pyplot as plt

import tools.utils as utils
from vista.utils import logging
logging.setLevel(logging.ERROR)


def main():
    # Parse arguments and config
    parser = argparse.ArgumentParser(description='Inspect optimal control')
    parser.add_argument('--config', type=str, default=None,
        help='Path to .yaml config file. Will overwrite default config')
    parser.add_argument('--mode', type=str, required=True,
        choices=['privileged_control'], help='Inspect mode')
    parser.add_argument('--outdir', type=str,
        default=os.environ.get('TMPDIR', '/tmp/vista/'), help='Output directory')
    parser.add_argument('--n-trials', type=int, default=10,
        help='Number of trials to be run')
    args = parser.parse_args()

    default_config_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'config/default.yaml')
    config = utils.load_yaml(default_config_path)

    if args.config is not None:
        args.config = utils.validate_path(args.config)
        utils.update_dict(config, utils.load_yaml(args.config))
    utils.preprocess_config(config) # validate paths

    # Instantiate dataset
    dataset_mod = import_module('.' + config.dataset.type, 'datasets')
    dataset = dataset_mod.VistaDataset(**config.dataset)
    dataset_iter = iter(dataset)

    # Initialization for different inspect mode
    if args.mode == 'privileged_control':
        fig, ax = plt.subplots(1, 1)
        args.outdir = utils.validate_path(args.outdir)
        if not os.path.isdir(args.outdir):
            os.makedirs(args.outdir)
    else:
        raise NotImplementedError(f'Unrecognized mode {args.mode}')

    # Run
    for trial_i in tqdm.tqdm(range(args.n_trials)):
        if args.mode == 'privileged_control':
            next(dataset_iter) # initialize some properties of dataset and sequence reset
            agent = dataset._agent
            human_traj, ego_traj = [], []
            def _seq_reset():
                return agent.done or dataset._snippet_i >= dataset.snippet_size # TODO: hacky
            while not _seq_reset():
                human_xy = agent.human_dynamics.numpy()[:2]
                ego_xy = agent.ego_dynamics.numpy()[:2]
                human_traj.append(human_xy)
                ego_traj.append(ego_xy)
                next(dataset_iter)
            human_traj = np.array(human_traj)
            ego_traj = np.array(ego_traj)

            ax.clear()
            ax.plot(human_traj[:,0], human_traj[:,1], c='r', linewidth=2., label='human')
            ax.plot(ego_traj[:,0], ego_traj[:,1], c='b', label='privileged control')
            xlim_abs_max = np.abs(ax.get_xlim()).max()
            xlim = max(xlim_abs_max, 1.0)
            ax.set_xlim(-xlim, xlim)
            ax.legend()
            fig.savefig(os.path.join(args.outdir, f'trial_{trial_i:02d}.jpg'))
        else:
            raise NotImplementedError(f'Unrecognized mode {args.mode}')


if __name__ == '__main__':
    main()
