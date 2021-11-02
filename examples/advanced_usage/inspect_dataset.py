import os
import argparse
from importlib import import_module
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skvideo.io import FFmpegWriter

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

    display_config = dict(road_buffer_size=1000)

    # Run
    for trial_i in tqdm.tqdm(range(args.n_trials)):

        def _seq_reset():
            return agent.done or dataset._snippet_i >= dataset.snippet_size  # TODO: hacky

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


if __name__ == '__main__':
    main()
