import os
import cv2
import argparse
import pickle
import torch
import torch.nn as nn
from importlib import import_module
from skvideo.io import FFmpegWriter

import tools.utils as utils
import datasets.utils as d_utils

import vista
from vista.tasks import LaneFollowing
from vista.entities.sensors import Camera, Lidar, EventCamera
from vista.utils import logging
logging.setLevel(logging.ERROR)


def main():
    # Parse arguments and config
    parser = argparse.ArgumentParser(description='IL/GPL in Vista')
    parser.add_argument('--trace-paths',
                        type=str,
                        nargs='+',
                        required=True,
                        help='Path to the traces to use for simulation')
    parser.add_argument('--ckpt',
                        type=str,
                        default=None,
                        required=True,
                        help='Path to checkpoint')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='Disable gpu')
    parser.add_argument('--n-episodes',
                        type=int,
                        default=1,
                        help='Number of episodes to be tested')
    parser.add_argument('--save-video',
                        action='store_true',
                        default=False,
                        help='Whether to save video')
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

    # Define task in vista
    if config.dataset.type in ['il_rgb_dataset', 'gpl_rgb_dataset']:
        utils.set_dict_value_by_str(config, 'dataset:camera_config:type',
                                    'camera')
        utils.set_dict_value_by_str(config,
                                    'dataset:camera_config:use_synthesizer',
                                    True)
        sensors_configs = [config.dataset.camera_config]
    elif config.dataset.type in ['gpl_lidar_dataset']:
        utils.set_dict_value_by_str(config, 'dataset:lidar_config:type',
                                    'lidar')
        sensors_configs = [config.dataset.lidar_config]
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
                    data['lidar'] = v[None, ...].to(device)
                elif isinstance(sensor, EventCamera):
                    raise NotImplementedError
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
                cv2.imshow('vis', vis_img)
                cv2.waitKey(1)
                video_writer.writeFrame(vis_img)


if __name__ == '__main__':
    main()
