import argparse
import numpy as np
import os
import copy

import vista
from vista.entities.sensors.camera_utils.ViewSynthesis import DepthModes
from vista.utils import logging
from vista.tasks import MultiAgentBase


def main(args):
    # Initialize the simulator
    trace_config = dict(
        road_width=4,
        reset_mode='default',
        master_sensor='camera_front',
    )
    car_config = dict(
        length=5.,
        width=2.,
        wheel_base=2.78,
        steering_ratio=14.7,
    )
    examples_path = os.path.dirname(os.path.realpath(__file__))
    sensors_config = [
        dict(
            type='camera',
            # camera params
            name='camera_front',
            rig_path=os.path.join(examples_path, "RIG.xml"),
            size=(200, 320),
            # rendering params
            depth_mode=DepthModes.FIXED_PLANE,
            use_lighting=False,
        )
    ]
    task_config = dict(n_agents=2,
                       mesh_dir=args.mesh_dir,
                       init_dist_range=[6., 6.],
                       init_lat_noise_range=[0., 0.])
    display_config = dict(road_buffer_size=1000, )

    ego_car_config = copy.deepcopy(car_config)
    ego_car_config['lookahead_road'] = True
    env = MultiAgentBase(trace_paths=args.trace_paths,
                         trace_config=trace_config,
                         car_configs=[ego_car_config, car_config],
                         sensors_configs=[sensors_config, []],
                         task_config=task_config,
                         logging_level='DEBUG')
    display = vista.Display(env.world, display_config=display_config)

    # Run
    env.reset()
    display.reset()
    done = False
    while not done:
        actions = generate_human_actions(env.world)
        observations, rewards, dones, infos = env.step(actions)
        done = np.any(list(dones.values()))

        img = display.render()
        # import cv2
        # cv2.imshow("test", img[:, :, ::-1])
        # cv2.waitKey(0)


def generate_human_actions(world):
    actions = dict()
    for agent in world.agents:
        actions[agent.id] = np.array([
            agent.trace.f_curvature(agent.timestamp),
            agent.trace.f_speed(agent.timestamp)
        ])
    return actions


if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser(
        description='Run the simulator with random actions')
    parser.add_argument('--trace-paths',
                        type=str,
                        nargs='+',
                        required=True,
                        help='Path to the traces to use for simulation')
    parser.add_argument('--mesh-dir',
                        type=str,
                        default=None,
                        help='Directory of meshes for virtual agents')
    args = parser.parse_args()

    main(args)
