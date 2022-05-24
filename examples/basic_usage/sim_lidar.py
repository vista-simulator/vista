import argparse
import numpy as np
import os
import cv2

import vista
from vista.utils import transform
from vista.entities.agents.Dynamics import tireangle2curvature


def main(args):
    world = vista.World(args.trace_path, trace_config={'road_width': 4})
    car = world.spawn_agent(
        config={
            'length': 5.,
            'width': 2.,
            'wheel_base': 2.78,
            'steering_ratio': 14.7,
            'lookahead_road': True
        })
    lidar_config = {
        'yaw_fov': (-180., 180.),
        'pitch_fov': (-21.0, 14.0)
    }
    lidar = car.spawn_lidar(lidar_config)
    display = vista.Display(world)

    world.reset()
    display.reset()

    while not car.done:
        action = follow_human_trajectory(car)
        car.step_dynamics(action)
        car.step_sensors()

        vis_img = display.render()
        cv2.imshow('Visualize LiDAR', vis_img[:, :, ::-1])
        cv2.waitKey(20)


def follow_human_trajectory(agent):
    action = np.array([
        agent.trace.f_curvature(agent.timestamp),
        agent.trace.f_speed(agent.timestamp)
    ])
    return action


if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser(
        description='Run the simulator with random actions')
    parser.add_argument('--trace-path',
                        type=str,
                        nargs='+',
                        help='Path to the traces to use for simulation')
    args = parser.parse_args()

    main(args)
