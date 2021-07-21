import argparse
import numpy as np

import vista
from vista.entities.sensors.camera_utils.ViewSynthesis import DepthModes
from vista.utils import logging

logging.setLevel(logging.DEBUG)


def main(args):
    # Initialize the simulator
    trace_config = dict(
        road_width=4,
        reset_mode='default',
        master_sensor='lidar_3d',
    )
    car_config = dict(
        length=5.,
        width=2.,
        wheel_base=2.8,
        steering_ratio=17.6,
    )
    lidar_config = dict(
        name='lidar_3d',
    )
    display_config = dict(
        road_buffer_size=1000,
    )
    world = vista.World(args.trace_path, trace_config)
    agent = world.spawn_agent(car_config)
    lidar = agent.spawn_lidar(lidar_config)
    display = vista.Display(world, display_config=display_config)

    # Main running loop
    while True:
        world.reset()
        display.reset()

        while not agent.done:
            action = np.array([agent.trace.f_curvature(agent.timestamp), 
                               agent.trace.f_speed(agent.timestamp)])
            agent.step_dynamics(action)
            agent.step_sensors()

            img = display.render()
            ### DEBUG
            logging.warning('Dump image for debugging and set pdb')
            import cv2; cv2.imwrite('test.png', img[:,:,::-1])
            import pdb; pdb.set_trace()
            ### DEBUG


if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser(
        description='Run the simulator with random actions')
    parser.add_argument(
        '--trace-path',
        type=str,
        nargs='+',
        help='Path to the traces to use for simulation')
    args = parser.parse_args()

    main(args)
