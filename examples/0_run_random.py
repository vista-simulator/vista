import argparse
import numpy as np

import vista
from vista.entities.sensors.camera_utils.ViewSynthesis import DepthModes
from vista.utils import logging

logging.setLevel(logging.DEBUG)


def main(args):
    # Initialize the simulator
    car_config = dict(
        length=5.,
        width=2.,
        wheel_base=2.8,
        steering_ratio=17.6,
    )
    camera_config = dict(
        # camera params
        name='camera_front',
        rig_path='~/data/traces/20200424-133758_blue_prius_cambridge_rain/RIG.xml',
        size=(250, 400),
        # rendering params
        depth_mode=DepthModes.FIXED_PLANE,
        use_lighting=False,
    )
    world = vista.World(args.trace_path)
    agent = world.spawn_agent(car_config)
    camera = agent.spawn_camera(camera_config)
    display = vista.Display(world)

    # Main running loop
    while True:
        world.reset()
        display.reset()

        while not agent.done:
            action = np.array([agent.trace.f_curvature(agent.timestamp), 
                               agent.trace.f_speed(agent.timestamp)])
            agent.step_dynamics(action)
            agent.step_sensors()

            # print(agent.timestamp, agent.frame_index, agent.frame_number)
            # import cv2; cv2.imwrite('test.png', agent.observations['camera_front'])
            # import pdb; pdb.set_trace()


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
