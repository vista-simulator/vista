import argparse
import numpy as np

import vista
from vista.entities.sensors.camera_utils.ViewSynthesis import DepthModes
from vista.utils import logging

logging.setLevel(logging.ERROR)


def main(args):
    # Initialize the simulator
    trace_config = dict(
        road_width=4,
        reset_mode='segment_start',
        master_sensor='front_center',
        # master_sensor='camera_front',
    )
    car_config = dict(
        length=5.,
        width=2.,
        wheel_base=2.78,
        steering_ratio=14.7,
    )
    camera_config1 = dict(
        # camera params
        name='front_center',
        # name='camera_front',
        rig_path='~/data/traces/20200424-133758_blue_prius_cambridge_rain/RIG.xml',
        # rig_path='~/projects/deepknight/gmsl-rig/RIG.xml',
        size=(200, 320),
        # rendering params
        depth_mode=DepthModes.FIXED_PLANE,
        use_lighting=False,
        use_synthesizer=False, # NOTE: don't do view synthesis
    )

    display_config = dict(
        road_buffer_size=1000,
    )
    world = vista.World(args.trace_path, trace_config)
    agent = world.spawn_agent(car_config)
    camera1 = agent.spawn_camera(camera_config1)
    display = vista.Display(world, display_config=display_config)

    # Main running loop
    while True:
        world.reset()
        display.reset()

        while not agent.done:
            agent.step_dataset(step_dynamics=False)

            print(agent.frame_index, agent.human_curvature, agent.speed)
            # camera = agent.observations["camera_front"]

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
