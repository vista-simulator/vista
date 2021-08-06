import argparse
import numpy as np
from pynput import keyboard
import cv2

import vista
from vista.entities.sensors.camera_utils.ViewSynthesis import DepthModes
from vista.utils import logging

logging.setLevel(logging.DEBUG)


# Setup the keyboard to listen for events
# Press the left and right arrow keys to turn.
steering_curvature = 0.0


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
        steering_ratio=14.7, #17.6,
    )
    camera_config1 = dict(
        # camera params
        name='camera_front',
        rig_path='~/data/traces/20200424-133758_blue_prius_cambridge_rain/RIG.xml',
        size=(250, 400),
        # rendering params
        depth_mode=DepthModes.FIXED_PLANE,
        use_lighting=False,
    )
    display_config = dict(
        road_buffer_size=1000,
    )
    world = vista.World(args.trace_path, trace_config)
    agent = world.spawn_agent(car_config)
    camera1 = agent.spawn_camera(camera_config1)
    display = vista.Display(world, display_config=display_config)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Main running loop
    stop_sim = False
    while not stop_sim:
        world.reset()
        display.reset()

        while not agent.done and not stop_sim:
            steering_curvature = 0.01 # DEBUG
            action = np.array([steering_curvature,
                               agent.trace.f_speed(agent.timestamp)])
            agent.step_dynamics(action)
            agent.step_sensors()
            logging.info(steering_curvature)

            img = display.render()

            cv2.imshow('VISTA', img[:,:,::-1])
            key = cv2.waitKey(20)
            if key == ord('q'):
                stop_sim = True
                break
            elif key == ord('r'):
                break


def on_press(key):
    """ Keyboard listener to update the global steering curvature variable """
    global steering_curvature

    if key == keyboard.Key.left:
        steering_curvature += 1e-3
    elif key == keyboard.Key.right:
        steering_curvature -= 1e-3


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
