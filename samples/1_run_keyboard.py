import argparse
from pynput import keyboard  # listen to keyboard !

import vista

# Setup the keyboard to listen for events
# Press the left and right arrow keys to turn.
steering_curvature = 0.0


def main(args):

    # Initialize the simulator
    world = vista.World(args.trace_path)
    agent = world.spawn_agent()
    camera = agent.spawn_camera()

    # Create a graphical display
    display = vista.Display(world)

    # Main running loop
    while True:
        total_reward, done = reset(world)

        while not done:
            state, done = agent.step(steering_curvature)
            display.render()

            total_reward += 1.0

            if done:
                print("[CRASH] Total Reward is {:+0.2f}".format(total_reward))


def on_press(key):
    """ Keyboard listener to update the global steering curvature variable """
    global steering_curvature

    if key == keyboard.Key.left:
        steering_curvature += 1e-3
    elif key == keyboard.Key.right:
        steering_curvature -= 1e-3


listener = keyboard.Listener(on_press=on_press)
listener.start()


def reset(world):
    """ Convience reset function at before starting a new episode """
    global steering_curvature
    steering_curvature = 0.0

    world.reset()
    total_reward = 0.0
    done = False
    return total_reward, done


if __name__ == "__main__":
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
