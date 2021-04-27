import argparse
import numpy as np

import vista


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
            # Sample a random steering curvature action
            action = np.random.randn() / 500.

            state, done = agent.step(action)
            display.render()

            total_reward += 1.0

            if done:
                print("[CRASH] Total Reward is {:+0.2f}".format(total_reward))


def reset(world):
    """ Convience reset function at before starting a new episode """
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
