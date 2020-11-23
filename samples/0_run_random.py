import argparse
import numpy as np

import vista


def main(args):

    # Initialize the simulator
    sim = vista.Simulator(args.trace_path)

    # Main running loop
    while True:
        total_reward, done = reset(sim)

        while not done:
            # Sample a random steering curvature action
            a = np.random.randn() / 500.

            s, r, done, info = sim.step(a)
            total_reward += r

            if done:
                print("[CRASH] Total Reward is {:+0.2f}".format(total_reward))


def reset(sim):
    """ Convience reset function at before starting a new episode """
    sim.reset()
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
