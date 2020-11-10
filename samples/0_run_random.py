import argparse
import numpy as np
import os
import sys

import vista

# Parse Arguments
parser = argparse.ArgumentParser(description='Run the simulator with random actions')
parser.add_argument('--trace-path', type=str, nargs='+', help='Path to the traces to use for simulation')
args = parser.parse_args()


# Initialize the simulator
sim = vista.Simulator(args.trace_path)


# Convience reset function at before starting a new episode
def reset():
    sim.reset()
    total_reward = 0.0
    done = False
    return total_reward, done


# Main running loop
while True:
    total_reward, done = reset()

    while not done:
        # Sample a random steering curvature action
        a = np.random.randn()/500.

        s, r, done, info = sim.step(a)
        total_reward += r

        if done:
            print("[CRASH] Total Reward is {:+0.2f}".format(total_reward))
