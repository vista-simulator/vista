import argparse
import numpy as np
import os
import sys

import vista

# Parse Arguments
parser = argparse.ArgumentParser(description='Run the simulator with random actions')
parser.add_argument('--trace-path', type=str, nargs='+', default='/home/amini/trace/20190205-102931_blue_prius_devens_rightside', help='Path to the traces to use for simulation')
args = parser.parse_args()


# Initialize the simulator
sim = vista.Simulator(args.trace_path)


# Convience reset function at before starting a new episode
def reset():
    print()
    sim.reset()
    total_reward = 0.0
    steps = 0.0
    done = False
    return total_reward, steps, done


# Main running loop
while True:
    total_reward, steps, done = reset()

    while not done:
        a = np.random.randn()/500. # Random steering curvature action

        s, r, done, info = sim.step(a)
        steps += 1
        total_reward += r

        if steps % 200 == 0 or done:
            print(("[CRASH] " if done else "")+"Step {}: total_reward is {:+0.2f}".format(steps, total_reward))
