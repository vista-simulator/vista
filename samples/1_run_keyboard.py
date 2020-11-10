import argparse
import signal
from pynput import keyboard # listen to keyboard !

import vista


# Parse Arguments
parser = argparse.ArgumentParser(description='Run the simulator with random actions')
parser.add_argument('--trace-path', type=str, nargs='+', help='Path to the traces to use for simulation')
args = parser.parse_args()


# Initialize the simulator
sim = vista.Simulator(args.trace_path)


# Setup the keyboard to listen for events
# Press the left and right arrow keys to turn.
steering_curvature = 0.0
def on_press(key):
    global steering_curvature

    if key == keyboard.Key.left:
        steering_curvature += 1e-3
    elif key == keyboard.Key.right:
        steering_curvature -= 1e-3

listener = keyboard.Listener(on_press=on_press)
listener.start()


# Convience reset function at before starting a new episode
def reset():
    global steering_curvature
    steering_curvature = 0.0

    sim.reset()
    total_reward = 0.0
    done = False
    return total_reward, done


# Main running loop
while True:
    total_reward, done = reset()

    while not done:
        a = steering_curvature

        s, r, done, info = sim.step(a)
        total_reward += r

        if done:
            print("[CRASH] Total Reward is {:+0.2f}".format(total_reward))
