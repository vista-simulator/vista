import argparse
import numpy as np
import os
import sys

deepknight_root = os.environ.get('DEEPKNIGHT_ROOT')
sys.path.insert(0, os.path.join(deepknight_root, 'simulator'))
from gym_deepknight.envs import DeepknightEnv

# Parse Arguments
parser = argparse.ArgumentParser(description='Run the simulator example')
parser.add_argument('--trace-path', type=str, nargs='+', default='/mnt/deepknight/trace/20190205-102931_blue_prius_devens_rightside', help='Path to the traces to use for simulation')
parser.add_argument('--joystick', action='store_false', help='Control with joystick')
args = parser.parse_args()

use_joystick = args.joystick
env_path = args.trace_path

# Environment creation
env = DeepknightEnv(env_path, (500,800))

if use_joystick:
    sys.path.insert(0, os.path.join(deepknight_root, 'simulator/gym_deepknight/envs/util/hci'))
    from Logitech import LogitechDriver

    device = LogitechDriver()
    def get_joystick_curvature():
        cur_angle = device.get_steering()
        angle_radians = cur_angle * np.pi/180.
        curvature = -np.tan(angle_radians/17.6) / 2.8
        return curvature

    def get_joystick_speed():
        throttle = device.get_throttle() #[0,1]
        speed = throttle * 30
        return speed

while True:
    print("ROLLOUT RESET")
    env.reset()
    total_reward = 0.0
    steps = 0
    done = False
    while not done:
        if use_joystick:
            # a = [get_joystick_curvature(), get_joystick_speed()]
            a = [get_joystick_curvature()]
        else:
            a = np.random.randn()/500. # Random action

        s, r, done, info = env.step(a)
        total_reward += r
        steps += 1
        if steps % 200 == 0 or done:
            print(("[CRASH] " if done else "")+"Step {}: total_reward is {:+0.2f}".format(steps, total_reward))
            # Add env.render?
env.close()
