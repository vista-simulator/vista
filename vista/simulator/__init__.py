import copy
import cv2
import gym
import numpy as np
import os
import sys
import tensorflow as tf
import time


# VISTA imports
from .World import World
from .CarDynamics import CarDynamics
from .util import drawing as draw
from .util import Camera
from .util import print_util as P
from .util import ViewSynthesis
from .util import TopicNames


# Configuration type variables
LOWER_CURVATURE_BOUND = -1/3.
UPPER_CURVATURE_BOUND = 1/3.
FPS = 30
GUI_FAME_NAME = "Deepknight Simulator"
WINDOW_SIZE = (500, 330, 3)
DRAWING_COORDS = {
    draw.Module.STEERING: draw.Box((.1,.75), (.3,1)),
    draw.Module.LANE:     draw.Box((.65,.7), (1,1)),
    draw.Module.INFO:     draw.Box((.6,0), (1,.7)),
    draw.Module.SPEED:    draw.Box((.3,.8), (.6,.95)),
    draw.Module.FRAME:    draw.Box((0,0), (.6,.7))
}


class Simulator(gym.Env):
    metadata = {
        'render.modes': ['gui', 'text'],
        'video.frames_per_second': FPS}

    def __init__(self, env_path, obs_size=None):
        # ENVIRONMENT SETUP

        # Create list of worlds with same tf session
        if isinstance(env_path, str):
            env_path = [env_path]

        self.worlds = []
        self.cars = []

        self.sess = tf.Session()
        self.camera = Camera("camera_front")
        if obs_size is None:
            obs_size = (self.camera.get_height(), self.camera.get_width())
        self.camera.resize(*obs_size)
        VS = ViewSynthesis(self.camera, sess=self.sess).get_as_py_func()

        for env in env_path:
            world = World(env, self.sess, TopicNames.camera_front, self.camera, VS)
            car = CarDynamics(world)
            self.worlds.append(world)
            self.cars.append(car)

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(low=LOWER_CURVATURE_BOUND, high=UPPER_CURVATURE_BOUND, shape=(1,), dtype=np.float64) #TODO: add speed as part of the action space?
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(obs_size[0], obs_size[1], 3), dtype=np.uint8)

        # Viewing
        self.viewer = None
        self.last_gui_update = time.time()

        # Rewards
        self.reward = 0.0

        self.seed() # TODO: use seed for reseting envs/traces
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        # print "RESETTING ENV"

        # Choose which car to reset
        self.index = self.find_env_reset()
        self.world = self.worlds[self.index]
        self.car = self.cars[self.index]

        self.car.reset()

        self.reward = 0.0
        # TODO? some flag that reset has been called
        # set human render to false? Use a default mode?

        starting_timestamp, self.car.human_dynamics = self.world.get_next_valid_timestamp(
            human=self.car.human_dynamics,
            desired_ego=self.car.ego_dynamics)
        frame = self.world.get_frame_from_time(starting_timestamp)

        return frame

    def step(self, action):
        action = np.array(action).reshape(-1,) # force action to be column vector
        curvature = action[0]
        velocity =  action[1] if action.shape[0]>1 else None

        info = self.car.step(curvature, velocity, delta_t=1./FPS)

        frame = self.world.get_frame_from_time(info['timestamp'])
        translated_frame = self.world.synthesize_frame(frame, info, self.car.relative_state)

        self.render(translated_frame, info)

        done = self.world.isCrashed

        step_reward = 1.0 if not done else 0.0 # simple reward function +1 if not crashed
        self.reward += step_reward

        if self.world.trace_done:
            translated_frame = self.reset()

        return translated_frame, step_reward, done, info

    def render(self, frame, info, mode='gui'):
        if mode=='gui':
            if self.viewer is None: # viewer is not yet created, so make the window
                cv2.namedWindow(GUI_FAME_NAME, cv2.WINDOW_NORMAL)
                cv2.moveWindow(GUI_FAME_NAME, 250, 0)
                cv2.resizeWindow(GUI_FAME_NAME, WINDOW_SIZE[0], WINDOW_SIZE[1])
                self.viewer = True

            window = self.draw_gui(frame, info)
            cv2.imshow(GUI_FAME_NAME, window)
            gui_delta_t = time.time() - self.last_gui_update
            timeout_ms = 1000 * max(1/1000., 1.0/FPS - gui_delta_t)
            # print "timeout: {}".format(timeout_ms)
            if cv2.waitKey(int(timeout_ms)) == ord(' '):
                print("pause")
                cv2.waitKey()
            self.last_gui_update = time.time()
        else:
            # Print text representation of sim with terminal
            print(self.get_text_representation())

    def draw_gui(self, frame, info):
        """ Creates the GUI window for the simulator.

        This creates multiple modules that can be placed anywhere in the overall frame. Within each module, items can be
        placed in the reference frame of the specific module. These function are held in drawing.py

        Args:
            frame (np array): The altered frame from ViewSynthesis
            info (dict): the dictionary holding the current timestep's kinematic data

        Returns:
            (np array): the fuse of all the modules into the full GUI screen
        """

        frame = copy.copy(frame)

        # Create each module, feed them into frame
        coordinates_list = [DRAWING_COORDS[module] for module in [draw.Module.STEERING, draw.Module.LANE, draw.Module.INFO, draw.Module.SPEED, draw.Module.FRAME]]

        steering_module = draw.draw_steering_wheel(
            WINDOW_SIZE,
            DRAWING_COORDS[draw.Module.STEERING],
            info['model_angle'])

        mini_car_on_road_module = draw.draw_mini_car_on_road(
            WINDOW_SIZE,
            DRAWING_COORDS[draw.Module.LANE],
            info['translation'],
            info['rotation'],
            self.world.road_width)

        info_module = draw.draw_info_sidebar(
            WINDOW_SIZE,
            DRAWING_COORDS[draw.Module.INFO],
            info)

        speed_module = draw.draw_speedometer(
            WINDOW_SIZE,
            DRAWING_COORDS[draw.Module.SPEED],
            model=2.23*info['human_velocity'],
            human=2.23*info['model_velocity'])

        frame_module = draw.draw_frame(
            WINDOW_SIZE,
            DRAWING_COORDS[draw.Module.FRAME],
            frame,
            self.world.camera,
            info['model_curvature'],
            info['human_curvature'])

        return draw.fuse(WINDOW_SIZE, [steering_module, mini_car_on_road_module, info_module, speed_module, frame_module], coordinates_list, draw_outline=False)

    def get_text_representation(self):
        """Runs the simulator with a text representation rather than draw_gui()

        Instead of drawing the simulator GUI, a simple text representation is run in the terminal.
        The car is represented by the * character, and the lane markers by the | character.
        This is called by running the program as --headless, or by setting self.headless_mode to True.

        Returns:
            terminal_road (str): Current timestep's car-road representation of this form: |      *      |

        """
        slots = 50
        terminal_road = "|" + " "*slots + "|" # 50 spaces inside to use for car placement
        loc = int((float(self.car.position[1]/self.world.road_width + 0.5) * slots) + 1) # Plus one for this character *
        if loc > slots:
            loc = slots
        if loc < 1:
            loc = 1
        terminal_road = terminal_road[0:loc] + str('*') + terminal_road[loc:]
        return terminal_road

    def find_env_reset(self):
        env_reset_probs = np.zeros(len(self.worlds))
        for i in range(len(self.worlds)):
            env = self.worlds[i]
            env_reset_probs[i] = env.num_of_frames
        env_reset_probs /= np.sum(env_reset_probs)
        new_env = np.random.choice(env_reset_probs.shape[0], 1, p=env_reset_probs)
        return new_env[0]

    def close(self):
        # End video stream
        self.sess.close()
        self.viewer = None
        for world in self.worlds:
            world.stream.close()
