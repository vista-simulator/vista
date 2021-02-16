import numpy as np
import os
from scipy.interpolate import interp1d
import sys

# VISTA imports
from .StateDynamics import StateDynamics
from .Memos import State
from .util import TopicNames


class CarDynamics:
    def __init__(self, world):
        ''' Initializes object to handle agent dynamics.
            Inputs:
                world -- initialized World object: contains trace path and handles frames
                theta_state -- agent theta position
                x_state -- agent x position
                y_state -- agent y position
        '''
        self.world = world
        self.f_speed, self.f_distance, self.f_curvature = self.interpolate(
            self.world.trace_path)

        self.relative_state = State(
            translation_x=0.0, translation_y=0.0, theta=0.0)
        self.human_dynamics = StateDynamics()
        self.ego_dynamics = StateDynamics()

        # Vehicle parameters
        self.wheel_base = 2.8
        self.steering_ratio = 17.6

    def interpolate(self, trace_path):
        ''' Interpolation functions for speed, distance, omega, and curvature.
            Returns speed, distance, omega, and curvature.
        '''
        speed = np.genfromtxt(
            os.path.join(trace_path, TopicNames.speed + '.csv'), delimiter=',')
        f_speed = interp1d(speed[:, 0], speed[:, 1], fill_value='extrapolate')

        distance = np.genfromtxt(
            os.path.join(trace_path, TopicNames.distance + '.csv'),
            delimiter=',')
        f_distance = interp1d(
            distance[:, 0], distance[:, 1], fill_value='extrapolate')

        # Human inverse_r from filtered odometry
        odometry = np.genfromtxt(
            os.path.join(trace_path, TopicNames.odometry + '.csv'),
            delimiter=',')
        f_omega = interp1d(
            odometry[:, 0], odometry[:, 4], fill_value='extrapolate')
        sample_times = odometry[:, 0]
        curvature = odometry[:, 4] / np.maximum(f_speed(sample_times), 1e-10)
        good_curvature_inds = np.abs(curvature) < 1 / 3.
        f_curvature = interp1d(
            sample_times[good_curvature_inds],
            curvature[good_curvature_inds],
            fill_value='extrapolate')

        return f_speed, f_distance, f_curvature

    def step(self, curvature, velocity, delta_t):

        current_timestamp = self.world.get_current_timestamp()
        model_curvature = curvature + 1e-11  # Avoid dividing by 0
        model_velocity = self.f_speed(
            current_timestamp) if velocity is None else velocity

        self.ego_dynamics.step(model_curvature, model_velocity, delta_t)

        next_valid_timestamp, self.human_dynamics = \
            self.world.get_next_valid_timestamp(
                human=self.human_dynamics, desired_ego=self.ego_dynamics)

        translation_x, translation_y, theta = self.compute_relative_transform()
        self.relative_state.update(translation_x, translation_y, theta)

        info = dict()
        info["timestamp"] = next_valid_timestamp
        info["first_time"] = self.world.first_time
        info["human_curvature"] = self.f_curvature(next_valid_timestamp)
        info["human_velocity"] = self.f_speed(next_valid_timestamp)
        info["model_curvature"] = model_curvature
        info["model_velocity"] = model_velocity
        info["model_angle"] = self.curvature_to_steering(model_curvature)
        info["rotation"] = self.relative_state.theta
        info["translation"] = self.relative_state.translation_x
        info["distance"] = self.f_distance(next_valid_timestamp) - \
            self.f_distance(self.world.first_time)
        info["done"] = self.world.isCrashed

        return info

    def compute_relative_transform(self):
        """ TODO
        """

        ego_x_state, ego_y_state, ego_theta_state = \
            self.ego_dynamics.numpy()
        human_x_state, human_y_state, human_theta_state = \
            self.human_dynamics.numpy()

        c = np.cos(human_theta_state)
        s = np.sin(human_theta_state)
        R_2 = np.array([[c, -s], [s, c]])
        xy_global_centered = np.array([[ego_x_state - human_x_state],
                                       [human_y_state - ego_y_state]])
        [[translation_x], [translation_y]] = np.matmul(R_2, xy_global_centered)
        translation_y *= -1  # negate the longitudinal translation (due to VS setup)

        # Adjust based on what the human did
        theta = ego_theta_state - human_theta_state

        # Check if crashed
        free_width = self.world.road_width - self.world.car_width
        max_translation = abs(translation_x) > free_width / 2.
        max_rotation = abs(theta) > np.pi / 10.
        if max_translation or max_rotation:
            self.world.isCrashed = True

        return translation_x, translation_y, theta

    def curvature_to_steering(self, curvature):
        tire_angle = np.arctan(self.wheel_base * curvature)
        angle = tire_angle * self.steering_ratio * 180. / np.pi
        return angle

    def reset(self):
        # print "RESETTING CAR"

        self.world.reset()
        self.relative_state.reset()
        self.ego_dynamics.reset()
        self.human_dynamics.reset()
