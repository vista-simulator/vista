import numpy as np


class StateDynamics(object):
    def __init__(self,
                 translational_state=0.0,
                 longitudinal_state=0.0,
                 theta_state=0.0):

        self.x_state = translational_state
        self.y_state = longitudinal_state
        self.theta_state = theta_state

    def reset(self):
        self.theta_state = 0  # used to build R matrix
        self.x_state = 0
        self.y_state = 0

    def step(self, curvature, velocity, delta_t):
        arc_length = abs(velocity * delta_t)
        theta = arc_length * curvature  # angle of traversed circle

        # Compute R
        self.theta_state += theta
        c = np.cos(self.theta_state)
        s = np.sin(self.theta_state)
        R = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])

        # Compute local x, y positions
        x = (1 - np.cos(theta)) / curvature
        y = np.sin(theta) / curvature

        # Transform positions from local to global
        R_2 = np.array([[c, -s], [s, c]])
        xy_local = np.array([[x], [y]])
        [[x_global], [y_global]] = np.matmul(R_2, xy_local)

        self.x_state += x_global
        self.y_state += y_global

        return self.x_state, self.y_state, self.theta_state

    def numpy(self):
        return np.array([self.x_state, self.y_state, self.theta_state])

    def copy(self):
        return StateDynamics(
            translational_state=self.x_state,
            longitudinal_state=self.y_state,
            theta_state=self.theta_state)

    def __repr__(self):
        return "<{}: [{}, {}, {}]>".format(self.__class__.__name__,
                                           self.x_state, self.y_state,
                                           self.theta_state)
