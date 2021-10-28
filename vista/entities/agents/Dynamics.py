from typing import Optional, List
import numpy as np
import scipy.integrate as ode_solve

from ...utils import logging


class State:
    """ Vehicle state without steering angle and velocity. """
    def __init__(self,
                 x: Optional[float] = 0.,
                 y: Optional[float] = 0.,
                 yaw: Optional[float] = 0.) -> None:
        self.update(x, y, yaw)

    def update(self, x: float, y: float, yaw: float) -> None:
        """ Set values. """
        self._x = x
        self._y = y
        self._yaw = yaw

    def reset(self) -> None:
        """ Reset to zeros. """
        self.update(0., 0., 0.)

    def numpy(self) -> np.ndarray:
        """ Convert to numpy. """
        return np.array([self._x, self._y, self._yaw])

    @property
    def x(self) -> float:
        """ Position of the car in x-axis. """
        return self._x

    @property
    def y(self) -> float:
        """ Position of the car in y-axis. """
        return self._y

    @property
    def yaw(self) -> float:
        """ Heading/yaw of the car. """
        return self._yaw

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}: ' + \
               f'[{self.x}, {self.y}, {self.yaw}]>'


class StateDynamics:
    """ Simple continuous kinematic (bicycle) model of a rear-wheel driven vehicle.
    Check out Eq.3 in https://www.autonomousrobots.nl/docs/17-schwartig-autonomy-icra.pdf

    Args:
        x (float): Position of the car in x-axis.
        y (float): Position of the car in y-axis.
        yaw (float): Heading/yaw of the car.
        steering (float): Steering angle of tires instead of steering wheel.
        speed (float): Forward speed.
        steering_bound (list): Upper and lower bound of steering angle.
        speed_bound (list): Upper and lower bound of speed.
        wheel_base(float): Wheel base.

    """
    def __init__(self,
                 x: Optional[float] = 0.,
                 y: Optional[float] = 0.,
                 yaw: Optional[float] = 0.,
                 steering: Optional[float] = 0.,
                 speed: Optional[float] = 0.,
                 steering_bound: Optional[List[float]] = [-0.75, 0.75],
                 speed_bound: Optional[List[float]] = [0., 15.],
                 wheel_base: Optional[float] = 2.8) -> None:
        self._x = 0.
        self._y = 0.
        self._yaw = 0.
        self._steering = 0.
        self._speed = 0.

        self.update(x, y, yaw, steering, speed)
        self._steering_bound = steering_bound
        self._speed_bound = speed_bound
        self._wheel_base = wheel_base

    def step(self,
             steering_velocity: float,
             acceleration: float,
             dt: float,
             max_steps: Optional[int] = 100) -> np.ndarray:
        """ Step the vehicle state by solving ODE of the continuous kinematic (bicycle) model.

        Args:
            steering_velocity (float): Steering velocity (of tire angle).
            acceleration (float): Acceleration.
            dt (float): Elapsed time.
            max_steps (int): Maximal step to run the ODE solver; default to 100.

        Returns:
            np.ndarray: The updated state (x, y, yaw, tire angle, speed).

        """

        # Define dynamics
        def _ode_func(t, z):
            _x, _y, _phi, _delta, _v = z
            u_delta = steering_velocity
            u_a = acceleration
            new_z = np.array([
                -_v * np.sin(_phi),  # swap x-y axis with sign change
                _v * np.cos(_phi),
                _v / self._wheel_base * np.tan(_delta),
                u_delta,
                u_a
            ])
            return new_z

        # Solve ODE
        z_0 = np.array(
            [self._x, self._y, self._yaw, self._steering, self._speed])
        solver = ode_solve.RK45(_ode_func, 0., z_0, dt)
        steps = 0
        while solver.status is 'running' and steps <= max_steps:
            solver.step()
            steps += 1
        if (dt - solver.t) < 0:
            logging.error('Reach max steps {} without reaching t_bound ({} < {})'.format( \
                max_steps, solver.t, solver.t_bound))

        self._x, self._y, self._yaw, self._steering, self._speed = solver.y

        # Clip by value bounds
        self._steering = np.clip(self._steering, *self._steering_bound)
        self._speed = np.clip(self._speed, *self._speed_bound)

        return self.numpy()

    def numpy(self) -> np.ndarray:
        """ Return a numpy array of vehicle state.

        Returns:
            np.ndarray: Vehicle state.

        """
        return np.array(
            [self._x, self._y, self._yaw, self._steering, self._speed])

    def copy(self):
        """ Create a copy. """
        return StateDynamics(x=self._x,
                             y=self._y,
                             yaw=self._yaw,
                             steering=self._steering,
                             speed=self._speed)

    def update(self, x: float, y: float, yaw: float, steering: float,
               speed: float) -> None:
        """ Set state.
        Args:
            x (float): Position of the car in x-axis.
            y (float): Position of the car in y-axis.
            yaw (float): Heading/yaw of the car.
            steering (float): Steering angle of tires instead of steering wheel.
            speed (float): Forward speed.

        """
        self._x = x
        self._y = y
        self._yaw = yaw
        self._steering = steering
        self._speed = speed

    def reset(self) -> None:
        """ Reset all state to zeros. """
        self.update(0., 0., 0., 0., 0.)

    @property
    def x(self) -> float:
        """ Position of the car in x-axis. """
        return self._x

    @property
    def y(self) -> float:
        """ Position of the car in y-axis. """
        return self._y

    @property
    def yaw(self) -> float:
        """ Heading/yaw of the car. """
        return self._yaw

    @property
    def steering(self) -> float:
        """ Steering (tire) angle. """
        return self._steering

    @property
    def steering_bound(self) -> List[float]:
        """ Lower and upper bound of steering (tire) angle. """
        return self._steering_bound

    @property
    def speed(self) -> float:
        """ Speed. """
        return self._speed

    @property
    def speed_bound(self) -> List[float]:
        """ Lower and upper bound of speed. """
        return self._speed_bound

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}: ' + \
            f'[{self.x}, {self.y}, {self.y}, {self.steering}, {self.speed}]>'


def curvature2tireangle(curvature: float, wheel_base: float) -> float:
    """ Convert curvature to tire angle.

    Args:
        curvature (float): Curvature.
        wheel_base (float): Wheel base.

    Returns:
        float: Tire angle.

    """
    return np.arctan(wheel_base * curvature)


def tireangle2curvature(tire_angle: float, wheel_base: float) -> float:
    """ Convert tire angel to curvature.

    Args:
        tire_angle (float): Tire angle.
        wheel_base (float): Wheel base.

    Returns:
        float: Curvature.

    """
    return np.tan(tire_angle) / wheel_base


def curvature2steering(curvature: float, wheel_base: float,
                       steering_ratio: float) -> float:
    """ Convert curvature to steering angle.

    Args:
        curvature (float): Curvature.
        wheel_base (float): Wheel base.
        steering_ratio (float): Steering ratio.

    Returns:
        float: Steering (wheel) angle.

    """
    tire_angle = curvature2tireangle(curvature, wheel_base)
    steering = tire_angle * steering_ratio * 180. / np.pi

    return steering


def steering2curvature(steering: float, wheel_base: float,
                       steering_ratio: float) -> float:
    """ Convert steering angle to curvature.

    Args:
        steering (float): Steering (wheel) angle.
        wheel_base (float): Wheel base.
        steering_ratio (float): Steering ratio.

    Returns:
        float: Curvature.

    """
    tire_angle = steering * (np.pi / 180.) / steering_ratio
    curvature = tireangle2curvature(tire_angle, wheel_base)

    return curvature


def update_with_perfect_controller(desired_state: List[float], dt: float,
                                   dynamics: StateDynamics) -> None:
    """ Update vehicle state assuming a perfect low-level controller. This
    basically simulate that the low-level controller can "instantaneously"
    achieve the desired state (e.g., steering (tire) angle and speed) with control
    command (steering velocity and accleration) in vehicle dynamics.

    Args:
        desired_state (List[float]): A list of desired vehicle state.
        dt (float): Elapsed time.
        dynamics (StateDynamics): Vehicle state.

    """
    # simulate condition when the desired state can be instantaneously achieved
    new_dyn = dynamics.numpy()
    new_dyn[-2:] = desired_state
    dynamics.update(*new_dyn)
    dynamics.step(0., 0., dt)
