from typing import List, Dict, Any, Optional
import numpy as np

from .Dynamics import State, StateDynamics, curvature2steering, curvature2tireangle, tireangle2curvature
from ..Entity import Entity

from ..sensors import BaseSensor, Camera
from ...core import World, Trace
from ...utils import transform


class Car(Entity):
    def __init__(self, world: World, car_config: Dict) -> None:
        """ Instantiate a Car object.
        
        Args:
            world (World): the world that this agent lives in
            car_config (Dict): configuration of the car
        """
        super(Car, self).__init__()

        # Pointer to the parent vista.World object where the agent lives
        self._parent: World = world

        # The trace to be associated with, updated every reset
        self._trace: Trace = None

        # A list of sensors attached to this agent (List[vista.Sensor]).
        self._sensors: List[BaseSensor] = []

        # State dynamics for tracking the virtual and human agent
        self._relative_state: State = State()
        self._ego_dynamics: StateDynamics = StateDynamics()
        self._human_dynamics: StateDynamics = StateDynamics()

        # Properties of a car
        self._length: float = car_config['length']
        self._width: float = car_config['width']
        self._wheel_base: float = car_config['wheel_base']
        self._steering_ratio: float = car_config['steering_ratio']
        self._speed: float = 0.
        self._curvature: float = 0.
        self._steering: float = 0.
        self._tire_angle: float = 0.
        self._human_speed: float = 0.
        self._human_curvature: float = 0.
        self._human_steering: float = 0.
        self._human_tire_angle: float = 0.
        self._timestamp: float = 0.
        self._frame_number: int = 0
        self._trace_index: int = 0
        self._segment_index: int = 0
        self._frame_index: int = 0
        self._observations: Dict[str, Any] = dict()

    def spawn_camera(self, cam_config: Dict) -> Camera:
        """ Spawn and attach a camera to this car.

        Args:
            cam_config (Dict): configuration for camera and rendering
        
        Returns:
            Camera: a vista camera sensor object spawned
        """
        cam = Camera(attach_to=self, config=cam_config)
        self._sensors.append(cam)

        return cam

    def reset(self, trace_index: int, segment_index: int, frame_index: int):
        # Update pointers to dataset
        self._trace = self.parent.traces[self.trace_index]
        self._timestamp = self.trace.get_master_timestamp(segment_index, frame_index)
        self._frame_number = self.trace.get_master_frame_number(segment_index, frame_index)
        self._trace_index = trace_index
        self._segment_index = segment_index
        self._frame_index = frame_index

        # Reset states and dynamics
        self._human_speed = self.trace.f_speed(self.timestamp)
        self._human_curvature = self.trace.f_curvature(self.timestamp)
        self._human_steering = curvature2steering(self.human_curvature,
                                                  self.wheel_base,
                                                  self.steering_ratio)
        self._human_tire_angle = curvature2tireangle(self.human_curvature, self.wheel_base)
        self._speed = self.human_speed
        self._curvature = self.human_curvature
        self._steering = self.human_steering
        self._tire_angle = curvature2tireangle(self.curvature, self.wheel_base)

        self.relative_state.reset()
        self.human_dynamics.update(0., 0., 0., self.human_tire_angle, self.human_speed)
        self.ego_dynamics.update(0., 0., 0., self.tire_angle, self.speed)

        # Reset sensors
        for sensor in self.sensors:
            sensor.reset()

        # Update observation
        self.step_sensors()

    def step_dynamics(self, action: np.ndarray, dt: Optional[float] = 1 / 30.):
        # Parse action
        action = np.array(action).reshape(-1)
        assert action.shape[0] == 2
        desired_curvature, desired_speed = action
        desired_tire_angle = curvature2tireangle(desired_curvature, self.wheel_base)

        # Run low-level controller and step vehicle dynamics TODO: non-perfect low-level controller
        tire_angle_velocity, accel = self._compute_perfect_control( \
            [desired_tire_angle, desired_speed], [self.tire_angle, self.speed], dt)
        self.ego_dynamics.step(tire_angle_velocity, accel, dt)

        # Update based on vehicle dynamics feedback
        self._tire_angle = self.ego_dynamics.steering
        self._speed = self.ego_dynamics.speed
        self._curvature = tireangle2curvature(self.tire_angle, self.wheel_base)
        self._steering = curvature2steering(self.curvature, self.wheel_base, self.steering_ratio)

        # Update human (reference) dynamics for assoication with the trace / dataset
        human = self.human_dynamics.copy()
        closest_dist = np.inf
        index = self.frame_index
        ts = self.trace.get_master_timestamp(self.segment_index, index)
        while True:
            next_index = index + 1
            exceed_end, next_ts = self.trace.get_master_timestamp( \
                self.segment_index, next_index, check_end=True)
            if exceed_end: # TODO: trigger trace done terminatal condition
                break

            current_state = [curvature2tireangle(self.curvature, self.wheel_base), self.speed]
            next_state = [curvature2tireangle(self.trace.f_curvature(next_ts), self.wheel_base),
                          self.trace.f_speed(next_ts)]
            tire_vel, acc = self._compute_perfect_control(next_state, current_state, next_ts - ts)
            human.step(tire_vel, acc, next_ts - ts)

            dist = np.linalg.norm(human.numpy()[:2] - self.ego_dynamics.numpy()[:2])
            print(dist, human.numpy()[:2], self.ego_dynamics.numpy()[:2]) # DEBUG
            if dist < closest_dist:
                closest_dist = dist
                index = next_index
                ts = next_ts
                self._human_dynamics = human.copy()
            else:
                break
        
        self._frame_index = index
        self._frame_number = self.trace.get_master_frame_number(self.segment_index, index)
        
        # Update timestamp based on where the car position is with respect to course distance of 
        # trace (may not be exactly the same as any of timestamps in the dataset)
        latlongyaw = transform.compute_relative_latlongyaw( \
            self.ego_dynamics.numpy()[:3], self.human_dynamics.numpy()[:3])
        latlongyaw_next = transform.compute_relative_latlongyaw( \
            self.ego_dynamics.numpy()[:3], human.numpy()[:3])
        # TODO: cannot handle case where second closest timestamp is in latlongyaw_previous
        import pdb; pdb.set_trace()
        
        if True:
            def debug(ego_state, human_state):
                ego_x_state, ego_y_state, ego_theta_state = ego_state
                human_x_state, human_y_state, human_theta_state = human_state

                c = np.cos(human_theta_state)
                s = np.sin(human_theta_state)
                R_2 = np.array([[c, -s], [s, c]])
                xy_global_centered = np.array([[ego_x_state - human_x_state],
                                            [human_y_state - ego_y_state]])
                [[translation_x], [translation_y]] = np.matmul(R_2, xy_global_centered)
                translation_y *= -1
                theta = ego_theta_state - human_theta_state

                return translation_x, translation_y, theta
            latlongyaw_old = debug(self.ego_dynamics.numpy()[:3], self.human_dynamics.numpy()[:3])
            print(np.abs(latlongyaw - latlongyaw_old))

    def step_sensors(self) -> None:
        self._observations = dict()
        for sensor in self.sensors:
            self._observations[sensor.name] = sensor.capture(self.timestamp)

    def _compute_perfect_control(self, desired_state: List[float], 
                                 current_state: List[float], dt: float):
        desired_state = np.array(desired_state)
        current_state = np.array(current_state)
        return (desired_state - current_state) / dt

    @property
    def trace(self) -> Trace:
        return self._trace

    @property
    def sensors(self) -> List[BaseSensor]:
        return self._sensors

    @property
    def relative_state(self) -> State:
        return self._relative_state

    @property
    def ego_dynamics(self) -> StateDynamics:
        return self._ego_dynamics

    @property
    def human_dynamics(self) -> StateDynamics:
        return self._human_dynamics

    @property
    def length(self) -> float:
        return self._length

    @property
    def width(self) -> float:
        return self._width

    @property
    def wheel_base(self) -> float:
        return self._wheel_base

    @property
    def steering_ratio(self) -> float:
        return self._steering_ratio

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def curvature(self) -> float:
        return self._curvature

    @property
    def steering(self) -> float:
        return self._steering

    @property
    def tire_angle(self) -> float:
        return self._tire_angle

    @property
    def human_speed(self) -> float:
        return self._human_speed

    @property
    def human_curvature(self) -> float:
        return self._human_curvature

    @property
    def human_steering(self) -> float:
        return self._human_steering

    @property
    def human_tire_angle(self) -> float:
        return self._human_tire_angle

    @property
    def timestamp(self) -> float:
        return self._timestamp

    @property
    def frame_number(self) -> int:
        return self._frame_number

    @property
    def trace_index(self) -> int:
        return self._trace_index

    @property
    def segment_index(self) -> int:
        return self._segment_index

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @property
    def observations(self) -> Dict[str, Any]:
        return self._observations

    def __repr__(self):
        raise NotImplementedError
