from typing import List, Dict, Any
import numpy as np

from .Dynamics import State, StateDynamics
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
        self._human_speed: float = 0.
        self._human_curvature: float = 0.
        self._human_steering: float = 0.
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
        # Reset states and dynamics
        self._relative_state.reset()
        self._ego_dynamics.reset()
        self._human_dynamics.reset()

        # Update pointers to dataset
        self._trace = self.parent.traces[self.trace_index]
        self._timestamp = self.trace.get_master_timestamp(segment_index, frame_index)
        self._frame_number = self.trace.get_master_frame_number(segment_index, frame_index)
        self._trace_index = trace_index
        self._segment_index = segment_index
        self._frame_index = frame_index

        # Reset sensors
        for sensor in self.sensors:
            sensor.reset()

        # Update observation
        self._observations = dict()
        for sensor in self.sensors:
            self._observations[sensor.name] = sensor.capture(self.timestamp)

    def step_dynamics(self, action: np.ndarray, dt: float):
        raise NotImplementedError

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
    def human_speed(self) -> float:
        return self._human_speed

    @property
    def human_steering(self) -> float:
        return self._human_steering

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
