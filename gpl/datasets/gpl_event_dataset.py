from typing import List, Dict, Any, Optional
import random
import numpy as np
import torch

import vista
from vista.entities.agents.Dynamics import tireangle2curvature
from .buffered_dataset import BufferedDataset
from .utils import transform_events
from .privileged_controller import get_controller


__all__ = ['VistaDataset', 'worker_init_fn']


class VistaDataset(BufferedDataset):
    def __init__(self, 
                 trace_paths: List[str],
                 trace_config: Dict[str, Any],
                 car_config: Dict[str, Any],
                 reset_config: Dict[str, Any],
                 privileged_control_config: Dict[str, Any],
                 event_camera_config: Dict[str, Any],
                 train: Optional[bool] = False,
                 buffer_size: Optional[int] = 1,
                 snippet_size: Optional[int] = 100,
                 shuffle: Optional[bool] = False,
                 **kwargs):
        super(VistaDataset, self).__init__(trace_paths, trace_config, car_config,
            train, buffer_size, snippet_size, shuffle)

        assert self.car_config['lookahead_road'] == True, \
            'Require lookahead_raod = True for privileged control'

        self._reset_config = reset_config
        self._privileged_control_config = privileged_control_config
        self._event_camera_config = event_camera_config

    def _simulate(self):
        # Initialization for singl-processing dataloader
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self._world = vista.World(self.trace_paths, self.trace_config)
            self._agent = self._world.spawn_agent(self.car_config)
            self._event_camera = self._agent.spawn_event_camera(self.event_camera_config)
            self._world.reset({self._agent.id: self.initial_dynamics_fn})

            self._privileged_controller = get_controller(self.privileged_control_config)

        # Data generator from simulation
        self._snippet_i = 0
        while True:
            # reset simulator
            if self._agent.done or self._snippet_i >= self.snippet_size:
                if worker_info is not None:
                    self._world.set_seed(worker_info.id)
                self._world.reset({self._agent.id: self.initial_dynamics_fn})
                self._snippet_i = 0

            # cache simulator state and take random action
            speed = self._agent.human_speed
            curvature_bound = [
                tireangle2curvature(_v, self._agent.wheel_base)
                for _v in self._agent.ego_dynamics.steering_bound]
            curvature = self._rng.uniform(*curvature_bound)

            cached_state = self._cache_state()

            # step simulator to get events
            action = np.array([curvature, speed])
            self._agent.step_dynamics(action, update_road=False)
            self._agent.step_sensors()
            sensor_name = self._event_camera.name
            events = self._agent.observations[sensor_name]

            # revert dynamics and step with privileged control
            self._revert_dynamics(action, cached_state)

            curvature, speed = self._privileged_controller(self._agent)
            action = np.array([curvature, speed])
            self._agent.step_dynamics(action)

            # update RGB previous frame based on privileged control; this is required to generate
            # correct events for the next branching (the rgb frame at t-1 for optical flow 
            # between t-1 and t)
            self._event_camera.capture(self._agent.timestamp, update_rgb_frame_only=True)

            # preprocess and produce data-label pairs
            data = transform_events(events, self._event_camera, self.train)
            label = np.array([curvature]).astype(np.float32)

            self._snippet_i += 1

            # NOTE: use event data from previous step that executed non-privileged (e.g., random)
            # control to break correlation of events and ego-motion that result from the nature
            # of derivative sensors
            yield {'event_camera': data, 'target': label}

    def initial_dynamics_fn(self, x, y, yaw, steering, speed):
        return [
            x + self._rng.uniform(*self.reset_config['x_perturbation']),
            y,
            yaw + self._rng.uniform(*self.reset_config['yaw_perturbation']),
            steering,
            speed,
        ]

    def _cache_state(self):
        cached_state_keys = ['ego_dynamics', 'human_dynamics', 'relative_state',
            'speed', 'curvature', 'steering', 'tire_angle', 'human_speed',
            'human_curvature', 'human_steering', 'human_tire_angle',
            'frame_index', 'frame_number']
        cached_state = dict()
        for key in cached_state_keys:
            val = getattr(self._agent, key)
            if hasattr(val, 'numpy'):
                val = val.numpy()
            cached_state[key] = val
        return cached_state

    def _revert_dynamics(self, action, cached_state):
        self._agent._done = False
        self._agent.step_dynamics(action, dt=-1/30.)
        for key, val in cached_state.items():
            attr = getattr(self._agent, key)
            if hasattr(attr, 'numpy'):
                attr.update(*val)
            else:
                setattr(self._agent, '_' + key, val)

    @property
    def privileged_control_config(self) -> Dict[str, Any]:
        return self._privileged_control_config

    @property
    def reset_config(self) -> Dict[str, Any]:
        return self._reset_config

    @property
    def event_camera_config(self) -> Dict[str, Any]:
        return self._event_camera_config


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset._world = vista.World(dataset.trace_paths, dataset.trace_config)
    dataset._agent = dataset._world.spawn_agent(dataset.car_config)
    dataset._event_camera = dataset._agent.spawn_event_camera(dataset.event_camera_config)
    dataset._world.set_seed(worker_id)
    dataset._rng = random.Random(worker_id)
    dataset._world.reset({dataset._agent.id: dataset.initial_dynamics_fn})

    dataset._privileged_controller = get_controller(dataset.privileged_control_config)
