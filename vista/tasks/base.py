from typing import Optional, List, Dict, Any

from .. import World
from ..utils import logging, misc


class Base:
    DEFAULT_CONFIG = dict()

    def __init__(self,
                 trace_paths: List[str],
                 trace_config: Dict,
                 car_config: Dict,
                 sensors_configs: Optional[List[Dict]] = [],
                 task_config: Optional[Dict] = dict(),
                 logging_level: Optional[str] = 'WARNING'):
        logging.setLevel(getattr(logging, logging_level))

        self._config = misc.merge_dict(task_config, self.DEFAULT_CONFIG)

        self._world: World = World(trace_paths, trace_config)
        agent = self._world.spawn_agent(car_config)
        for sensor_config in sensors_configs:
            sensor_type = sensor_config.pop('type')
            if sensor_type == 'camera':
                agent.spawn_camera(sensor_config)
            else:
                raise NotImplementedError(
                    'Unrecognized sensor type {}'.format(sensor_type))

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError
