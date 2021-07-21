import os
from typing import Dict
import numpy as np
import h5py

from .BaseSensor import BaseSensor
from .lidar_utils import LidarSynthesis
from ..Entity import Entity
from ...utils import logging


class Lidar(BaseSensor):
    def __init__(self, attach_to: Entity, config: Dict) -> None:
        super(Lidar, self).__init__(attach_to, config)

        logging.debug('Not actually streaming lidar data when reading')
        self._streams: Dict[str, h5py.File] = dict()
        # TODO: initialize lidar novel view synthesis object

    def reset(self) -> None:
        logging.info('Lidar ({}) reset'.format(self.id))

        # Initiate lidar data stream based on current reference pointer to the dataset. All data
        # streams are handled by the main lidar and shared across all Lidar objects in an agent.
        multi_sensor = self.parent.trace.multi_sensor
        if self.name == multi_sensor.main_lidar:
            for lidar_name in multi_sensor.lidar_names:
                fpath = os.path.join(self.parent.trace.trace_path, lidar_name + '.h5')
                stream = h5py.File(fpath, 'r')
                self._streams[lidar_name] = stream
        else:
            main_name = multi_sensor.main_lidar
            main_sensor = [_s for _s in self.parent.sensors if _s.name == main_name]
            assert len(main_sensor) == 1, 'Cannot find main sensor {}'.format(main_name)
            main_sensor = main_sensor[0]
            assert isinstance(main_sensor, Lidar), 'Main sensor is not Lidar object'
            self._streams = main_sensor.streams

        # TODO: reset lidar synthesis

    def capture(self, timestamp: float) -> np.ndarray:
        logging.info('Lidar ({}) capture'.format(self.id))

        # Get frame at the closest smaller timestamp from dataset.
        multi_sensor = self.parent.trace.multi_sensor
        all_frame_nums = multi_sensor.get_frames_from_times([timestamp])
        for lidar_name in multi_sensor.lidar_names:
            stream = self.streams[lidar_name]
            frame_num = all_frame_nums[lidar_name][0]
            pc = stream['xyz'][frame_num]

        # TODO: Interpolate frame at the exact timestamp

        # TODO: Synthesis by rendering
        lat, long, yaw = self.parent.relative_state.numpy()
        raise NotImplementedError

    @property
    def config(self) -> Dict:
        return self._config

    @property
    def streams(self) -> Dict[str, h5py.File]:
        return self._streams
