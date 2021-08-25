import os
from typing import Dict
import numpy as np
import h5py

from .BaseSensor import BaseSensor
from .lidar_utils import LidarSynthesis, Pointcloud
from ..Entity import Entity
from ...utils import logging


class Lidar(BaseSensor):
    def __init__(self, attach_to: Entity, config: Dict) -> None:
        super(Lidar, self).__init__(attach_to, config)

        logging.debug('Not actually streaming lidar data when reading')
        self._streams: Dict[str, h5py.File] = dict()

        # Initialize lidar novel view synthesis object
        self._view_synthesis = LidarSynthesis()

    def reset(self) -> None:
        logging.info(f'Lidar ({self.id}) reset')

        # Initiate lidar data stream based on current reference pointer to the
        # dataset. All data streams are handled by the main lidar and shared
        # across all Lidar objects in an agent.
        multi_sensor = self.parent.trace.multi_sensor
        if self.name == multi_sensor.main_lidar:
            for lidar_name in multi_sensor.lidar_names:
                fpath = os.path.join(self.parent.trace.trace_path,
                                     lidar_name + '.h5')
                stream = h5py.File(fpath, 'r')
                self._streams[lidar_name] = stream
        else:
            main_name = multi_sensor.main_lidar
            main_sensor = [
                _s for _s in self.parent.sensors if _s.name == main_name
            ]
            assert len(main_sensor) == 1, \
                    f'Cannot find main sensor {main_name}'

            main_sensor = main_sensor[0]
            assert isinstance(main_sensor, Lidar), \
                    'Main sensor is not Lidar object'
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
            xyz = stream['xyz'][frame_num]
            intensity = stream['intensity'][frame_num]
            pcd = Pointcloud(xyz, intensity)
            pcd = pcd[pcd.dist > 2.5]
            # TODO: when is it possible for there to be multiple (multi_sensor.lidar_names)?

        # TODO: Interpolate frame at the exact timestamp
        pass

        # TODO: Synthesis by rendering
        # self.parent.reslative_state.update(0, 0, yaw=np.sin(timestamp))
        lat, long, yaw = self.parent.relative_state.numpy()
        logging.debug(f"state: {lat} {long} {yaw} \t timestamp {timestamp}")
        trans = np.array([long, lat, 0])
        rot = np.array([0., 0, yaw])  # TODO: should yaw be Y or Z?
        rendered_lidar = self.view_synthesis.synthesize(
            trans,
            rot,
            pcd=pcd,
            return_as_pcd=True,
        )

        logging.debug("Visualizing the rendered lidar scan")

        return rendered_lidar

    @property
    def config(self) -> Dict:
        return self._config

    @property
    def streams(self) -> Dict[str, h5py.File]:
        return self._streams

    @property
    def view_synthesis(self) -> LidarSynthesis:
        return self._view_synthesis
