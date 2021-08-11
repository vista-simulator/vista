import os
import numpy as np
from typing import Dict, List, Optional
from collections import OrderedDict

from . import TopicNames
from ...utils import logging


class MultiSensor:
    def __init__(
            self,
            trace_dir: str,
            master_sensor: Optional[str] = TopicNames.master_topic) -> None:
        self._trace_dir: str = trace_dir
        self._master_sensor: str = master_sensor

        # Get frame-to-timestamp mapping for every sensors
        self._sensor_frame_to_time: Dict = dict()
        sensor_topic_names = [TopicNames.lidar_3d
                              ] + [_x for _x in TopicNames.cameras]
        assert master_sensor in sensor_topic_names, \
            f'Master sensor {master_sensor} not in topic names' + \
            f'{sensor_topic_names}. Please check camera config or TopicNames.py'
        for fname in os.listdir(self._trace_dir):
            sensor_name, ext = os.path.splitext(fname)
            if sensor_name in sensor_topic_names and ext == '.csv':
                fpath = os.path.join(self._trace_dir, fname)
                data = np.genfromtxt(fpath,
                                     delimiter=',',
                                     skip_header=1,
                                     dtype=np.float64)
                frame_to_time = OrderedDict()
                for i in range(data.shape[0]):
                    frame_to_time[int(data[i, 0])] = data[i, 1]
                self._sensor_frame_to_time[sensor_name] = frame_to_time

        self._sensor_names: List[str] = list(self._sensor_frame_to_time.keys())
        assert master_sensor in self.sensor_names, \
            'No timestamp data for the master sensor {}'.format(master_sensor)

    def get_time_from_frame_num(self, sensor: str, frame_num: int) -> float:
        """ Compute the timestamp associated to a frame in a video return
        None if we dont have information about that frame.

        Args:
            sensor (str): sensor name
            frame_num (int): frame number

        Returns:
            float: timestamp associated with the given sensor and frame number
        """
        return self._sensor_frame_to_time[sensor].get(frame_num, None)

    def get_frames_from_times(self,
                              timestamps: List[float],
                              fetch_smaller: Optional[bool] = False) -> Dict[str, List[int]]:
        """ Takes in a list of timestamps and returns corresponding frame
        numbers for each sensor. Note that since sensors are not necessarily
        sync'ed, the returned frame numbers are the one with the closest
        (smaller) timestamps.

        Args:
            timestamps (list): a list of timestamps
            fetch_smaller (bool): whether to fetch the closes AND smaller timestamps

        Returns:
            dict: corresponding frame numbers for all sensors
        """
        frames = dict()
        timestamps = np.array(timestamps)
        for sensor in self.sensor_names:
            frame_to_time = self._sensor_frame_to_time[sensor]

            frames[sensor] = []
            pointer = 0
            for ts in timestamps:
                while pointer < len(frame_to_time):
                    if ts >= frame_to_time[pointer] and ts < frame_to_time[
                            pointer + 1]:
                        if fetch_smaller:
                            frames[sensor].append(pointer)
                        else:
                            if np.abs(frame_to_time[pointer] - ts) >= \
                                np.abs(ts - frame_to_time[pointer + 1]):
                                frames[sensor].append(pointer)
                            else:
                                frames[sensor].append(pointer + 1)
                        break
                    else:
                        pointer += 1

        return frames

    def get_master_timestamps(self) -> List[float]:
        # using values() works since it's an ordered dict
        timestamps = list(
            self._sensor_frame_to_time[self._master_sensor].values())
        return timestamps

    def set_main_sensor(self, sensor_type: str, sensor_name: str) -> None:
        assert sensor_type in ['camera', 'lidar']
        setattr(self, '_main_{}'.format(sensor_type), sensor_name)

    @property
    def sensor_names(self) -> List[str]:
        return self._sensor_names

    @property
    def camera_names(self) -> List[str]:
        logging.debug('Hacky way to include RGB camera with name front_center')
        return [_x for _x in self._sensor_names if 'camera' in _x or 'front_center' == _x]

    @property
    def main_camera(self) -> str:
        return self._main_camera if hasattr(self, '_main_camera') else None

    @property
    def lidar_names(self) -> List[str]:
        return [_x for _x in self._sensor_names if 'lidar' in _x]

    @property
    def main_lidar(self) -> str:
        return self._main_lidar if hasattr(self, '_main_lidar') else None

    @property
    def master_sensor(self) -> str:
        return self._master_sensor
