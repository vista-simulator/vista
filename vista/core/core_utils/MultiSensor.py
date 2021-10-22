import os
import numpy as np
from typing import Dict, List, Optional
from collections import OrderedDict
import h5py

from . import TopicNames
from ...utils import logging


class MultiSensor:
    """ This class handles synchronization across multiple sensors in a trace. It basically
    reads timestamps files associated with every sensors existing in the trace and construct
    several helper functions for conversion between an unified timestamp used in the simulator
    and timestamp (or frame number, etc) of different sensors.

    Args:
        trace_dir (str): Directory to a trace.
        master_sensor (str): Name of the master sensor.

    Raises:
        AssertionError: (1) The name of the master sensor not included in a predefined
        set of topic names specified in  (2) No timestamp data for the master sensor.

    """
    def __init__(
            self,
            trace_dir: str,
            master_sensor: Optional[str] = TopicNames.master_topic) -> None:
        self._trace_dir: str = trace_dir
        self._master_sensor: str = master_sensor

        # Get frame-to-timestamp mapping for every sensors
        self._sensor_frame_to_time: Dict = dict()
        sensor_topic_names = [TopicNames.lidar_3d] + list(TopicNames.cameras)
        assert master_sensor in sensor_topic_names, \
            f'Master sensor {master_sensor} not in topic names' + \
            f'{sensor_topic_names}. Please check camera config or TopicNames.py'
        for fname in os.listdir(self._trace_dir):
            sensor_name, ext = os.path.splitext(fname)
            if sensor_name in sensor_topic_names:
                fpath = os.path.join(self._trace_dir, fname)
                if ext == '.csv':
                    data = np.genfromtxt(fpath,
                                         delimiter=',',
                                         skip_header=1,
                                         dtype=np.float64)
                    frames, times = (data[:, 0], data[:, 1])
                elif ext == ".h5":
                    f = h5py.File(fpath, "r")
                    times = f["timestamp"][:, 0]
                    frames = np.arange(times.shape[0])
                else:
                    continue  # Not implemented yet...

                frame_to_time = OrderedDict()
                for i in range(len(frames)):
                    frame_to_time[int(frames[i])] = times[i]
                self._sensor_frame_to_time[sensor_name] = frame_to_time

        self._sensor_names: List[str] = list(self._sensor_frame_to_time.keys())
        assert master_sensor in self.sensor_names, \
            f'No timestamp data for the master sensor {master_sensor}'

    def get_time_from_frame_num(self, sensor: str, frame_num: int) -> float:
        """ Compute the timestamp associated to a frame in a video return
        None if we dont have information about that frame.

        Args:
            sensor (str): Sensor name.
            frame_num (int): Frame number.

        Returns:
            float: Timestamp associated with the given sensor and frame number.
        """
        return self._sensor_frame_to_time[sensor].get(frame_num, None)

    def get_frames_from_times(
            self,
            timestamps: List[float],
            fetch_smaller: Optional[bool] = False) -> Dict[str, List[int]]:
        """ Takes in a list of timestamps and returns corresponding frame
        numbers for each sensor. Note that since sensors are not necessarily
        sync'ed, the returned frame numbers are the one with the closest
        (smaller) timestamps.

        Args:
            timestamps (list): A list of timestamps.
            fetch_smaller (bool): Whether to fetch the closes and smaller timestamps.

        Returns:
            dict: Corresponding frame numbers for all sensors.
        """
        frames = dict()
        timestamps = np.array(timestamps)
        for sensor in self.sensor_names:
            frame_to_time = self._sensor_frame_to_time[sensor]

            frames[sensor] = []
            pointer = 0
            for ts in timestamps:
                while pointer < len(frame_to_time) - 1:
                    if ts >= frame_to_time[pointer] and ts < frame_to_time[
                            pointer + 1]:
                        if fetch_smaller:
                            frames[sensor].append(pointer)
                        else:
                            if np.abs(frame_to_time[pointer] - ts) >= \
                                np.abs(ts - frame_to_time[pointer + 1]):
                                frames[sensor].append(pointer + 1)
                            else:
                                frames[sensor].append(pointer)
                        break
                    else:
                        pointer += 1

        return frames

    def get_master_timestamps(self) -> List[float]:
        """ Get all timestamps of the main sensor.

        Returns:
            list: A list of timestamp.
        """
        # using values() works since it's an ordered dict
        timestamps = list(
            self._sensor_frame_to_time[self._master_sensor].values())
        return timestamps

    def set_main_sensor(self, sensor_type: str, sensor_name: str) -> None:
        """ Set main sensor based on sensor's type and name.

        Args:
            sensor_type (str): Type of the sensor to be set (camera, lidar, or event camera).
            sensor_name (str): Name of the sensor to be set.
        """
        assert sensor_type in ['camera', 'lidar', 'event_camera']
        setattr(self, '_main_{}'.format(sensor_type), sensor_name)

    @property
    def sensor_names(self) -> List[str]:
        """ A list of all sensors' names. """
        return self._sensor_names

    @property
    def camera_names(self) -> List[str]:
        """ A list of RGB cameras' names. """
        logging.debug('Hacky way to include RGB camera with name front_center')
        return [
            _x for _x in self._sensor_names
            if 'camera' in _x or _x == 'front_center'
        ]

    @property
    def main_camera(self) -> str:
        """ The main RGB camera object. """
        return self._main_camera if hasattr(self, '_main_camera') else None

    @property
    def lidar_names(self) -> List[str]:
        """ A list of LiDARs' names. """
        return [_x for _x in self._sensor_names if 'lidar' in _x]

    @property
    def main_lidar(self) -> str:
        """ The main LiDAR object """
        return self._main_lidar if hasattr(self, '_main_lidar') else None

    @property
    def main_event_camera(self) -> str:
        """ The main event camera object. """
        return self._main_event_camera if hasattr(
            self, '_main_event_camera') else None

    @property
    def master_sensor(self) -> str:
        """ The name of the master sensor. """
        return self._master_sensor
