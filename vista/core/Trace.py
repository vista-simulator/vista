import os
from typing import Optional, List, Any, Dict, Callable, Union, Tuple
import numpy as np
from scipy.interpolate import interp1d

from .core_utils import MultiSensor, LabelSearch, TopicNames
from ..utils import logging, misc


class Trace:
    """ This is a helper class for handling a trace in the dataset. It allows straightforward
    access to any pointer in the trace. A trace is separated into several segments based on
    timestamps (split to segments if the difference of two consecutive timestamps is larger than
    a value). Each segment contains frames that can be accessed through specifying frame index.
    Note that frame index used in VISTA is not necessarily the same as frame number in the dataset.
    Overall, we use ``segment_index``, ``frame_index``, ``frame_number``, ``timestamp`` in this
    class and there are several functions for conversion among them.
    The :class:`Trace` also performs filtering including,

        * Filter frames based on annotation label (`video_label.csv`) if there is any.
        * Filter by end-of-trace and time difference across consecutive frames.

    This class also handles sampling of pointer to the trace, i.e., sampling segment and frame
    index.

    Args:
        trace_path (str): Path to a trace.
        trace_config (dict): Configuration of the trace object.

    """
    DEFAULT_LABELS = [
        'day|night', 'dry|rain|snow',
        'local|residential|highway|unpaved|indoor', 'stable', '.*', '.*'
    ]
    RESET_CONFIG = {
        'default': {
            'n_bins': 100,
            'smoothing_factor': 0.001,
            'hist_power': 2.,
        },
        'segment_start': {
            'first_n_percent': 0.0,
        },
    }
    DEFAULT_CONFIG = {
        'reset_mode': 'default',
        'master_sensor': TopicNames.master_topic,
        'labels': DEFAULT_LABELS,
        'max_timestamp_diff_across_frames': 0.2,
        'road_width': 4,
    }

    def __init__(
        self, trace_path: str, trace_config: Optional[Dict] = dict()) -> None:
        self._trace_path: str = trace_path
        self._config: Dict = misc.merge_dict(trace_config, self.DEFAULT_CONFIG)

        # Get function representation of state information
        self._f_speed, self._f_curvature = self._get_states_func()

        # Divide trace to good segments based on video labels and timestamps
        self._multi_sensor: MultiSensor = MultiSensor(
            self._trace_path, self._config['master_sensor'])
        self._labels: LabelSearch = LabelSearch(*self._config['labels'])

        good_frames, good_timestamps = self._divide_to_good_segments()
        self._good_frames: Dict[str, List[int]] = good_frames
        self._good_timestamps: Dict[str, List[float]] = good_timestamps

        self._num_of_frames: int = np.sum([
            len(_v)
            for _v in self._good_frames[self._multi_sensor.master_sensor]
        ])

    def find_segment_reset(self) -> int:
        """ Sample a segment index based on number of frames in each segment. Segments with more
        frames will be sampled with a higher probability.

        Returns:
            int: Index of a segment.

        """
        segment_reset_probs = np.zeros(
            len(self._good_frames[self._multi_sensor.master_sensor]))
        for i in range(segment_reset_probs.shape[0]):
            segment = self._good_frames[self._multi_sensor.master_sensor][i]
            segment_reset_probs[i] = len(segment)
        segment_reset_probs /= np.sum(segment_reset_probs)
        new_segment_index = self._rng.choice(segment_reset_probs.shape[0],
                                             p=segment_reset_probs)

        return new_segment_index

    def find_frame_reset(self, segment_index: int) -> int:
        """ Sample a frame index in a segment.

        Args:
            segment_index (int): Index of the segment to be sampled from.

        Returns:
            int: Frame index.

        Raises:
            NotImplementedError: Unrecognized trace reset mode.

        """
        # Compute sample probability
        timestamps = self.good_timestamps[
            self._multi_sensor.master_sensor][segment_index]

        if self._config['reset_mode'] == 'default':
            # bias toward large road curvature
            n_bins = Trace.RESET_CONFIG['default']['n_bins']
            smoothing_factor = Trace.RESET_CONFIG['default'][
                'smoothing_factor']
            hist_power = Trace.RESET_CONFIG['default']['hist_power']

            curvatures = np.abs(self.f_curvature(timestamps))
            curvatures = np.clip(curvatures, 0, 1 / 3.)
            hist, bin_edges = np.histogram(curvatures, n_bins, density=False)
            bins = np.digitize(curvatures, bin_edges, right=True)
            hist_density = hist / float(np.sum(hist))
            probs = 1.0 / (
                (hist_density[bins - 1] + smoothing_factor)**hist_power)
            probs /= np.sum(probs)
        elif self._config['reset_mode'] == 'uniform':  # uniform sampling
            n_timestamps = len(timestamps)
            probs = np.ones((n_timestamps, )) / n_timestamps
        elif self._config[
                'reset_mode'] == 'segment_start':  # bias toward the start of a segment
            first_n_percent = Trace.RESET_CONFIG['segment_start'][
                'first_n_percent']

            n_timestamps = len(timestamps)
            probs = np.zeros((n_timestamps, ))
            to_idx = max(int(first_n_percent * n_timestamps), 1)
            probs[:to_idx] = 1.
            probs /= np.sum(probs)
        else:
            raise NotImplementedError(
                f'Unrecognized trace reset mode {self._reset_mode}')

        # Sample frame index
        frame_index = self._rng.choice(probs.shape[0], p=probs)

        return frame_index

    def get_master_timestamp(
            self,
            segment_index: int,
            frame_index: int,
            check_end: Optional[bool] = False
    ) -> Union[float, Tuple[bool, float]]:
        """ Get timestamp from the master sensor.

        Args:
            segment_index (int): Segment index.
            frame_index (int): Frame index.
            check_end (bool): Whether to check if the given segment and frame index
                              exceed the end of the trace; default to False.

        Returns:
            Return a tuple (``bool_a``, ``float_a``), where ``bool_a`` is whether the given segment
            and frame index exceed the end of the trace and ``float_a`` is timestamp; otherwise
            timestamp (float).

        """
        master_name = self.multi_sensor.master_sensor
        if check_end:
            exceed_end = frame_index >= len(
                self.good_timestamps[master_name][segment_index])
            frame_index = -1 if exceed_end else frame_index
            return exceed_end, self.good_timestamps[master_name][
                segment_index][frame_index]
        else:
            return self.good_timestamps[master_name][segment_index][
                frame_index]

    def get_master_frame_number(
            self,
            segment_index: int,
            frame_index: int,
            check_end: Optional[bool] = False) -> Union[int, Tuple[bool, int]]:
        """ Get frame number from the master sensor.

        Args:
            segment_index (int): Segment index.
            frame_index (int): Frame index.
            check_end (bool): Whether to check if the given segment and frame index
                              exceed the end of the trace; default to False.

        Returns:
            Return a tuple (``bool_a``, ``int_a``), where ``bool_a`` is whether the given segment
            and frame index exceed the end of the trace and ``int_a`` is frame number; otherwise
            frame number (int).

        """
        master_name = self.multi_sensor.master_sensor
        if check_end:
            exceed_end = frame_index >= len(
                self.good_timestamps[master_name][segment_index])
            frame_index = -1 if exceed_end else frame_index
            return exceed_end, self.good_frames[master_name][segment_index][
                frame_index]
        else:
            return self.good_frames[master_name][segment_index][frame_index]

    def _divide_to_good_segments(
        self,
        min_speed: float = 2.5,
    ) -> Dict[str, List[int]]:
        """ Divide a trace into good segments based on video labels and time difference between
        consecutive frames. Note that only master sensor is used for the time difference check
        since every sensors may have triggering frequencies.

        Returns:
            Return a tuple (``dict_a``, ``dict_b``), where
            ``dict_a``: good frames for all sensors. Key is sensor name and value
                        is a list with each element as frame indices of a good
                        segment, i.e., a good frame number =
                        dict[sensor_name][which_good_segment][i].
            ``dict_b``: timestamps of good frames.

        """
        # Filter by video labels
        _, good_labeled_timestamps = self._labels.find_good_labeled_frames(
            self._trace_path)
        if good_labeled_timestamps is None:
            logging.warning('No video_label.csv')
            good_labeled_timestamps = np.array(
                self._multi_sensor.get_master_timestamps())

        good_speed_inds = self.f_speed(good_labeled_timestamps) > min_speed
        good_labeled_timestamps = good_labeled_timestamps[good_speed_inds]

        # Filter by end-of-trace and time difference across consecutive frames
        good_frames = {_k: [] for _k in self._multi_sensor.sensor_names}
        good_timestamps = {_k: [] for _k in self._multi_sensor.sensor_names}
        segment_start = 0
        for i in range(good_labeled_timestamps.shape[0]):
            trace_end = i == good_labeled_timestamps.shape[0] - 1
            if not trace_end:
                time_diff = good_labeled_timestamps[
                    i + 1] - good_labeled_timestamps[i]
                time_diff_too_large = time_diff >= self._config[
                    'max_timestamp_diff_across_frames']
            else:
                time_diff_too_large = False

            if trace_end or time_diff_too_large:
                good_segment_frames = self._multi_sensor.get_frames_from_times(
                    good_labeled_timestamps[segment_start:i])

                for k, v in good_segment_frames.items():
                    good_frames[k].append(v)

                    segment_timestamps = []
                    for frame_num in v:
                        segment_timestamps.append(
                            self._multi_sensor.get_time_from_frame_num(
                                k, frame_num))
                    good_timestamps[k].append(segment_timestamps)

                segment_start += i

        return good_frames, good_timestamps

    def _get_states_func(self):
        """ Read speed and IMU data from the dataset and perform 1D interpolation to convert
        discrete data points to continuous functions. Curvature is computed as yaw rate (from
        IMU) divided by speed.

        Returns:
            Return a tuple (``func_a``, ``func_b``), where ``func_a`` is 1D interpolation
            function for speed and ``func_b`` is 1D interpolation for curvature.

        """
        # Read from dataset
        speed = np.genfromtxt(os.path.join(self._trace_path,
                                           TopicNames.speed + '.csv'),
                              delimiter=',')
        imu = np.genfromtxt(os.path.join(self._trace_path,
                                         TopicNames.imu + '.csv'),
                            delimiter=',')

        # Obtain function representation of speed
        f_speed = interp1d(speed[:, 0], speed[:, 1], fill_value='extrapolate')

        # Obtain function representation of curvature
        timestamps = imu[:, 0]
        yaw_rate = imu[:, 6]
        curvature = yaw_rate / np.maximum(f_speed(timestamps), 1e-10)
        good_curvature_inds = np.abs(curvature) < 1 / 3.
        f_curvature = interp1d(timestamps[good_curvature_inds],
                               curvature[good_curvature_inds],
                               fill_value='extrapolate')

        return f_speed, f_curvature

    def set_seed(self, seed) -> None:
        """ Set random seed.

        Args:
            seed (int): Random seed.

        """
        self._seed = seed
        self._rng = np.random.default_rng(self.seed)

    @property
    def seed(self) -> int:
        """ Random seed for sampling pointer in :meth:`find_segment_reset`
            and :meth:`find_frame_reset`. """
        return self._seed

    @property
    def trace_path(self) -> str:
        """ Path to the directory that contains all data of this trace. """
        return self._trace_path

    @property
    def multi_sensor(self) -> MultiSensor:
        """ MultiSensor object. Please check the definition of this class. """
        return self._multi_sensor

    @property
    def good_frames(self) -> Dict[str, List[int]]:
        """ A dictionary of good frame numbers, where keys are sensor names and
            values are lists of frame numbers (int). """
        return self._good_frames

    @property
    def good_timestamps(self) -> Dict[str, List[float]]:
        """ A dictionary of good timestamps, where keys are sensor names and
            values are a list of timestamps (float). """
        return self._good_timestamps

    @property
    def num_of_frames(self) -> int:
        """ Number of good frames. """
        return self._num_of_frames

    @property
    def f_curvature(self) -> Callable:
        """ A 1D interpolation function for curvature of the ego-car. """
        return self._f_curvature

    @property
    def f_speed(self) -> Callable:
        """ A 1D interpolation function for speed of the ego-car. """
        return self._f_speed

    @property
    def reset_mode(self) -> str:
        """ Trace reset mode. """
        return self._reset_mode

    @reset_mode.setter
    def reset_mode(self, reset_mode):
        assert isinstance(reset_mode, str)
        self._reset_mode = reset_mode

    @property
    def road_width(self) -> float:
        """ Road width. """
        return self._config['road_width']

    @property
    def param_file(self) -> float:
        """ Path to the sensor parameters file for this trace. """
        return os.path.join(self._trace_path, "params.xml")

    def __repr__(self) -> str:
        trace_name = os.path.basename(self.trace_path)
        return '<{}: {}>'.format(self.__class__.__name__, trace_name)
