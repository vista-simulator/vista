import os
from typing import Optional, List, Any, Dict, Callable
import numpy as np
from scipy.interpolate import interp1d

from .core_utils import MultiSensor, LabelSearch, TopicNames


class Trace:
    DEFAULT_LABELS = ['day|night', 'dry|rain|snow', 'local|residential|highway|unpaved|indoor',
                      'stable', '.*', '.*']
    RESET_CONFIG = {
        'default': {
            'n_bins': 100,
            'smoothing_factor': 0.001,
        },
        'segment_start': {
            'first_n_percent': 0.4,
        },
    }

    def __init__(self, trace_path: str, reset_mode: Optional[str] = 'default',
                 master_sensor: Optional[str] = TopicNames.master_topic,
                 labels: Optional[List[Any]] = DEFAULT_LABELS,
                 max_timestamp_diff_across_frames: Optional[float] = 0.2) -> None:
        self._trace_path: str = trace_path
        self._reset_mode: str = reset_mode
        self._max_timestamp_diff_across_frames = max_timestamp_diff_across_frames

        # Divide trace to good segments based on video labels and timestamps
        self._multi_sensor: MultiSensor = MultiSensor(self._trace_path, master_sensor)
        self._labels: LabelSearch = LabelSearch(*labels)

        good_frames, good_timestamps = self._divide_to_good_segments()
        self._good_frames: Dict[str, List[int]] = good_frames
        self._good_timestamps: Dict[str, List[float]] = good_timestamps

        self._num_of_frames: int = np.sum([len(_v) for _v in \
            self._good_frames[self._multi_sensor.master_sensor]])

        # Get function representation of state information
        self._f_speed, self._f_curvature = self._get_states_func()

    def find_segment_reset(self) -> int:
        """ Sample a segment index based on number of frames in each segment. Segments with more
            frames will be sampled with a higher probability.

        Args:
            None

        Returns:
            int: index to a segment
        """
        segment_reset_probs = np.zeros(len(self._good_frames[self._multi_sensor.master_sensor]))
        for i in range(segment_reset_probs.shape[0]):
            segment = self._good_frames[self._multi_sensor.master_sensor][i]
            segment_reset_probs[i] = len(segment)
        segment_reset_probs /= np.sum(segment_reset_probs)
        new_segment_index = np.random.choice(segment_reset_probs.shape[0], p=segment_reset_probs)

        return new_segment_index

    def find_frame_reset(self, segment_index: int) -> int:
        """ Sample a frame index in a segment.

        Args:
            segment_index (int): index of the segment to be sampled from

        Returns:
            int: frame index

        Raises:
            NotImplementedError: for invalid reset mode
        """
        # Compute sample probability
        timestamps = self.good_timestamps[self._multi_sensor.master_sensor][segment_index]
        if self._reset_mode == 'default': # bias toward large road curvature
            n_bins = Trace.RESET_CONFIG['default']['n_bins']
            smoothing_factor = Trace.RESET_CONFIG['default']['smoothing_factor']

            curvatures = np.abs(self.f_curvature(timestamps))
            curvatures = np.clip(curvatures, 0, 1 / 3.)
            hist, bin_edges = np.histogram(curvatures, n_bins, density=False)
            bins = np.digitize(curvatures, bin_edges, right=True)
            hist_density = hist / float(np.sum(hist))
            probs = 1.0 / (hist_density[bins - 1] + smoothing_factor)
            probs /= np.sum(probs)
        elif self._reset_mode == 'uniform': # uniform sampling
            n_timestamps = len(timestamps)
            probs = np.ones((n_timestamps,)) / n_timestamps
        elif self._reset_mode == 'segment_start': # bias toward the start of a segment
            first_n_percent = Trace.RESET_CONFIG['segment_start']['first_n_percent']

            n_timestamps = len(timestamps)
            probs = np.zeros((n_timestamps,))
            probs[:int(first_n_percent * n_timestamps)] = 1.
            probs /= np.sum(probs)
        else:
            raise NotImplementedError('Unrecognized trace reset mode {}'.format(self._reset_mode))

        # Sample frame index
        frame_index = np.random.choice(probs.shape[0], p=probs)
        
        return frame_index

    def get_master_timestamp(self, segment_index: int, frame_index: int) -> float:
        master_name = self.multi_sensor.master_sensor
        return self.good_timestamps[master_name][segment_index][frame_index]

    def get_master_frame_number(self, segment_index: int, frame_index: int) -> float:
        master_name = self.multi_sensor.master_sensor
        return self.good_frames[master_name][segment_index][frame_index]

    def _divide_to_good_segments(self) -> Dict[str, List[int]]:
        """ Divide a trace into good segments based on video labels and time difference between
            consecutive frames. Note that only master sensor is used for the time difference check
            since every sensors may have triggering frequencies.

        Args:
            None
        
        Returns:
            dict: good frames for all sensors. Key is sensor name and value is a list with each 
                  element as frame indices of a good segment, 
                  i.e., a good frame number = dict[sensor_name][which_good_segment][i]
            dict: timestamps of good frames
        """
        # Filter by video labels (TODO: make optional)
        _, good_labeled_timestamps = self._labels.find_good_labeled_frames(self._trace_path)

        # Filter by end-of-trace and time difference across consecutive frames
        good_frames = {_k: [] for _k in self._multi_sensor.sensor_names}
        good_timestamps = {_k: [] for _k in self._multi_sensor.sensor_names}
        segment_start = 0
        for i in range(good_labeled_timestamps.shape[0]):
            trace_end = i == good_labeled_timestamps.shape[0] - 1
            if not trace_end:
                time_diff_too_large = good_labeled_timestamps[i+1] - good_labeled_timestamps[i] \
                    >= self._max_timestamp_diff_across_frames
            else:
                time_diff_too_large = False

            if trace_end or time_diff_too_large:
                good_segment_frames = self._multi_sensor.get_frames_from_times( \
                    good_labeled_timestamps[segment_start:i])

                for k, v in good_segment_frames.items():
                    good_frames[k].append(v)
                    
                    segment_timestamps = []
                    for frame_num in v:
                        segment_timestamps.append(
                            self._multi_sensor.get_time_from_frame_num(k, frame_num))
                    good_timestamps[k].append(segment_timestamps)

                segment_start += i

        return good_frames, good_timestamps

    def _get_states_func(self):
        # Read from dataset
        speed = np.genfromtxt(os.path.join(self._trace_path,
                                           TopicNames.speed + '.csv'),
                              delimiter=',')
        odometry = np.genfromtxt(os.path.join(self._trace_path,
                                              TopicNames.odometry + '.csv'),
                                 delimiter=',')

        # Obtain function representation of speed
        f_speed = interp1d(speed[:, 0], speed[:, 1], fill_value='extrapolate')

        # Obtain function representation of curvature
        timestamps = odometry[:, 0]
        yaw_rate = odometry[:, 4]
        curvature = yaw_rate / np.maximum(f_speed(timestamps), 1e-10)
        good_curvature_inds = np.abs(curvature) < 1 / 3.
        f_curvature = interp1d(timestamps[good_curvature_inds],
                               curvature[good_curvature_inds],
                               fill_value='extrapolate')

        return f_speed, f_curvature

    @property
    def trace_path(self) -> str:
        return self._trace_path

    @property
    def multi_sensor(self) -> MultiSensor:
        return self._multi_sensor

    @property
    def good_frames(self) -> Dict[str, List[int]]:
        return self._good_frames

    @property
    def good_timestamps(self) -> Dict[str, List[int]]:
        return self._good_timestamps
    
    @property
    def num_of_frames(self) -> int:
        return self._num_of_frames

    @property
    def f_curvature(self) -> Callable:
        return self._f_curvature

    @property
    def f_speed(self) -> Callable:
        return self._f_speed

    @property
    def reset_mode(self) -> str:
        return self._reset_mode

    @reset_mode.setter
    def reset_mode(self, reset_mode):
        assert isinstance(reset_mode, str)
        self._reset_mode = reset_mode
