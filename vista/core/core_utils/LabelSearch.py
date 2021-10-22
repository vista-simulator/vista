import os
import re
import csv
from typing import Tuple
import numpy as np


class LabelSearch(object):
    """ This class handles annotations of the collected traces and filter out
    frames that are not good based on several fields including ``time of day``,
    ``weather``, ``road type``, ``maneuver``, ``direction``, ``tag``. Please
    check data annotation process for more details on each field.

    Args:
        time_of_day (str): Time of day to be considered as good frames.
        weather (str): Weather condition to be considered as good frames.
        road_type (str): Road type to be considered as good frames.
        maneuver (str): Human maneuver to be considered as good frames.
        direction (str): Direction to be considered as good frames.
        tag (str): Other tags to be considered as good frames.

    """
    FIELDS = [
        'timestamp', 'time_of_day', 'weather', 'road_type', 'maneuver',
        'direction', 'tag'
    ]

    def __init__(self, time_of_day: str, weather: str, road_type: str,
                 maneuver: str, direction: str, tag: str) -> None:
        self._time_of_day = time_of_day
        self._weather = weather
        self._road_type = road_type
        self._maneuver = maneuver
        self._direction = direction
        self._tag = tag

    def find_good_labeled_frames(
            self, trace_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """ Find good frames based on video labels. Assume video labels have
            consistent timestampswith the specified master sensor/topic.

        Args:
            trace_path (str): path to a trace

        Returns:
            Return a tuple (``arr_1``, ``arr_2``), where ``arr_1`` is a boolean
            mask of good frames, and `arr_2` is timestamps of all good frames.
        """
        fpath = os.path.join(trace_path, 'video_labels.csv')
        has_video_label = os.path.exists(fpath)
        if has_video_label:
            is_good_frames = []
            good_timestamps = []
            with open(fpath, 'r') as f:
                reader = csv.DictReader(f,
                                        fieldnames=LabelSearch.FIELDS,
                                        delimiter=',')
                for line in reader:  # for each line
                    # assume consistency between master sensor and video labels
                    good_timestamps.append(float(line.pop('timestamp')))

                    match = True
                    for field in line.keys():
                        regex = getattr(self,
                                        '_' + field)  # get the search regex
                        res = re.search(regex, line[field])
                        if not res:
                            match = False
                            break
                    if match:
                        is_good_frames.append(True)
                    else:
                        is_good_frames.append(False)
            is_good_frames = np.array(is_good_frames)
            good_timestamps = np.array(good_timestamps)[is_good_frames]
        else:
            is_good_frames = None
            good_timestamps = None

        return is_good_frames, good_timestamps
