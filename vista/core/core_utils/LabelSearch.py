import os
import re
import csv
from typing import Tuple
import numpy as np


class LabelSearch(object):
    FIELDS = ['timestamp', 'time_of_day', 'weather', 'road_type', 'maneuver', 'direction', 'tag']

    def __init__(self, time_of_day: str, weather: str, road_type: str, maneuver: str, 
                 direction: str, tag: str) -> None:
        self._time_of_day = time_of_day
        self._weather = weather
        self._road_type = road_type
        self._maneuver = maneuver
        self._direction = direction
        self._tag = tag

    def find_good_labeled_frames(self, trace_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """ Find good frames based on video labels. Assume video labels have consistent timestamps
            with the specified master sensor/topic.

        Args:
            trace_path (str): path to a trace

        Returns:
            np.ndarray: a boolean mask of good frames
            np.ndarray: timestamps of all good frames
        """
        is_good_frames = []
        good_timestamps = []
        with open(os.path.join(trace_path, 'video_labels.csv'), 'r') as f:
            reader = csv.DictReader(f, fieldnames=LabelSearch.FIELDS, delimiter=',')
            for line in reader: # for each line
                # assume consistency between master sensor and video labels
                good_timestamps.append(float(line.pop('timestamp')))

                match = True
                for field in line.keys():
                    regex = getattr(self, '_' + field) # get the search regex
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

        return is_good_frames, good_timestamps
