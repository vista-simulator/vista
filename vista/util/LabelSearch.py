import csv
import numpy as np
import re
import os

class LabelSearch(object):
    ''' struct to hold labels to accept when building caches '''

    FIELDS = ['timestamp','time_of_day','weather','road_type','maneuver','direction','tag']

    def __init__(self, time_of_day, weather, road_type, maneuver, direction, tag, probability=1):
        self.time_of_day = time_of_day
        self.weather = weather
        self.road_type = road_type
        self.maneuver = maneuver
        self.direction = direction
        self.tag = tag
        self.probability = probability

    def find_good_labeled_frames(self, trace_path):
        good_frames = []
        with open(os.path.join(trace_path, 'video_labels.csv'), "r") as f:
            reader = csv.DictReader(f, fieldnames=LabelSearch.FIELDS, delimiter=",")
            for line in reader: # for each line
                timestamp = line.pop('timestamp')
                match = True
                for field in line.keys():
                    regex = getattr(self, field) # get the search regex
                    res = re.search(regex, line[field])
                    if not res:
                        match = False
                        break
                if match:
                    good_frames.append(True)
                else:
                    good_frames.append(False)

        return np.array(good_frames)

def get_synced_labeled_timestamps(trace_path, good_label_bitmap):
    all_synced_timestamps = []
    with open(os.path.join(trace_path, 'video_labels.csv'), "r") as f:
        reader = csv.DictReader(f, fieldnames=LabelSearch.FIELDS, delimiter=",")
        for line in reader: # for each line
            timestamp = np.float64(line.pop('timestamp'))
            all_synced_timestamps.append(timestamp)
    synced_labeled_timestamps = np.array(all_synced_timestamps)[good_label_bitmap]
    return synced_labeled_timestamps
