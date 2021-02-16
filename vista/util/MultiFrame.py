'''
This class handles the camera timestamps for multiple video streams.
It is an interface to synchronize the streams and find frames recorded at the
same instant for all cameras.
@author Alexander Amini
@email  amini@mit.edu
@date   Nov 24, 2017
'''

import csv
import os
from collections import OrderedDict
import numpy as np
from scipy.interpolate import interp1d

from . import TopicNames

class Sensors:
    def __init__(self, trace_dir):
        self.trace_dir = trace_dir

    def get_func(self, topic, t=0, y=[1], fill_value='extrapolate'):
        data = np.genfromtxt(os.path.join(self.trace_path, topic+'.csv'), delimiter=',')
        func = interp1d(data[:,t],data[:,y], axis=0, fill_value=fill_value)
        return func

class MultiFrame:
    def __init__(self, trace_dir):
        # trace_dir is the path to a trace created with extract_bag.py
        self.trace_dir = trace_dir

        self.cam_to_frame_time = dict()
        for filename in os.listdir(trace_dir):
            if not filename.startswith(".") and filename.endswith(".avi") and "segmentation" not in filename:
                sensor = os.path.splitext(filename)[0]
                data = np.genfromtxt(os.path.join(trace_dir, sensor + '.csv'), delimiter=',', dtype=np.float64)
                frame_to_time = dict()
                for i in range(data.shape[0]):
                    frame_to_time[int(data[i,0])] = data[i,1]

                self.cam_to_frame_time[sensor] = frame_to_time

        with open(os.path.join(trace_dir, 'master_clock.csv')) as csvfile:
            reader = csv.DictReader(csvfile)
            self.master_time_to_frame = {name: [] for name in reader.fieldnames}
            for row in reader:
                for name in reader.fieldnames:
                    self.master_time_to_frame[name].append(float(row[name]))

        self.fieldnames = reader.fieldnames
        self.camera_names = reader.fieldnames
        self.camera_names.remove('timestamp')

        self.master_counter = 0

    ''' returns the cameras collected in the trace '''
    def get_camera_names(self):
        return self.cameras_names

    ''' compute the timestamp associated to a frame in a video
        return None if we dont have information about that frame '''
    def get_time_from_frame_num(self, camera, frame_num):
        return self.cam_to_frame_time[camera].get(frame_num, None)

    ''' takes in a list of timestamps and returns corresponding
        frame numbers for each camera '''
    def get_frames_from_times(self, timestamps):

        # get the index of every desired timestamp in the master list
        _, index_to_timestamps = ismember(timestamps, self.master_time_to_frame['timestamp'])
        frames = dict()
        for camera in self.camera_names:
            # same as: frames[camera] = self.master_time_to_frame[camera][index_to_timestamps]
            frames[camera] = [self.master_time_to_frame[camera][i] for i in index_to_timestamps]

        return frames

    ''' checks if a frame was recorded at a time when all the others were also recorded '''
    def is_good_frame(self, camera, frame):
        time = self.get_time_from_frame_num(camera, frame)
        if time is not None and time in self.master_time_to_frame['timestamp']:
            return True
        return False

    ''' returns the timestamp and frame number for each of the cameras at given master index '''
    def get_multi_frames(self, i):
        out = {'timestamp':self.master_time_to_frame['timestamp'][i]}
        for field in self.fieldnames:
            out[field] = self.master_time_to_frame[field][i]
        return out

    ''' returns good timestamps '''
    def get_master_timestamp(self):
        return np.array(self.master_time_to_frame['timestamp'])

    ''' returns good timestamps '''
    def num_master_timestamps(self):
        return len(self.master_time_to_frame['timestamp'])

    def has_all_cameras(self, cameras):
        for cam in cameras:
            if cam not in self.camera_names:
                return False
        return True

def ismember(a_vec, b_vec):
    """ MATLAB equivalent ismember function """

    bool_ind = np.isin(a_vec,b_vec)
    common = a_vec[bool_ind]
    common_unique, common_inv  = np.unique(common, return_inverse=True)     # common = common_unique[common_inv]
    b_unique, b_ind = np.unique(b_vec, return_index=True)  # b_unique = b_vec[b_ind]
    common_ind = b_ind[np.isin(b_unique, common_unique, assume_unique=True)]
    return bool_ind, common_ind[common_inv]

# TESTS
# m = MultiFrame('../data/traces/blue_prius_stata3cam_2017-11-24-14-26-56')
# print m.is_good_frame('camera_front',127)
# print m.get_next_multi_frames()
