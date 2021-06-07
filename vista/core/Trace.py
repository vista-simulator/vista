import numpy as np
import os
from scipy.interpolate import interp1d

from ..util import MultiFrame, LabelSearch, TopicNames, \
    get_synced_labeled_timestamps


class Trace:
    def __init__(self, trace_path, reset_mode='default'):

        print("Spawning {}".format(trace_path))

        self.trace_path = trace_path
        self.reset_mode = reset_mode
        self.road_width = 4  # meters
        self.which_camera = "front_center"  # HARDCODING for now...

        self.masterClock = MultiFrame(trace_path)
        self.syncedLabeledFrames = self.getSyncedLabeledFrames(trace_path)

        # Create synced and labeled timestamps for each segment
        self.syncedLabeledTimestamps = []
        self.num_of_frames = 0
        for segment in self.syncedLabeledFrames:
            segment_timestamps = []
            for frame_num in segment[self.which_camera]:
                segment_timestamps.append(
                    self.masterClock.get_time_from_frame_num(
                        self.which_camera, frame_num))
            self.syncedLabeledTimestamps.append(segment_timestamps)
            self.num_of_frames += len(segment_timestamps)

        # Setup some state information functions and reset probabilities
        (self.f_position, self.f_speed, self.f_curvature, self.f_distance) = \
            self.get_interp_functions(trace_path)

    def getSyncedLabeledFrames(self, trace_path):

        labels = [
            LabelSearch('day|night', 'dry|rain|snow',
                        'local|residential|highway|unpaved|indoor', 'stable',
                        '.*', '.*', 1),
        ]

        # Initialize MultiFrame object
        masterTimestamp = self.masterClock.get_master_timestamp()
        masterTimestamp = masterTimestamp.reshape((-1, 1))

        # Get good frames to loop through
        goodLabeledFrames = np.ones((masterTimestamp.shape[0]), dtype=bool)
        for search_query in labels:
            goodLabeledFrames *= search_query.find_good_labeled_frames(
                trace_path)  # Logical AND via multiplication
        syncedLabeledTimestamps = get_synced_labeled_timestamps(
            trace_path, goodLabeledFrames)

        # Need to divide the stream into good chunks; each chunk becomes a Trace
        syncedLabeledFrames_ = []  # FIX time
        start = 0
        for i in range(len(syncedLabeledTimestamps)):
            # CHECK:
            # Handle case where trace goes to end inclusive
            if syncedLabeledTimestamps[i] == syncedLabeledTimestamps[-1] or \
                    syncedLabeledTimestamps[i + 1] - syncedLabeledTimestamps[i] >= .2:
                syncedLabeledFrames = self.masterClock.get_frames_from_times(
                    syncedLabeledTimestamps[start:i])
                frames = {}
                frames['front_center'] = syncedLabeledFrames['front_center']
                syncedLabeledFrames_.append(frames)
                start = i + 1

        # TODO: Ensure other components expect list of dictionaries
        return syncedLabeledFrames_

    def get_interp_functions(self, trace_path):
        # Human inverse_r from filtered odometry
        speed = np.genfromtxt(os.path.join(trace_path,
                                           TopicNames.speed + '.csv'),
                              delimiter=',')
        distance = np.genfromtxt(os.path.join(trace_path,
                                              TopicNames.distance + '.csv'),
                                 delimiter=',')
        odometry = np.genfromtxt(os.path.join(trace_path,
                                              TopicNames.odometry + '.csv'),
                                 delimiter=',')

        f_speed = interp1d(speed[:, 0], speed[:, 1], fill_value='extrapolate')
        f_position = interp1d(odometry[:, 0],
                              odometry[:, 1:3],
                              axis=0,
                              fill_value='extrapolate')

        sample_times = odometry[:, 0]
        curvature = odometry[:, 4] / np.maximum(f_speed(sample_times), 1e-10)
        good_curvature_inds = np.abs(curvature) < 1 / 3.
        f_curvature = interp1d(sample_times[good_curvature_inds],
                               curvature[good_curvature_inds],
                               fill_value='extrapolate')

        f_distance = interp1d(distance[:, 0],
                              distance[:, 1],
                              fill_value='extrapolate')

        return f_position, f_speed, f_curvature, f_distance

    def get_curv_reset_probs(self, segment_index):
        if self.reset_mode == 'default':
            # Computing probablities for resetting to places with higher curvature for current trace
            current_timestamps = self.syncedLabeledTimestamps[segment_index]
            # curvatures = np.abs(f_curvature(self.syncedLabeledTimestamps))
            curvatures = np.abs(self.f_curvature(current_timestamps))  # MODIFIED
            curvatures = np.clip(curvatures, 0, 1 / 3.)
            hist, bin_edges = np.histogram(curvatures, 100, density=False)
            bins = np.digitize(curvatures, bin_edges, right=True)
            hist_density = hist / float(np.sum(hist))
            smoothing_factor = 0.001
            curv_reset_probs = 1.0 / (hist_density[bins - 1] + smoothing_factor)
            curv_reset_probs /= np.sum(curv_reset_probs)
        elif self.reset_mode == 'uniform':
            n_timestamps = len(self.syncedLabeledTimestamps[segment_index])
            curv_reset_probs = np.ones((n_timestamps,)) / n_timestamps
        else:
            raise NotImplementedError('Unrecognized curve reset mode {}'.format(self.reset_mode))

        # ''' HARDCODING RESET TO END OF TRACE FOR TESTING PURPOSES'''
        # end_reset_probs = np.zeros(np.size(curv_reset_probs))
        # end_reset_probs[-10] = 1.0
        # return end_reset_probs
        # ''' END OF HARDCODING RESET TO END OF TRACE FOR TESTING PURPOSES'''

        return curv_reset_probs

    def find_curvature_reset(self, curv_reset_probs):
        new_sample = np.random.choice(curv_reset_probs.shape[0],
                                      1,
                                      p=curv_reset_probs)
        return new_sample[0]

    def find_segment_reset(self):
        segment_reset_probs = np.zeros(len(self.syncedLabeledFrames))
        for i in range(len(self.syncedLabeledFrames)):
            segment = self.syncedLabeledFrames[i]
            segment_reset_probs[i] = len(segment[self.which_camera])
        segment_reset_probs /= np.sum(segment_reset_probs)
        new_segment = np.random.choice(segment_reset_probs.shape[0],
                                       size=1,
                                       p=segment_reset_probs)
        return new_segment[0]
