'''World to handle trace data/frames.'''

import numpy as np
import os
from scipy.interpolate import interp1d
import sys
import tensorflow as tf

deepknight_root = os.environ.get('DEEPKNIGHT_ROOT')
sys.path.insert(0, os.path.join(deepknight_root, 'util/'))
from Camera import Camera
from FFWrapper import FFReader
from LabelSearch import LabelSearch, get_synced_labeled_timestamps
from MultiFrame import MultiFrame
import print_util as P
from TopicNames import TopicNames
from ViewSynthesis import ViewSynthesis


class World:
    def __init__(self, trace_path, sess, which_camera, camera, VS):
        self.trace_path = trace_path

        # Initialize configuration variables
        ''' Labels of data (can search with regex) '''
        self.labels = [
            LabelSearch('day|night', 'dry|rain|snow', 'local|residential|highway|unpaved|indoor', 'stable', '.*', '.*', 1 ),
        ]
        self.road_width = 4
        self.car_width = 2

        # Create camera and View Synthesis objects
        self.sess = sess
        self.which_camera = which_camera
        self.camera = camera
        self.view_synthesizer = VS

        # Setup variables
        self.syncedLabeledFrames = self.spawn(trace_path)
        self.masterClock = MultiFrame(trace_path)
        self.current_frame_index = 0
        self.current_trace_index = 0
        self.isCrashed = False
        self.trace_done = False
        self.syncedLabeledTimestamps = [] # Create synced and labeled timestamps for each trace
        self.num_of_frames = 0
        for trace in self.syncedLabeledFrames:
            trace_timestamps = []
            for frame_num in trace[self.which_camera]:
                trace_timestamps.append(self.masterClock.get_time_from_frame_num(self.which_camera,frame_num))
            self.syncedLabeledTimestamps.append(trace_timestamps)
            self.num_of_frames += len(trace_timestamps)
        self.f_position, self.f_speed, self.f_curvature = self.get_interp_functions()
        self.curv_reset_probs = self.get_curv_reset_probs(self.current_trace_index, self.f_curvature)

        # Initialize stream
        video = os.path.join(trace_path, self.which_camera+'.avi')
        self.stream = FFReader(video, custom_size=(self.camera.get_height(),self.camera.get_width()), verbose=False)

        self.reset() # Randomly resets to a trace in provided path

    def reset(self):
        """ Resets the environment due to the end of a RL episode. """
        # print "RESETTING WORLD"

        # Find index of next trace
        self.current_trace_index = self.find_trace_reset()
        self.curv_reset_probs = self.get_curv_reset_probs(self.current_trace_index, self.f_curvature) # Get new curv reset probs for it

        # Reset favoring places with higher curvatures
        self.current_frame_index = self.find_curvature_reset(self.curv_reset_probs)

        # First timestamp of trace
        self.first_time = self.masterClock.get_time_from_frame_num(self.which_camera, self.syncedLabeledFrames[self.current_trace_index][self.which_camera][self.current_frame_index]) # MODIFIED
        # seek_sec = self.stream.frame_to_secs(self.syncedLabeledFrames[self.which_camera][self.current_frame_index])
        seek_sec = self.stream.frame_to_secs(self.syncedLabeledFrames[self.current_trace_index][self.which_camera][self.current_frame_index]) # MODIFIED
        self.stream.seek(seek_sec)

        self.trace_done = False
        self.isCrashed = False

    def find_trace_reset(self):
        trace_reset_probs = np.zeros(len(self.syncedLabeledFrames))
        for i in range(len(self.syncedLabeledFrames)):
            trace = self.syncedLabeledFrames[i]
            trace_reset_probs[i] = len(trace[self.which_camera])
        trace_reset_probs /= np.sum(trace_reset_probs)
        new_trace = np.random.choice(trace_reset_probs.shape[0], 1, p=trace_reset_probs)
        return new_trace[0]


    def find_curvature_reset(self, curv_reset_probs):
        new_sample = np.random.choice(curv_reset_probs.shape[0], 1, p=curv_reset_probs)
        return new_sample[0]

    def get_interp_functions(self):
        # Human inverse_r from filtered odometry #TODO: move to better place, all sensors should be handled in world but can be accessed from env
        speed = np.genfromtxt(os.path.join(self.trace_path, TopicNames.speed + '.csv'), delimiter=',')
        f_speed = interp1d(speed[:,0], speed[:,1], fill_value='extrapolate')
        odometry = np.genfromtxt(os.path.join(self.trace_path, TopicNames.odometry + '.csv'), delimiter=',')
        f_position = interp1d(odometry[:,0], odometry[:,1:3], axis=0, fill_value='extrapolate')
        sample_times = odometry[:,0]
        curvature = odometry[:,4] / np.maximum(f_speed(sample_times), 1e-10)
        good_curvature_inds = np.abs(curvature) < 1/3.
        f_curvature = interp1d(sample_times[good_curvature_inds], curvature[good_curvature_inds], fill_value='extrapolate')

        return f_position, f_speed, f_curvature


    def get_curv_reset_probs(self, trace_index, f_curvature):
        # Computing probablities for resetting to places with higher curvature for current trace
        current_timestamps = self.syncedLabeledTimestamps[self.current_trace_index]
        # curvatures = np.abs(f_curvature(self.syncedLabeledTimestamps))
        curvatures = np.abs(f_curvature(current_timestamps)) # MODIFIED
        curvatures = np.clip(curvatures, 0, 1/3.)
        hist, bin_edges = np.histogram(curvatures, 100, density=False)
        bins = np.digitize(curvatures, bin_edges, right=True)
        hist_density = hist / float(np.sum(hist))
        smoothing_factor = 0.001
        curv_reset_probs = 1.0/(hist_density[bins-1]+smoothing_factor)
        curv_reset_probs /= np.sum(curv_reset_probs)
        #
        # ''' HARDCODING RESET TO END OF TRACE FOR TESTING PURPOSES'''
        # end_reset_probs = np.zeros(np.size(curv_reset_probs))
        # end_reset_probs[-10] = 1.0
        # return end_reset_probs
        # ''' END OF HARDCODING RESET TO END OF TRACE FOR TESTING PURPOSES'''

        return curv_reset_probs


    def get_next_valid_timestamp(self, human, desired_ego):
        first_time = self.first_time
        current_time = self.get_current_timestamp()
        index = self.current_frame_index
        time = self.get_timestamp(index)
        human = human.copy() # dont edit the original
        closest_dist = float('inf')
        while True:
            next_time = self.get_timestamp(index)

            last_human = human.copy()
            human.step(
                curvature=self.f_curvature(time),
                velocity=self.f_speed(time),
                delta_t=next_time-time
            )

            dist = np.linalg.norm(human.numpy()[:2] - desired_ego.numpy()[:2])
            if dist < closest_dist:
                closest_dist = dist
                index += 1
            else:
                break

        self.current_frame_index = index - 1
        closest_time = self.get_current_timestamp()
        return closest_time, last_human


    def get_frame_from_time(self, worldtime):
        # do this _once_ in the init for speed
        frame_to_worldtime = self.masterClock.cam_to_frame_time[self.which_camera]
        worldtime_to_frame = dict([[v,k] for k,v in frame_to_worldtime.items()])

        desired_frame_num = worldtime_to_frame[worldtime]

        if desired_frame_num < self.stream.frame_num:
            print "SEEKING SINCE {} < {}".format(desired_frame_num, self.stream.frame_num)
            desired_seek_sec = self.stream.frame_to_secs(desired_frame_num)
            self.stream.seek(desired_seek_sec)

        while self.stream.frame_num != desired_frame_num:
            self.stream.read()

        desired_frame = self.stream.image.copy()
        return desired_frame


    def get_timestamp(self, index):
        if index >= len(self.syncedLabeledTimestamps[self.current_trace_index])-1:
            print "END OF TRACE"
            self.trace_done = True # Done var will be set to True in deepknight env step
            return self.syncedLabeledTimestamps[self.current_trace_index][-1] # Return last timestamp
        return self.syncedLabeledTimestamps[self.current_trace_index][index]


    def get_current_timestamp(self):
        return self.get_timestamp(self.current_frame_index)


    def synthesize_frame(self, frame, info, relative_state):
        translated_frame = self.view_synthesizer(
            theta=relative_state.theta,
            translation_x=relative_state.translation_x,
            translation_y=relative_state.translation_y,
            image=frame)[0]

        translated_frame = np.uint8(translated_frame)

        return translated_frame

    def spawn(self, trace_path):
        print P.INFO("Spawning {}".format(trace_path))

        # Initialize MultiFrame object
        masterClock = MultiFrame(trace_path)
        masterTimestamp = masterClock.get_master_timestamp().reshape((-1,1))

        # Get good frames to loop through
        goodLabeledFrames = np.ones((masterTimestamp.shape[0]), dtype=bool)
        for search_query in self.labels:
            goodLabeledFrames *= search_query.find_good_labeled_frames(trace_path) # Logical AND via multiplication
        syncedLabeledTimestamps = get_synced_labeled_timestamps(trace_path, goodLabeledFrames)

        # Need to divide the stream into good chunks; each chunk becomes a Trace
        syncedLabeledFrames_ = [] # FIX time
        start = 0
        for i in range(len(syncedLabeledTimestamps)):
            # CHECK:
            # Handle case where trace goes to end inclusive
            if syncedLabeledTimestamps[i] == syncedLabeledTimestamps[-1] or \
                syncedLabeledTimestamps[i+1] - syncedLabeledTimestamps[i] >= .2 :
                print("SPAWN: FOUND NEW TRACE")
                syncedLabeledFrames = masterClock.get_frames_from_times(syncedLabeledTimestamps[start:i])
                frames = {}
                frames['camera_front'] = syncedLabeledFrames['camera_front']
                syncedLabeledFrames_.append(frames)
                start = i+1
                break

        # TODO: Ensure other components expect list of dictionaries
        return syncedLabeledFrames_
