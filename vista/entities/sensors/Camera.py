from ffio import FFReader
import numpy as np
import os

from .BaseSensor import *
from ...util import ViewSynthesis
from ...util import Camera as CameraParams


class Camera(BaseSensor):
    def __init__(self, attach_to=None):
        super(Camera, self).__init__()

        self._parent = attach_to

        # Camera sensors synthesize new images from a view synthesizer
        self.which_camera = self.parent.trace.which_camera
        self.camera = CameraParams(self.which_camera)
        self.camera.resize(250, 400)  #Hardcode FIXME
        self.view_synthesizer = ViewSynthesis(self.camera).synthesize

        # Reset the sensor based on the position of the parent
        self.reset()

    def capture(self, timestamp):
        frame = self.get_frame_from_time(timestamp)
        translated_frame = self.synthesize_frame(frame,
                                                 self.parent.relative_state)
        return translated_frame

    def synthesize_frame(self, frame, relative_state):
        translated_frame = self.view_synthesizer(
            theta=relative_state.theta,
            translation_x=relative_state.translation_x,
            translation_y=relative_state.translation_y,
            image=frame)[0]
        translated_frame = np.uint8(translated_frame)

        return translated_frame

    def get_frame_from_time(self, worldtime):
        # do this _once_ in the init for speed
        frame_to_worldtime = self.trace.masterClock.cam_to_frame_time[
            self.which_camera]
        worldtime_to_frame = dict([[v, k]
                                   for k, v in frame_to_worldtime.items()])

        desired_frame_num = worldtime_to_frame[worldtime]

        if desired_frame_num < self.stream.frame_num:
            print("SEEKING SINCE {} < {}".format(desired_frame_num,
                                                 self.stream.frame_num))
            desired_seek_sec = self.stream.frame_to_secs(desired_frame_num)
            self.stream.seek(desired_seek_sec)

        while self.stream.frame_num != desired_frame_num:
            self.stream.read()

        desired_frame = self.stream.image.copy()
        return desired_frame

    def reset(self):
        car = self.parent
        self.trace = car.world.traces[car.current_trace_index]

        # Initialize stream
        video = os.path.join(self.trace.trace_path, self.which_camera + '.avi')
        if hasattr(self, 'stream'): # NOTE: need to close first otherwise it breaks pipe
            self.stream.close()
        self.stream = FFReader(video, custom_size=(250, 400), verbose=False)

        seek_sec = self.stream.frame_to_secs(
            self.trace.syncedLabeledFrames[car.current_trace_index][
                self.trace.which_camera][car.current_frame_index])  # MODIFIED
        self.stream.seek(seek_sec)

    def __repr__(self):
        return f"Camera(id={self.id})"
