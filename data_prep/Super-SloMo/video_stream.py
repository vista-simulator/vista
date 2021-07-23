# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located at docs.prophesee.ai/licensing and in the "LICENSE" file accompanying
# this file.
"""
Video .mp4 or .avi Iterator
Right now backend is OpenCV
"""
from __future__ import absolute_import

import os, glob, random, time
import numpy as np
import cv2


class TimedVideoStream(object):
    """TimedVideoStream:
    Wrapper opening both a video stream and
    a file of timestamps.
    If it does not exist,
    it generates them with a regular period of
    1/default_fps.

    """
    def __init__(
        self,
        video_filename,
        height=-1,
        width=-1,
        seek_frame=0,
        max_frames=-1,
        random_start=False,
        rgb=False,
        default_fps=240
    ):
        self.video = VideoStream(video_filename,
                             height,width,seek_frame,
                             max_frames,random_start,
                             rgb)

        ts_path = os.path.splitext(video_filename)[0] + '_ts.npy'
        if os.path.exists(ts_path):
            self.timestamps = np.load(ts_path) * 1e6
        else:
            self.timestamps = np.linspace(0, (len(self.video) / default_fps) * 1e6, len(self.video))

        self.height, self.width = self.video.height, self.video.width

        if random_start and seek_frame > 0:
            self.timestamps = self.timestamps[seek_frame:]

    def __iter__(self):
        return zip(iter(self.video), iter(self.timestamps))

    def __len__(self):
        return len(self.timestamps)


class VideoStream(object):
    """VideoStream: a video iterator

    Args:
        video_filename (str): path to video
        height (int): desired height (original if -1)
        width (int): desired width (original if -1)
        seek_frame (int): desired first frame
        random_start (bool): randomly start anywhere
        rgb (bool): send color images
    """
    def __init__(
        self,
        video_filename,
        height=-1,
        width=-1,
        seek_frame=0,
        max_frames=-1,
        random_start=False,
        rgb=False,
    ):
        self.height = height
        self.width = width
        self.random_start = random_start
        self.max_frames = max_frames
        self.rgb = rgb

        self.filename = video_filename
        self.cap = cv2.VideoCapture(video_filename)

        if self.random_start:
            num_frames = int(self.cap.get(cv2.cv2.CAP_PROP_FRAME_COUNT))
            seek_frame = random.randint(0, num_frames // 2)
            if seek_frame > 0:
                self.cap.set(cv2.cv2.CAP_PROP_POS_FRAMES, seek_frame)
        else:
            seek_frame = 0 if seek_frame == -1 else seek_frame
            if seek_frame > 0:
                self.cap.set(cv2.cv2.CAP_PROP_POS_FRAMES, seek_frame)
        self.start = seek_frame
        if self.height == -1 or self.width == -1:
            self.height, self.width = self.original_size()

    def original_size(self):
        height, width = (
            self.cap.get(cv2.cv2.CAP_PROP_FRAME_HEIGHT),
            self.cap.get(cv2.cv2.CAP_PROP_FRAME_WIDTH),
        )
        return int(height), int(width)

    def __len__(self):
        if self.max_frames > -1:
            return self.max_frames
        else:
            num_frames = int(self.cap.get(cv2.cv2.CAP_PROP_FRAME_COUNT))
            return num_frames - self.start

    def __next__(self):

        if not self.cap:
            raise StopIteration

        ret, frame = self.cap.read()

        if ret:
            if self.rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (self.width, self.height), 0, 0, cv2.INTER_AREA)
        else:
            print('error reading the frame')
        return ret, frame

    def __iter__(self):
        for i in range(len(self)):
            ret, frame = next(self)
            if not ret:
                continue
            yield frame
