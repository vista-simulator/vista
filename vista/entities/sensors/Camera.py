import os
import glob
from typing import Dict, List
import numpy as np
import h5py
from ffio import FFReader

from .camera_utils import CameraParams, ViewSynthesis
from .BaseSensor import BaseSensor
from ..Entity import Entity
from ...utils import logging, misc


class Camera(BaseSensor):
    def __init__(self, attach_to: Entity, config: Dict) -> None:
        """ Instantiate a Camera object.

        Args:
            attach_to (Entity): a parent object to be attached to
            config (Dict): configuration of the sensor
        """
        super(Camera, self).__init__(attach_to, config)

        self._config['rig_path'] = os.path.expanduser(self._config['rig_path'])
        self._camera_param: CameraParams = CameraParams(
            self.name, self._config['rig_path'])
        self._camera_param.resize(*self._config['size'])
        self._streams: Dict[str, FFReader] = dict()
        self._flow_streams: Dict[str, List[FFReader]] = dict()
        self._flow_meta: Dict[str, h5py.File] = dict()
        if self._config.get('use_synthesizer', True):
            self._view_synthesis: ViewSynthesis = ViewSynthesis(
                self._camera_param, self._config)
        else:
            self._view_synthesis = None

    def reset(self) -> None:
        logging.info('Camera ({}) reset'.format(self.id))

        # Close stream already open
        for stream in self.streams.values():
            stream.close()

        for flow_stream in [
                _vv for _v in self.flow_streams.values() for _vv in _v.values()
        ]:
            flow_stream.close()

        # Fetch video stream from the associated trace. All video streams are handled by the main
        # camera and shared across all cameras in an agent.
        multi_sensor = self.parent.trace.multi_sensor
        if self.name == multi_sensor.main_camera:
            for camera_name in multi_sensor.camera_names:
                # get video stream
                video_path = os.path.join(self.parent.trace.trace_path,
                                          camera_name + '.avi')
                cam_h, cam_w = self.camera_param.get_height(
                ), self.camera_param.get_width()
                stream = FFReader(video_path,
                                  custom_size=(cam_h, cam_w),
                                  verbose=False)
                self._streams[camera_name] = stream

                # seek based on timestamp
                frame_num = self.parent.trace.good_frames[camera_name][
                    self.parent.segment_index][self.parent.frame_index]
                seek_sec = self._streams[camera_name].frame_to_secs(frame_num)
                self._streams[camera_name].seek(seek_sec)

                # get flow
                flow_meta_path = os.path.join(self.parent.trace.trace_path,
                                              camera_name + '_flow_meta.h5')
                if os.path.exists(flow_meta_path):
                    self._flow_meta[camera_name] = h5py.File(
                        flow_meta_path, 'r')

                    self._flow_streams[camera_name] = dict()
                    for flow_name in self.flow_meta[camera_name].keys():
                        flow_path = os.path.join(
                            self.parent.trace.trace_path,
                            camera_name + '_flow_{}.mp4'.format(flow_name))
                        flow_stream = FFReader(flow_path, verbose=False)
                        flow_frame_num = frame_num  # flow for (frame_num, frame_num + 1)
                        flow_seek_sec = flow_stream.frame_to_secs(flow_frame_num)
                        flow_stream.seek(flow_seek_sec)
                        self._flow_streams[camera_name][flow_name] = flow_stream
                else:
                    logging.warning('No flow data')
        else:  # use shared streams from the main camera
            main_name = multi_sensor.main_camera
            main_sensor = [
                _s for _s in self.parent.sensors if _s.name == main_name
            ]
            assert len(main_sensor) == 1, 'Cannot find main sensor {}'.format(
                main_name)
            main_sensor = main_sensor[0]
            assert isinstance(main_sensor,
                              Camera), 'Main sensor is not Camera object'
            self._streams = main_sensor.streams

            self._flow_streams = main_sensor.flow_streams
            self._flow_meta = main_sensor.flow_meta

        # Add background mesh for all video stream in view synthesis
        if self.view_synthesis is not None:
            parent_sensor_dict = {_s.name: _s for _s in self.parent.sensors}
            for camera_name in self.streams.keys():
                if camera_name not in self.view_synthesis.bg_mesh_names:
                    if camera_name in parent_sensor_dict.keys():
                        camera_param = parent_sensor_dict[camera_name].camera_param
                    else:
                        camera_param = CameraParams(camera_name,
                                                    self._config['rig_path'])
                        camera_param.resize(*self._config['size'])
                    self.view_synthesis.add_bg_mesh(camera_param)

    def capture(self, timestamp: float) -> np.ndarray:
        logging.info('Camera ({}) capture'.format(self.id))

        # Get frame at the closest smaller timestamp from dataset
        multi_sensor = self.parent.trace.multi_sensor
        if self.name == multi_sensor.main_camera:
            fetch_smaller = self.flow_streams != dict() # only when using optical flow
            all_frame_nums = multi_sensor.get_frames_from_times([timestamp], fetch_smaller)
            for camera_name in multi_sensor.camera_names:
                # rgb camera stream
                stream = self.streams[camera_name]
                frame_num = all_frame_nums[camera_name][0]

                if frame_num < stream.frame_num:
                    seek_sec = stream.frame_to_secs(frame_num)
                    stream.seek(seek_sec)

                while stream.frame_num != frame_num:
                    stream.read()

                # flow stream
                if self.flow_streams != dict():
                    flow_frame_num = frame_num  # flow for (frame_num, frame_num + 1)
                    for flow_stream in self.flow_streams[camera_name].values():
                        if flow_frame_num < flow_stream.frame_num:
                            flow_seek_sec = flow_stream.frame_to_secs(
                                flow_frame_num)
                            flow_stream.seek(flow_seek_sec)

                        while flow_stream.frame_num != flow_frame_num:
                            flow_stream.read()

        frames = dict()
        for camera_name in multi_sensor.camera_names:
            frames[camera_name] = self.streams[camera_name].image.copy()

        # Interpolate frame at the exact timestamp
        if self.flow_streams != dict():
            for camera_name in self.flow_streams.keys():
                frame = frames[camera_name]

                flow = dict()
                for flow_name, flow_minmax in self.flow_meta[camera_name].items():
                    flow_stream = self.flow_streams[camera_name][flow_name]
                    flow[flow_name] = misc.img2flow(
                        flow_stream.image.copy(),
                        flow_minmax[int(flow_stream.frame_num)], frame.shape[:2])

                frame_num = int(self.streams[camera_name].frame_num)
                curr_ref_ts = multi_sensor.get_time_from_frame_num(
                    camera_name, frame_num)
                next_ref_ts = multi_sensor.get_time_from_frame_num(
                    camera_name, frame_num + 1)

                logging.warning('Stream frame number exceed 1 non-intentionally')
                self.streams[camera_name].read(
                )  # NOTE: stream frame number exceed 1 here
                next_frame = self.streams[camera_name].image.copy()
                frames[camera_name] = misc.biinterp(frame, next_frame,
                                                    flow['forward'],
                                                    flow['backward'], timestamp,
                                                    curr_ref_ts, next_ref_ts)

        # Synthesis by rendering
        if self.view_synthesis is not None:
            lat, long, yaw = self.parent.relative_state.numpy()
            trans = np.array([lat, 0., -long])
            rot = np.array([0., yaw, 0.])
            rendered_frame, _ = self.view_synthesis.synthesize(trans, rot, frames)
        else:
            rendered_frame = frames[self.name]

        return rendered_frame

    @property
    def config(self) -> Dict:
        return self._config

    @property
    def camera_param(self) -> CameraParams:
        return self._camera_param

    @property
    def streams(self) -> Dict[str, FFReader]:
        return self._streams

    @property
    def flow_streams(self) -> Dict[str, List[FFReader]]:
        return self._flow_streams

    @property
    def flow_meta(self) -> Dict[str, h5py.File]:
        return self._flow_meta

    @property
    def view_synthesis(self) -> ViewSynthesis:
        return self._view_synthesis

    def __repr__(self) -> str:
        return '<{} (id={})> '.format(self.__class__.__name__, self.id) + \
               'name: {} '.format(self.name) + \
               'size: {}x{} '.format(self.camera_param.get_height(),
                                     self.camera_param.get_width()) + \
               '#streams: {} '.format(len(self.streams))
