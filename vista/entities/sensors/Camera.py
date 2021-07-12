import os
from typing import Dict, List
import numpy as np
from ffio import FFReader

from .camera_utils import CameraParams, ViewSynthesis
from .BaseSensor import BaseSensor
from ..Entity import Entity
from ...utils import logging


class Camera(BaseSensor):
    def __init__(self, attach_to: Entity, config: Dict) -> None:
        """ Instantiate a Camera object.

        Args:
            attach_to (Entity): a parent object to be attached to
            config (Dict): configuration of the sensor
        """
        super(Camera, self).__init__(attach_to, config)

        self._config['rig_path'] = os.path.expanduser(self._config['rig_path'])
        self._camera_param: CameraParams = CameraParams(self.name, 
                                                        self._config['rig_path'])
        self._camera_param.resize(*self._config['size'])
        self._streams: Dict[str, FFReader] = dict()
        self._view_synthesis: ViewSynthesis = ViewSynthesis(self._camera_param, self._config)

    def reset(self) -> None:
        logging.info('Camera ({}) reset'.format(self.id))

        # Close stream already open
        for stream in self.streams.values():
            stream.close()

        # Fetch video stream from the associated trace. All video streams are handled by the master
        # sensor and shared across all cameras. This requires master sensor to be a camera.
        multi_sensor = self.parent.trace.multi_sensor
        if self.name == multi_sensor.master_sensor:
            for camera_name in multi_sensor.camera_names:
                # get video stream
                video_path = os.path.join(self.parent.trace.trace_path, camera_name + '.avi')
                cam_h, cam_w = self.camera_param.get_height(), self.camera_param.get_width()
                stream = FFReader(video_path, custom_size=(cam_h, cam_w), verbose=False)
                self._streams[camera_name] = stream

                # seek based on timestamp
                frame_num = self.parent.trace.good_frames[camera_name] \
                    [self.parent.segment_index][self.parent.frame_index]
                seek_sec = self._streams[camera_name].frame_to_secs(frame_num)
                self._streams[camera_name].seek(seek_sec)
        else: # use shared streams from the master camera
            master_name = multi_sensor.master_sensor
            master = [_s for _s in self.parent.sensors if _s.name == master_name]
            assert len(master) == 1, 'Cannot find master sensor {}'.format(master_name)
            master = master[0]
            assert isinstance(master, Camera), 'Master sensor is not Camera object'
            self._streams = master.streams

        # Add background mesh for all video stream in view synthesis
        parent_sensor_dict = {_s.name: _s for _s in self.parent.sensors}
        for camera_name in self.streams.keys():
            if camera_name not in self.view_synthesis.bg_mesh_names:
                if camera_name in parent_sensor_dict.keys():
                    camera_param = parent_sensor_dict[camera_name].camera_param
                else:
                    camera_param = CameraParams(camera_name, self._config['rig_path'])
                    camera_param.resize(*self._config['size'])
                self.view_synthesis.add_bg_mesh(camera_param)

    def capture(self, timestamp: float) -> np.ndarray:
        logging.info('Camera ({}) capture'.format(self.id))

        # Get frame at the closest smaller timestamp from dataset
        multi_sensor = self.parent.trace.multi_sensor
        if self.name == multi_sensor.master_sensor:
            all_frame_nums = multi_sensor.get_frames_from_times([timestamp])
            for camera_name in multi_sensor.camera_names:
                stream = self.streams[camera_name]
                frame_num = all_frame_nums[camera_name][0]

                if frame_num < stream.frame_num:
                    seek_sec = stream.frame_to_secs(frame_num)
                    stream.seek(seek_sec)

                while stream.frame_num != frame_num:
                    stream.read()
        
        frames = dict()
        for camera_name in multi_sensor.camera_names:
            frames[camera_name] = self.streams[camera_name].image.copy()

        # TODO: Interpolate frame at the exact timestamp
        logging.warning('Frame interpolation at exact timestamp not implemented')

        # Synthesis by rendering
        lat, long, yaw = self.parent.relative_state.numpy()
        trans = np.array([lat, 0., -long])
        rot = np.array([0., yaw, 0.])
        rendered_frame, _ = self.view_synthesis.synthesize(trans, rot, frames)

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
    def view_synthesis(self) -> ViewSynthesis:
        return self._view_synthesis

    def __repr__(self) -> str:
        return '<{} (id={})> '.format(self.__class__.__name__, self.id) + \
               'name: {} '.format(self.name) + \
               'size: {}x{} '.format(self.camera_param.get_height(), 
                                     self.camera_param.get_width()) + \
               '#streams: {} '.format(len(self.streams))
