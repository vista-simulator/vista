import os
import sys
from typing import Dict, List, Optional
import numpy as np
from ffio import FFReader
import cv2
import torch

from .camera_utils import CameraParams, ViewSynthesis
from .BaseSensor import BaseSensor
from .Camera import Camera
from ..Entity import Entity
from ...utils import logging, misc


class EventCamera(BaseSensor):
    def __init__(self, attach_to: Entity, config: Dict) -> None:
        """ Instantiate an EventCamera object.

        Args:
            attach_to (Entity): a parent object to be attached to
            config (Dict): configuration of the sensor
        """
        super(EventCamera, self).__init__(attach_to, config)

        self._config['rig_path'] = os.path.expanduser(self._config['rig_path'])
        self._camera_param: CameraParams = CameraParams(
            self.name, self._config['rig_path'])
        self._camera_param.resize(*self._config['size'])
        self._base_camera_param: CameraParams = CameraParams(
            self._config['base_camera_name'], self._config['rig_path'])
        self._base_camera_param.resize(*self._config['base_size'])
        self._streams: Dict[str, FFReader] = dict()
        vs_cam_param = self._base_camera_param if \
            self.config['reproject_pixel'] else self._camera_param
        self._view_synthesis: ViewSynthesis = ViewSynthesis(
            vs_cam_param, self._config, init_with_bg_mesh=False)

        # load video interpolation model
        sys.path.append(os.path.abspath(self._config['optical_flow_root']))
        from slowmo_warp import SlowMoWarp
        self._config['checkpoint'] = os.path.abspath(self._config['checkpoint'])
        height = self._config['base_size'][0] if \
            self.config['reproject_pixel'] else self._config['size'][0]
        width = self._config['base_size'][1] if \
            self.config['reproject_pixel'] else self._config['size'][1]
        self._interp = SlowMoWarp(height=height,
                                  width=width,
                                  checkpoint=self._config['checkpoint'],
                                  lambda_flow=self._config['lambda_flow'],
                                  cuda=self._config['use_gpu'])
        self._prev_frame: np.ndarray = None
        self._prev_timestamp: float = None

    def reset(self) -> None:
        logging.info('Event camera ({}) reset'.format(self.id))

        # Close stream already open
        for stream in self.streams.values():
            stream.close()

        # Fetch video stream from the associated trace. This is mostly the same as how we handle
        # video stream in Camera object. The difference and the reason of not sharing with Camera
        # is (1) EventCamera can be a standalone object to be initiated and more importantly (2)
        multi_sensor = self.parent.trace.multi_sensor
        if self.name == multi_sensor.main_event_camera:
            for camera_name in multi_sensor.camera_names: # base cameras are regular cameras
                # get video stream
                video_path = os.path.join(self.parent.trace.trace_path,
                                          camera_name + '.avi')
                cam_h, cam_w = self.base_camera_param.get_height(
                ), self.base_camera_param.get_width()
                stream = FFReader(video_path,
                                  custom_size=(cam_h, cam_w),
                                  verbose=False)
                self._streams[camera_name] = stream

                # seek based on timestamp
                frame_num = self.parent.trace.good_frames[camera_name][
                    self.parent.segment_index][self.parent.frame_index]
                seek_sec = self._streams[camera_name].frame_to_secs(frame_num)
                self._streams[camera_name].seek(seek_sec)
        else:  # use shared streams from the main camera
            main_name = multi_sensor.main_event_camera
            main_sensor = [
                _s for _s in self.parent.sensors if _s.name == main_name
            ]
            assert len(main_sensor) == 1, 'Cannot find main sensor {}'.format(
                main_name)
            main_sensor = main_sensor[0]
            assert isinstance(main_sensor,
                              Camera), 'Main sensor is not Camera object'
            self._streams = main_sensor.streams
        
        # Add background mesh for all video stream in view synthesis. Note that we use camera 
        # parameters from the base camera since 2D-to-3D part is determined by the regular RGB video
        # streams. Reprojecting back to 2D is then be handled by event camera configuration
        parent_sensor_dict = {_s.name: _s for _s in self.parent.sensors}
        for camera_name in self.streams.keys():
            if camera_name not in self.view_synthesis.bg_mesh_names:
                if camera_name in parent_sensor_dict.keys():
                    camera_param = parent_sensor_dict[camera_name].camera_param
                else:
                    camera_param = CameraParams(camera_name,
                                                self._config['rig_path'])
                    camera_param.resize(*self._config['base_size'])
                self.view_synthesis.add_bg_mesh(camera_param)

        # Reset previous frame
        self._prev_frame = None

    def capture(self, timestamp: float, 
                update_rgb_frame_only: Optional[bool] = False) -> np.ndarray:
        logging.info('Event camera ({}) capture'.format(self.id))

        # Get frame at the closest smaller timestamp from dataset. In event camera simulation, we 
        # don't use flow interpolation since event generation is already continuous-time
        multi_sensor = self.parent.trace.multi_sensor
        if self.name == multi_sensor.main_event_camera:
            all_frame_nums = multi_sensor.get_frames_from_times([timestamp], fetch_smaller=False)
            for camera_name in multi_sensor.camera_names:
                # rgb camera stream
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
        
        # Synthesis by rendering with compensation of relative pose between event camera parameters
        # and the base (RGB) camera parameters since we set the view synthesizer with event camera
        # parameters and project frame to 3D meshes using base camera parameters
        lat, long, yaw = self.parent.relative_state.numpy()
        trans = np.array([lat, 0., -long])
        rot = np.array([0., yaw, 0.])
        rendered_frame, _ = self.view_synthesis.synthesize(trans, rot, frames)

        # work with RGB instead of BGR; need copy otherwise fail at converting to tensor
        rendered_frame = rendered_frame[:,:,::-1].copy()

        if update_rgb_frame_only:
            self._prev_frame = rendered_frame
            self._prev_timestamp = timestamp
        else:
            events = [[], []]
            if self.prev_frame is not None:
                with torch.no_grad():
                    out = self._interp.forward_warp(self.prev_frame, rendered_frame, 
                                                    max_sf=self.config['max_sf'],
                                                    use_max_flow=True)
                
                interp = [self.prev_frame] + out['interpolated']
                if len(interp) >= 2:
                    last_interp_frame = cv2.cvtColor(interp[0], cv2.COLOR_RGB2GRAY)
                    dt = (timestamp - self.prev_timestamp) / out['sf']
                    event_timestamp = self.prev_timestamp
                    for interp_frame in interp[1:]:
                        interp_frame = cv2.cvtColor(interp_frame, cv2.COLOR_RGB2GRAY)
                        log_d_interp = np.log(interp_frame / 255. + 1e-8) - \
                                    np.log(last_interp_frame / 255. + 1e-8)
                        sample_fn = lambda mu, sig: np.clip(np.random.normal(mu, sig), 0., 1.)
                        positive_C = sample_fn(self.config['positive_threshold'], 
                                            self.config['sigma_positive_threshold'])
                        negative_C = -sample_fn(-self.config['negative_threshold'],
                                                self.config['sigma_negative_threshold'])
                        if self.config['reproject_pixel']:
                            # TODO: don't need to recompute this every time
                            cam_h = self.camera_param.get_height()
                            cam_w = self.camera_param.get_width()
                            world_rays = self.view_synthesis._world_rays[self.base_camera_param.name]
                            K = self.camera_param.get_K()
                            uv = np.matmul(K, world_rays).T[:,:2][:,::-1]

                            def extract_uv(_mask):
                                _fl_mask = _mask.flatten()
                                _uv = uv[_fl_mask]
                                
                                _mode = 1
                                if _mode == 0:
                                    _uv = np.round(_uv).astype(np.int)
                                elif _mode == 1:
                                    _uv_floor = np.floor(_uv)
                                    _uv_ceil = np.ceil(_uv)
                                    diff_floor = np.abs(_uv - _uv_floor)
                                    diff_ceil = np.abs(_uv - _uv_ceil)

                                    n_uv = _uv.shape[0]
                                    zeros = np.zeros((n_uv,))
                                    _uv_score = np.concatenate([
                                        np.stack([_uv_floor[:,0], _uv_floor[:,1], zeros], axis=1),
                                        np.stack([_uv_floor[:,0], _uv_ceil[:,1], zeros], axis=1),
                                        np.stack([_uv_ceil[:,0], _uv_floor[:,1], zeros], axis=1),
                                        np.stack([_uv_ceil[:,0], _uv_ceil[:,1], zeros], axis=1)], 
                                        axis=0)

                                    _uv_score[:n_uv,2] = diff_ceil[:,0] * diff_ceil[:,1]
                                    _uv_score[n_uv:2*n_uv,2] = diff_ceil[:,0] * diff_floor[:,1]
                                    _uv_score[2*n_uv:3*n_uv,2] = diff_floor[:,0] * diff_ceil[:,1]
                                    _uv_score[3*n_uv:,2] = diff_floor[:,0] * diff_floor[:,1]

                                    _uv = _uv_score[_uv_score[:,2]>=0.2, :2].astype(np.int)
                                else:
                                    _uv_floor = np.floor(_uv)
                                    _uv_ceil = np.ceil(_uv)
                                    _uv = np.concatenate([
                                        np.stack([_uv_floor[:,0], _uv_floor[:,1]], axis=1),
                                        np.stack([_uv_floor[:,0], _uv_ceil[:,1]], axis=1),
                                        np.stack([_uv_ceil[:,0], _uv_floor[:,1]], axis=1),
                                        np.stack([_uv_ceil[:,0], _uv_ceil[:,1]], axis=1)], axis=0)
                                    _uv = _uv.astype(np.int)
                                
                                _valid_mask = (_uv[:,0] > 0) & (_uv[:,0] < cam_h) & \
                                            (_uv[:,1] > 0) & (_uv[:,1] < cam_w)
                                _uv = _uv[_valid_mask]
                                return _uv

                            positive_uv = extract_uv(log_d_interp >= positive_C)
                            negative_uv = extract_uv(log_d_interp <= negative_C)
                        else:
                            positive_uv = np.argwhere(log_d_interp >= positive_C)
                            negative_uv = np.argwhere(log_d_interp <= negative_C)
                        event_timestamp += dt
                        event_timestamp_us = (event_timestamp * 1e6).astype(np.int64)
                        positive_events = np.concatenate([positive_uv, 
                            np.tile([event_timestamp_us, 1], (positive_uv.shape[0], 1))], axis=1)
                        negative_events = np.concatenate([negative_uv, 
                            np.tile([event_timestamp_us, 1], (negative_uv.shape[0], 0))], axis=1)
                        events[0].append(positive_events)
                        events[1].append(negative_events)
                        last_interp_frame = interp_frame

            self._prev_frame = rendered_frame
            self._prev_timestamp = timestamp

            return events

    @property
    def config(self) -> Dict:
        return self._config
    
    @property
    def streams(self) -> Dict[str, FFReader]:
        return self._streams

    @property
    def camera_param(self) -> CameraParams:
        return self._camera_param

    @property
    def base_camera_param(self) -> CameraParams:
        return self._base_camera_param

    @property
    def view_synthesis(self) -> ViewSynthesis:
        return self._view_synthesis

    @property
    def prev_frame(self) -> np.ndarray:
        return self._prev_frame

    @property
    def prev_timestamp(self) -> float:
        return self._prev_timestamp
