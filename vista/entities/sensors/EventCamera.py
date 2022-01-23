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

try:
    from metavision_core.event_io.raw_reader import RawReader
except ImportError:
    logging.warning(
        'Fail to import module for event camera. Remember to do ' +
        'source <some-dir>/openeb/build/utils/scripts/setup_env.sh' +
        'Can ignore this if not using it')


class EventCamera(BaseSensor):
    """ A event camera sensor object that synthesizes event data locally around the RGB
    dataset given a viewpoint (potentially different from the dataset) and timestamp with
    video interpolation and an event emission model.

    Args:
        attach_to (Entity): A parent object (car) to be attached to.
        config (Dict): Configuration of the sensor. An example (default) config is,

            >>> DEFAULT_CONFIG = {
                'rig_path': None,
                # Event camera
                'name': 'event_camera_front',
                'original_size': (480, 640),
                'size': (240, 320),
                'optical_flow_root': '../data_prep/Super-SloMo',
                'checkpoint': '../data_prep/Super-SloMo/ckpt/SuperSloMo.ckpt',
                'lambda_flow': 0.5,
                'max_sf': 16,
                'use_gpu': True,
                'positive_threshold': 0.1,
                'sigma_positive_threshold': 0.02,
                'negative_threshold': -0.1,
                'sigma_negative_threshold': 0.02,
                'reproject_pixel': False,
                'subsampling_ratio': 0.5,
                # RGB rendering
                'base_camera_name': 'camera_front',
                'base_size': (600, 960),
                'depth_mode': 'FIXED_PLANE',
                'use_lighting': False,
            }

    Note that event camera simulation requires third-party dependence and pretrained
    checkpoint for video interpolation.

    """
    DEFAULT_CONFIG = {
        'rig_path': None,
        # Event camera
        'name': 'event_camera_front',
        'original_size': (480, 640),
        'size': (240, 320),
        'optical_flow_root': '../data_prep/Super-SloMo',
        'checkpoint': '../data_prep/Super-SloMo/ckpt/SuperSloMo.ckpt',
        'lambda_flow': 0.5,
        'max_sf': 16,
        'use_gpu': True,
        'positive_threshold': 0.1,
        'sigma_positive_threshold': 0.02,
        'negative_threshold': -0.1,
        'sigma_negative_threshold': 0.02,
        'reproject_pixel': False,
        'subsampling_ratio': 0.5,
        # RGB rendering
        'base_camera_name': 'camera_front',
        'base_size': (600, 960),
        'depth_mode': 'FIXED_PLANE',
        'use_lighting': False,
    }

    def __init__(self, attach_to: Entity, config: Dict) -> None:
        super(EventCamera, self).__init__(attach_to, config)

        self._config['rig_path'] = os.path.expanduser(self._config['rig_path'])
        self._camera_param: CameraParams = CameraParams(
            self._config['rig_path'], self.name)
        self._camera_param.resize(*self._config['size'])
        self._base_camera_param: CameraParams = CameraParams(
            self._config['rig_path'], self._config['base_camera_name'])
        self._base_camera_param.resize(*self._config['base_size'])
        self._config['use_synthesizer'] = self._config.get(
            'use_synthesizer', True)
        if self._config['use_synthesizer']:
            self._streams: Dict[str, FFReader] = dict()
            vs_cam_param = self._base_camera_param if \
                self.config['reproject_pixel'] else self._camera_param
            self._view_synthesis: ViewSynthesis = ViewSynthesis(
                vs_cam_param, self._config, init_with_bg_mesh=False)

            # load video interpolation model
            sys.path.append(os.path.abspath(self._config['optical_flow_root']))
            from slowmo_warp import SlowMoWarp
            self._config['checkpoint'] = os.path.abspath(
                self._config['checkpoint'])
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
        else:
            self._streams: Dict[str, RawReader] = dict()
            self._prev_timestamp: float = None

    def reset(self) -> None:
        """ Reset Event camera sensor by initiating RGB data stream (as it's simulated
        using RGB data) based on current reference pointer to the dataset.

        """
        logging.info(f'Event camera ({self.id}) reset')

        if self._config['use_synthesizer']:
            # Close stream already open
            for stream in self.streams.values():
                stream.close()

            # Fetch video stream from the associated trace. This is mostly the same as how we handle
            # video stream in Camera object. The difference and the reason of not sharing with
            # Camera is (1) EventCamera can be a standalone object to be initiated and more
            # importantly (2) TODO
            multi_sensor = self.parent.trace.multi_sensor
            if self.name == multi_sensor.main_event_camera:
                for camera_name in multi_sensor.camera_names:  # base cameras are regular cameras
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
                    seek_sec = self._streams[camera_name].frame_to_secs(
                        frame_num)
                    self._streams[camera_name].seek(seek_sec)
            else:  # use shared streams from the main camera
                main_name = multi_sensor.main_event_camera
                main_sensor = [
                    _s for _s in self.parent.sensors if _s.name == main_name
                ]
                assert len(
                    main_sensor) == 1, 'Cannot find main sensor {}'.format(
                        main_name)
                main_sensor = main_sensor[0]
                assert isinstance(main_sensor,
                                  Camera), 'Main sensor is not Camera object'
                self._streams = main_sensor.streams

            # Add background mesh for all video stream in view synthesis. Note that we use camera
            # parameters from the base camera since 2D-to-3D part is determined by the regular RGB
            # video streams. Reprojecting back to 2D is then be handled by event camera
            # configuration
            parent_sensor_dict = {_s.name: _s for _s in self.parent.sensors}
            for camera_name in self.streams.keys():
                if camera_name not in self.view_synthesis.bg_mesh_names:
                    if camera_name in parent_sensor_dict.keys():
                        camera_param = parent_sensor_dict[
                            camera_name].camera_param
                    else:
                        camera_param = CameraParams(self._config['rig_path'],
                                                    camera_name)
                        camera_param.resize(*self._config['base_size'])
                    self.view_synthesis.add_bg_mesh(camera_param)

            # Reset previous frame
            self._prev_frame = None

            if self.config['reproject_pixel']:
                cam_h = self.camera_param.get_height()
                cam_w = self.camera_param.get_width()
                world_rays = self.view_synthesis._world_rays[
                    self.base_camera_param.name]
                K = self.camera_param.get_K()
                self._uv = np.matmul(K, world_rays).T[:, :2][:, ::-1]
        else:
            # TODO: check if we need to close event data stream
            logging.debug(
                'Not sure if it is ok to not close the event data stream')

            raw_path = os.path.join(self.parent.trace.trace_path,
                                    f'{self.name}.raw')
            stream = RawReader(raw_path)
            self._streams[self.name] = stream

            # Reset previous timestamp
            self._prev_timestamp = None

    def capture(self,
                timestamp: float,
                update_rgb_frame_only: Optional[bool] = False) -> np.ndarray:
        """ Synthesize event data based on current timestamp and transformation
        between the novel viewpoint to be simulated and the nominal viewpoint from
        the pre-collected RGB dataset. In a very high level, it basically performs
        video interpolation across consecutive RGB frames, extract events with an
        event emission model, projects the events to the event camera space. Note
        that the simulation is running a deep network (SuperSloMo) for video interpolation.

        Args:
            timestamp (float): Timestamp that allows to retrieve a pointer to
                the dataset for data-driven simulation (synthesizing RGB image
                from real RGB video).

        Returns:
            np.ndarray: A synthesized event data.

        """
        logging.info(f'Event camera ({self.id}) capture')

        if self._config['use_synthesizer']:
            # Get frame at the closest smaller timestamp from dataset. In event camera simulation,
            # we don't use flow interpolation since event generation is already continuous-time.
            multi_sensor = self.parent.trace.multi_sensor
            if self.name == multi_sensor.main_event_camera:
                all_frame_nums = multi_sensor.get_frames_from_times(
                    [timestamp], fetch_smaller=False)
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

            # Synthesis by rendering with compensation of relative pose between event camera
            # parameters and the base (RGB) camera parameters since we set the view synthesizer
            # with event camera parameters and project frame to 3D meshes using base camera
            # parameters
            lat, long, yaw = self.parent.relative_state.numpy()
            trans = np.array([lat, 0., -long])
            rot = np.array([0., yaw, 0.])
            rendered_frame, _ = self.view_synthesis.synthesize(
                trans, rot, frames)

            # work with RGB instead of BGR; need copy otherwise fail at converting to tensor
            rendered_frame = rendered_frame[:, :, ::-1].copy()

            if update_rgb_frame_only:
                self._prev_frame = rendered_frame
                self._prev_timestamp = timestamp
            else:
                events = [[], []]
                if self.prev_frame is not None:
                    with torch.no_grad():
                        out = self._interp.forward_warp(
                            self.prev_frame,
                            rendered_frame,
                            max_sf=self.config['max_sf'],
                            use_max_flow=True)

                    flow = list(
                        map(lambda _x: _x.cpu().numpy()[0].transpose(1, 2, 0),
                            out['flow']))

                    interp = [self.prev_frame] + out['interpolated']
                    if len(interp) >= 2:
                        last_interp_frame = cv2.cvtColor(
                            interp[0], cv2.COLOR_RGB2GRAY)
                        dt = (timestamp - self.prev_timestamp) / out['sf']
                        event_timestamp = self.prev_timestamp
                        for interp_frame in interp[1:]:
                            interp_frame = cv2.cvtColor(
                                interp_frame, cv2.COLOR_RGB2GRAY)
                            log_d_interp = np.log(interp_frame / 255. + 1e-8) - \
                                        np.log(last_interp_frame / 255. + 1e-8)
                            sample_fn = lambda mu, sig: np.clip(
                                np.random.normal(mu, sig), 0., 1.)
                            positive_C = sample_fn(
                                self.config['positive_threshold'],
                                self.config['sigma_positive_threshold'])
                            negative_C = -sample_fn(
                                -self.config['negative_threshold'],
                                self.config['sigma_negative_threshold'])
                            if self.config['reproject_pixel']:
                                cam_h = self.camera_param.get_height()
                                cam_w = self.camera_param.get_width()

                                def extract_uv(_mask):
                                    _fl_mask = _mask.flatten()
                                    _uv = self._uv[_fl_mask]

                                    _mode = 1
                                    if _mode == 0:
                                        _uv = np.round(_uv).astype(np.int)
                                    elif _mode == 1:
                                        _uv_floor = np.floor(_uv)
                                        _uv_ceil = np.ceil(_uv)
                                        diff_floor = np.abs(_uv - _uv_floor)
                                        diff_ceil = np.abs(_uv - _uv_ceil)

                                        n_uv = _uv.shape[0]
                                        zeros = np.zeros((n_uv, ))
                                        _uv_score = np.concatenate([
                                            np.stack([
                                                _uv_floor[:, 0],
                                                _uv_floor[:, 1], zeros
                                            ],
                                                     axis=1),
                                            np.stack([
                                                _uv_floor[:, 0],
                                                _uv_ceil[:, 1], zeros
                                            ],
                                                     axis=1),
                                            np.stack([
                                                _uv_ceil[:, 0],
                                                _uv_floor[:, 1], zeros
                                            ],
                                                     axis=1),
                                            np.stack([
                                                _uv_ceil[:, 0], _uv_ceil[:, 1],
                                                zeros
                                            ],
                                                     axis=1)
                                        ],
                                                                   axis=0)

                                        _uv_score[:n_uv,
                                                  2] = diff_ceil[:,
                                                                 0] * diff_ceil[:,
                                                                                1]
                                        _uv_score[
                                            n_uv:2 * n_uv,
                                            2] = diff_ceil[:,
                                                           0] * diff_floor[:,
                                                                           1]
                                        _uv_score[
                                            2 * n_uv:3 * n_uv,
                                            2] = diff_floor[:,
                                                            0] * diff_ceil[:,
                                                                           1]
                                        _uv_score[
                                            3 * n_uv:,
                                            2] = diff_floor[:,
                                                            0] * diff_floor[:,
                                                                            1]

                                        _uv = _uv_score[
                                            _uv_score[:, 2] >= 0.2, :2].astype(
                                                np.int)
                                    else:
                                        _uv_floor = np.floor(_uv)
                                        _uv_ceil = np.ceil(_uv)
                                        _uv = np.concatenate([
                                            np.stack([
                                                _uv_floor[:, 0], _uv_floor[:,
                                                                           1]
                                            ],
                                                     axis=1),
                                            np.stack([
                                                _uv_floor[:, 0], _uv_ceil[:, 1]
                                            ],
                                                     axis=1),
                                            np.stack([
                                                _uv_ceil[:, 0], _uv_floor[:, 1]
                                            ],
                                                     axis=1),
                                            np.stack([
                                                _uv_ceil[:, 0], _uv_ceil[:, 1]
                                            ],
                                                     axis=1)
                                        ],
                                                             axis=0)
                                        _uv = _uv.astype(np.int)

                                    _valid_mask = (_uv[:,0] > 0) & (_uv[:,0] < cam_h) & \
                                                (_uv[:,1] > 0) & (_uv[:,1] < cam_w)
                                    _uv = _uv[_valid_mask]
                                    return _uv

                                positive_uv = extract_uv(
                                    log_d_interp >= positive_C)
                                negative_uv = extract_uv(
                                    log_d_interp <= negative_C)
                            else:
                                positive_uv = np.argwhere(
                                    log_d_interp >= positive_C)
                                negative_uv = np.argwhere(
                                    log_d_interp <= negative_C)
                            event_timestamp += dt
                            event_timestamp_us = (event_timestamp *
                                                  1e6).astype(np.int64)
                            positive_events = np.concatenate([
                                positive_uv,
                                np.tile([event_timestamp_us, 1],
                                        (positive_uv.shape[0], 1))
                            ],
                                                             axis=1)
                            negative_events = np.concatenate([
                                negative_uv,
                                np.tile([event_timestamp_us, 1],
                                        (negative_uv.shape[0], 0))
                            ],
                                                             axis=1)
                            events[0].append(positive_events)
                            events[1].append(negative_events)
                            last_interp_frame = interp_frame

                self._prev_frame = rendered_frame
                self._prev_timestamp = timestamp

                return events
        else:
            if self._prev_timestamp is not None:
                stream = self.streams[self.name]

                multi_sensor = self.parent.trace.multi_sensor
                ref_sensor = multi_sensor.master_sensor
                # TODO: didn't do dedicated alignment between ros clock and event camera clock
                logging.debug(
                    'No dedicated alignment between ros and event camera clocks'
                )
                offset_timestamp = multi_sensor.get_time_from_frame_num(
                    ref_sensor, 0)

                event_timestamp_us = int(
                    (self._prev_timestamp - offset_timestamp) * 1e6)
                if event_timestamp_us < stream.current_time:
                    stream.reset()  # cannot seek backward
                stream.seek_time(event_timestamp_us)

                dt_us = int((timestamp - self._prev_timestamp) * 1e6)
                events = stream.load_delta_t(dt_us)
                if stream.is_done():
                    self.parent._done = True

                ori_cam_h, ori_cam_w = self.config['original_size']
                cam_h = self.camera_param.get_height()
                cam_w = self.camera_param.get_width()

                # NOTE: do subsampling otherwise too many events left
                subsample_factor = (cam_h * cam_w) / float(
                    ori_cam_h * ori_cam_w) * self.config['subsampling_ratio']
                n_events = events['x'].shape[0]
                subsample_mask = np.random.choice(np.arange(n_events),
                                                  size=int(n_events *
                                                           subsample_factor),
                                                  replace=False)
                events_x = events['x'][subsample_mask]
                events_y = events['y'][subsample_mask]
                events_p = events['p'][subsample_mask]

                mul_x = cam_w / float(ori_cam_w)
                mul_y = cam_h / float(ori_cam_h)
                events_x = (events_x * mul_x).astype(np.int)
                events_y = (events_y * mul_y).astype(np.int)

                positive_mask = events_p == 1
                negative_mask = events_p == 0
                positive_events = np.array(
                    [events_y[positive_mask], events_x[positive_mask]]).T
                negative_events = np.array(
                    [events_y[negative_mask], events_x[negative_mask]]).T

                events = [[positive_events], [negative_events]]
            else:
                events = [[], []]

            self._prev_timestamp = timestamp

            return events

    @property
    def config(self) -> Dict:
        """ Configuration of this sensor. """
        return self._config

    @property
    def streams(self) -> Dict[str, FFReader]:
        """ Data stream of RGB image/video dataset to be simulated from. """
        return self._streams

    @property
    def camera_param(self) -> CameraParams:
        """ Camera parameters of the virtual event camera. """
        return self._camera_param

    @property
    def base_camera_param(self) -> CameraParams:
        """ Camera parameters of the RGB camera. """
        return self._base_camera_param

    @property
    def view_synthesis(self) -> ViewSynthesis:
        """ View synthesizer object. """
        return self._view_synthesis

    @property
    def prev_frame(self) -> np.ndarray:
        """ Previous RGB frame. """
        return self._prev_frame

    @property
    def prev_timestamp(self) -> float:
        """ Previous timestamp. """
        return self._prev_timestamp
