import copy
import os
import glob
from typing import Dict, List, Any
import numpy as np
import h5py
from ffio import FFReader
import pyrender

from .camera_utils import CameraParams, ViewSynthesis, ZNEAR, ZFAR
from .BaseSensor import BaseSensor
from ..Entity import Entity
from ...utils import logging, misc, transform


class Camera(BaseSensor):
    """ A RGB camera sensor object that synthesizes RGB image locally around the
    dataset given a viewpoint (potentially different from the dataset) and timestamp.

    Args:
        attach_to (Entity): A parent object (Car) to be attached to.
        config (Dict): Configuration of the sensor. An example (default) is,

            >>> DEFAULT_CONFIG = {
                'depth_mode': 'FIXED_PLANE',
                'znear': ZNEAR,
                'zfar': ZFAR,
                'use_lighting': False,
                'directional_light_intensity': 10,
                'recoloring_factor': 0.5,
                'use_synthesizer': True,
            }

            Check :class:`Viewsynthesis` object for more details about the configuration.

    """
    DEFAULT_CONFIG = {
        'name': 'camera_front',
        'depth_mode': 'FIXED_PLANE',
        'znear': ZNEAR,
        'zfar': ZFAR,
        'use_lighting': False,
        'directional_light_intensity': 10,
        'recoloring_factor': 0.5,
        'use_synthesizer': True,
    }

    def __init__(self, attach_to: Entity, config: Dict) -> None:
        super(Camera, self).__init__(attach_to, config)

        # Define the CameraParams object for each physical camera in each trace
        self._input_cams = {
            trace: CameraParams(trace.param_file, self.name)
            for trace in attach_to.parent.traces
        }
        for _cp in self._input_cams.values():
            _cp.resize(*config['size'])

        # Now define the virtual camera in VISTA (defaults copied from the
        # first trace)
        first_trace = attach_to.parent.traces[0]
        self._virtual_cam = copy.deepcopy(self._input_cams[first_trace])
        self._virtual_cam.resize(*config['size'])

        self._streams: Dict[str, FFReader] = dict()
        self._flow_streams: Dict[str, List[FFReader]] = dict()
        self._flow_meta: Dict[str, h5py.File] = dict()
        if self._config.get('use_synthesizer', True):
            self._view_synthesis: ViewSynthesis = ViewSynthesis(
                self._virtual_cam, self._config)
        else:
            self._view_synthesis = None

    def reset(self) -> None:
        """ Reset RGB camera sensor by initiating RGB data stream based on
        current reference pointer to the dataset.

        """
        logging.info(f'Camera ({self.id}) reset')

        # Close stream already open
        for stream in self.streams.values():
            stream.close()

        for flow_stream in [
                _vv for _v in self.flow_streams.values()
                for _vv in _v.values()
        ]:
            flow_stream.close()

        # Fetch video stream from the associated trace. All video streams are handled by the main
        # camera and shared across all cameras in an agent.
        multi_sensor = self.parent.trace.multi_sensor
        trace_camera = self._input_cams[self.parent.trace]
        if self.name == multi_sensor.main_camera:
            for camera_name in multi_sensor.camera_names:
                # get video stream
                video_path = os.path.join(self.parent.trace.trace_path,
                                          camera_name + '.avi')
                cam_h, cam_w = (trace_camera.get_height(),
                                trace_camera.get_width())
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
                        flow_seek_sec = flow_stream.frame_to_secs(
                            flow_frame_num)
                        flow_stream.seek(flow_seek_sec)
                        self._flow_streams[camera_name][
                            flow_name] = flow_stream
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
                        camera_param = parent_sensor_dict[
                            camera_name]._input_cams[self.parent.trace]
                    else:
                        camera_param = self._virtual_cam
                    self.view_synthesis.add_bg_mesh(camera_param)

    def capture(self, timestamp: float, **kwargs) -> np.ndarray:
        """ Synthesize RGB image based on current timestamp and transformation
        between the novel viewpoint to be simulated and the nominal viewpoint from
        the pre-collected dataset. Note that if there exists optical flow data in
        the trace directory, the :class:`Camera` object will take the optical flow
        to interpolate across frame to the exact timestamp as opposed to retrieving
        the RGB frame with the closest timestamp in the dataset.

        Args:
            timestamp (float): Timestamp that allows to retrieve a pointer to
                the dataset for data-driven simulation (synthesizing RGB image
                from real RGB video).

        Returns:
            np.ndarray: A synthesized RGB image.

        """
        logging.info(f'Camera ({self.id}) capture')

        # Get frame at the closest smaller timestamp from dataset
        multi_sensor = self.parent.trace.multi_sensor
        if self.name == multi_sensor.main_camera:
            fetch_smaller = self.flow_streams != dict(
            )  # only when using optical flow
            all_frame_nums = multi_sensor.get_frames_from_times([timestamp],
                                                                fetch_smaller)
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
                for flow_name, flow_minmax in self.flow_meta[
                        camera_name].items():
                    flow_stream = self.flow_streams[camera_name][flow_name]
                    flow[flow_name] = misc.img2flow(
                        flow_stream.image.copy(),
                        flow_minmax[int(flow_stream.frame_num)],
                        frame.shape[:2])

                frame_num = int(self.streams[camera_name].frame_num)
                curr_ref_ts = multi_sensor.get_time_from_frame_num(
                    camera_name, frame_num)
                next_ref_ts = multi_sensor.get_time_from_frame_num(
                    camera_name, frame_num + 1)

                logging.warning(
                    'Stream frame number exceed 1 non-intentionally')
                self.streams[camera_name].read(
                )  # NOTE: stream frame number exceed 1 here
                next_frame = self.streams[camera_name].image.copy()
                frames[camera_name] = misc.biinterp(frame, next_frame,
                                                    flow['forward'],
                                                    flow['backward'],
                                                    timestamp, curr_ref_ts,
                                                    next_ref_ts)

        # Synthesis by rendering
        if self.view_synthesis is not None:
            latlongyaw = self.parent.relative_state.numpy()
            trans, rot = transform.latlongyaw2vec(latlongyaw)
            rendered_frame, _ = self.view_synthesis.synthesize(
                trans, rot, frames)
        else:
            rendered_frame = frames[self.name]

        return rendered_frame

    def update_scene_object(self, name: str, scene_object: pyrender.Mesh,
                            pose: Any) -> None:
        """ Update pyrender mesh object in the scene for rendering.

        Args:
            name (str): Name of the scene object.
            scene_object (pyrender.Mesh): The scene object.
            pose (Any): The pose of the scene object.

        """
        trans, rotvec = transform.latlongyaw2vec(pose)
        quat = transform.euler2quat(rotvec)
        self.view_synthesis.update_object_node(name, scene_object, trans, quat)

    @property
    def config(self) -> Dict:
        """ Configuration of the RGB camera sensor. """
        return self._config

    @property
    def camera_param(self) -> CameraParams:
        """ Camera parameters of the virtual camera. """
        return self._virtual_cam

    @property
    def streams(self) -> Dict[str, FFReader]:
        """ Data stream of RGB image/video dataset to be simulated from. """
        return self._streams

    @property
    def flow_streams(self) -> Dict[str, List[FFReader]]:
        """ Data stream of optical flow (if any). """
        return self._flow_streams

    @property
    def flow_meta(self) -> Dict[str, h5py.File]:
        """ Meta data of optical flow (if any). """
        return self._flow_meta

    @property
    def view_synthesis(self) -> ViewSynthesis:
        """ View synthesizer object. """
        return self._view_synthesis

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} (id={self.id})> ' + \
               f'name: {self.name} ' + \
               f'size: {self.virtual_cam.get_height()}x{self.virtual_cam.get_width()} ' + \
               f'#streams: {len(self.streams)} '
