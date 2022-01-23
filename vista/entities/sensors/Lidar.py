import os
from typing import Dict, Any
import numpy as np
import h5py

from .BaseSensor import BaseSensor
from .lidar_utils import LidarSynthesis, Pointcloud
from ..Entity import Entity
from ...core import Trace
from ...utils import logging, parse_params


class Lidar(BaseSensor):
    """ A LiDAR sensor object that synthesizes LiDAR measurement locally around the
    dataset given a viewpoint (potentially different from the dataset) and timestamp.

    Args:
        attach_to (Entity): A car to be attached to.
        config (dict): Configuration of LiDAR sensor. An example (default) is,

            >>> DEFAULT_CONFIG = {
                'name': 'lidar_3d',
                'yaw_fov': None,
                'pitch_fov': None,
                'culling_r': 1,
                'use_synthesizer': True,
            }

            Check :class:`Lidarsynthesis` object for more details about the configuration.

    """
    DEFAULT_CONFIG = {
        'name': 'lidar_3d',
        'yaw_fov': None,
        'pitch_fov': None,
        'culling_r': 1,
        'use_synthesizer': True,
    }

    def __init__(self, attach_to: Entity, config: Dict) -> None:
        super(Lidar, self).__init__(attach_to, config)

        logging.debug('Not actually streaming lidar data when reading')
        self._streams: Dict[str, h5py.File] = dict()

        # Initialize lidar novel view synthesis object
        if self.config['use_synthesizer']:
            # Make one view synthesizer for each trace within
            # attach_to.parent.traces (different traces may have different
            # input sensors specs). For each synthesizer, pass in the input
            # and output lidar params. Then during synthesis, use the
            # appropraite synthesizer.
            self._view_synthesizers = {}
            for trace in attach_to.parent.traces:
                pfile = parse_params.ParamsFile(trace.param_file)
                in_params, _ = pfile.parse_lidar(self.name)
                self._view_synthesizers[trace] = LidarSynthesis(
                    load_model=True,
                    input_yaw_fov=in_params['yaw_fov'],
                    input_pitch_fov=in_params['pitch_fov'],
                    **self.config)

        else:
            self._view_synthesizers = None

    def reset(self) -> None:
        """ Reset LiDAR sensor by initiating LiDAR data stream based on
        current reference pointer to the dataset.

        """
        logging.info(f'Lidar ({self.id}) reset')

        # Initiate lidar data stream based on current reference pointer to the
        # dataset. All data streams are handled by the main lidar and shared
        # across all Lidar objects in an agent.
        multi_sensor = self.parent.trace.multi_sensor
        if self.name == multi_sensor.main_lidar:
            for lidar_name in multi_sensor.lidar_names:
                fpath = os.path.join(self.parent.trace.trace_path,
                                     lidar_name + '.h5')
                stream = h5py.File(fpath, 'r')
                self._streams[lidar_name] = stream
        else:
            main_name = multi_sensor.main_lidar
            main_sensor = [
                _s for _s in self.parent.sensors if _s.name == main_name
            ]
            assert len(main_sensor) == 1, \
                    f'Cannot find main sensor {main_name}'

            main_sensor = main_sensor[0]
            assert isinstance(main_sensor, Lidar), \
                    'Main sensor is not Lidar object'
            self._streams = main_sensor.streams

        # reset lidar synthesis
        pass

    def capture(self, timestamp: float, **kwargs) -> np.ndarray:
        """ Synthesize LiDAR point cloud based on current timestamp and transformation
        between the novel viewpoint to be simulated and the nominal viewpoint from the
        pre-collected dataset.

        Args:
            timestamp (float): Timestamp that allows to retrieve a pointer to
                the dataset for data-driven simulation (synthesizing point cloud
                from real LiDAR sweep).

        Returns:
            np.ndarray: Synthesized point cloud.

        """
        logging.info(f'Lidar ({self.id}) capture')

        # Get frame at the closest smaller timestamp from dataset.
        multi_sensor = self.parent.trace.multi_sensor
        all_frame_nums = multi_sensor.get_frames_from_times([timestamp])
        for lidar_name in multi_sensor.lidar_names:
            stream = self.streams[lidar_name]
            frame_num = all_frame_nums[lidar_name][0]
            xyz = stream['xyz'][frame_num]
            intensity = stream['intensity'][frame_num]
            pcd = Pointcloud(xyz, intensity)
            pcd = pcd[pcd.dist > 2.5]
            # TODO: when is it possible for there to be multiple (multi_sensor.lidar_names)?

        # Synthesis by rendering
        if self.config['use_synthesizer']:
            # self.parent.reslative_state.update(0, 0, yaw=np.sin(timestamp))
            lat, long, yaw = self.parent.relative_state.numpy()
            logging.debug(
                f"state: {lat} {long} {yaw} \t timestamp {timestamp}")
            trans = np.array([-long, lat, 0])
            rot = np.array([0., 0, yaw])  # TODO: should yaw be Y or Z?
            synthesizer = self._view_synthesizers[self.parent.trace]
            new_pcd, new_dense = synthesizer.synthesize(trans, rot, pcd)
        else:
            new_pcd = pcd

        return new_pcd

    def update_scene_object(self, name: str, scene_object: Any,
                            pose: Any) -> None:
        """ Adding virtual object in LiDAR synthesis is not yet implemented. """
        raise NotImplementedError

    @property
    def config(self) -> Dict:
        """ Configuration of the LiDAR sensor. """
        return self._config

    @property
    def streams(self) -> Dict[str, h5py.File]:
        """ Data stream of LiDAR dataset to be simulated from. """
        return self._streams

    @property
    def view_synthesis(self) -> LidarSynthesis:
        """ Wiew synthesizer object for the first trace. """
        first = list(self._view_synthesizers.keys())[0]
        return self._view_synthesizers[first]
