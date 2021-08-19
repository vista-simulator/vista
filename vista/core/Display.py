from threading import Event
from typing import Optional, Dict, Tuple, List, Any
from collections import deque
from vista.entities.sensors.EventCamera import EventCamera
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from shapely.geometry import LineString
from descartes import PolygonPatch
import cv2

from . import World
from ..entities.agents.Dynamics import StateDynamics, update_with_perfect_controller, \
                                       curvature2tireangle
from ..entities.agents import Car
from ..entities.sensors import Camera, EventCamera
from ..utils import logging, transform, misc


class Display:
    DEFAULT_DISPLAY_CONFIG = {
        'road_buffer_size': 200,
        'birdseye_map_size': (30, 20),
        'gs_bev_w': 2,
        'gs_agent_w': 4,
        'gs_h': 6,
    }

    def __init__(self,
                 world: World,
                 fps: Optional[float] = 30,
                 display_config: Optional[Dict] = dict()):
        # Get arguments
        self._world: World = world
        self._fps: float = fps
        self._config: Dict = misc.merge_dict(display_config,
                                             self.DEFAULT_DISPLAY_CONFIG)

        # Initialize data for plotting
        self._road: deque[np.ndarray] = deque(
            maxlen=self._config['road_buffer_size'])
        self._road_frame_idcs: deque[int] = deque(
            maxlen=self._config['road_buffer_size'])
        self._road_dynamics: StateDynamics = StateDynamics()

        # Get agents with sensors (to be visualized)
        self._agents_with_sensors: List[Car] = []
        for agent in self._world.agents:
            if len(agent.sensors) > 0:
                is_camera = [isinstance(_v, Camera) for _v in agent.sensors]
                if len(is_camera) > 0:
                    self._agents_with_sensors.append(agent)
                if len(is_camera) < len(agent.sensors):
                    logging.warning(
                        'Cannot visualize sensor other than Camera')
        n_agents_with_sensors = len(self._agents_with_sensors)
        max_n_sensors = max(
            [len(_v.sensors) for _v in self._agents_with_sensors])

        # Specify colors for agents and road
        colors = list(cm.get_cmap('Set1').colors)
        rgba2rgb = lambda rgba: np.clip(
            (1 - rgba[:3]) * rgba[3] + rgba[:3], 0., 1.)
        colors = [np.array(list(c) + [0.6]) for c in colors]
        colors = list(map(rgba2rgb, colors))
        self._agent_colors: List[Tuple] = colors
        self._road_color: Tuple = list(cm.get_cmap('Dark2').colors)[-1]

        # Initialize figure
        self._artists: Dict[Any] = dict()
        self._axes: Dict[plt.Axes] = dict()
        figsize = (6.4 * n_agents_with_sensors + 3.2, 3.2 * max_n_sensors)
        self._fig: plt.Figure = plt.figure(figsize=figsize)
        self._fig.patch.set_facecolor('black')  # use black background
        self._gs = self._fig.add_gridspec(
            self._config['gs_h'],
            self._config['gs_agent_w'] * n_agents_with_sensors +
            self._config['gs_bev_w'])
        assert self._config['gs_h'] % max_n_sensors == 0, \
            'Height of grid ({}) can not be exactly divided by max number of sensors ({})'.format( \
                self._config['gs_h'], max_n_sensors)
        gs_agent_h = self._config['gs_h'] // max_n_sensors

        # Initialize birds eye view
        self._axes['bev'] = self._fig.add_subplot(
            self._gs[:, -self._config['gs_bev_w']:])
        self._axes['bev'].set_facecolor('black')
        self._axes['bev'].set_xticks([])
        self._axes['bev'].set_yticks([])
        self._axes['bev'].set_title('Top-down View',
                                    color='white',
                                    size=20,
                                    weight='bold')
        self._axes['bev'].set_xlim(-self._config['birdseye_map_size'][1] / 2.,
                                   self._config['birdseye_map_size'][1] / 2.)
        self._axes['bev'].set_ylim(-self._config['birdseye_map_size'][0] / 2.,
                                   self._config['birdseye_map_size'][0] / 2.)

        # Initialize plot for sensory measurement
        logging.debug(
            'Does not handle preprocessed (cropped/resized) observation')
        for i, agent in enumerate(self._agents_with_sensors):
            for j, sensor in enumerate(agent.sensors):
                if not (isinstance(sensor, Camera) or isinstance(sensor, EventCamera)):
                    logging.error('Unrecognized sensor type {}'.format(type(sensor)))
                    continue
                gs_ij = self._gs[gs_agent_h * j:gs_agent_h * (j + 1),
                                 self._config['gs_agent_w'] *
                                 i:self._config['gs_agent_w'] * (i + 1)]
                ax_name = 'a{}s{}'.format(i, j)
                self._axes[ax_name] = self._fig.add_subplot(gs_ij)
                self._axes[ax_name].set_xticks([])
                self._axes[ax_name].set_yticks([])
                self._axes[ax_name].set_title('Init',
                                              color='white',
                                              size=20,
                                              weight='bold')

                img_shape = (sensor.camera_param.get_height(),
                             sensor.camera_param.get_width(), 3)
                placeholder = fit_img_to_ax(
                    self._fig, self._axes[ax_name],
                    np.zeros(img_shape, dtype=np.uint8))
                self._artists['im:{}'.format(
                    ax_name)] = self._axes[ax_name].imshow(placeholder)
        self._fig.tight_layout()

    def reset(self) -> None:
        # Reset road deque
        self._road.clear()
        self._road.append(self.ref_agent.human_dynamics.numpy()[:3])
        self._road_dynamics = self.ref_agent.human_dynamics.copy()
        self._road_frame_idcs.clear()
        self._road_frame_idcs.append(self.ref_agent.frame_index)

    def render(self):
        # Update road (in global coordinate)
        exceed_end = False
        while self._road_frame_idcs[-1] < (
                self.ref_agent.frame_index +
                self._config['road_buffer_size'] / 2.) and not exceed_end:
            exceed_end, ts = self._get_timestamp(self._road_frame_idcs[-1])
            self._road_frame_idcs.append(self._road_frame_idcs[-1] + 1)
            exceed_end, next_ts = self._get_timestamp(
                self._road_frame_idcs[-1])

            state = [
                curvature2tireangle(self.ref_agent.trace.f_curvature(ts),
                                    self.ref_agent.wheel_base),
                self.ref_agent.trace.f_speed(ts)
            ]
            update_with_perfect_controller(state, next_ts - ts,
                                           self._road_dynamics)
            self._road.append(self._road_dynamics.numpy()[:3])

        # Update road in birds eye view (in reference agent's coordinate)
        ref_pose = self.ref_agent.human_dynamics.numpy()[:3]
        logging.debug(
            'Computation of road in reference frame not vectorized')
        road_in_ref = np.array([
            transform.compute_relative_latlongyaw(_v, ref_pose)
            for _v in self._road
        ])
        road_half_width = self.ref_agent.trace.road_width / 2.
        patch = LineString(road_in_ref).buffer(road_half_width)
        patch = PolygonPatch(patch,
                             fc=self._road_color,
                             ec=self._road_color,
                             zorder=1)
        self._update_patch(self._axes['bev'], 'patch:road', patch)

        # Update agent in birds eye view
        for i, agent in enumerate(self._world.agents):
            poly = misc.agent2poly(agent, self.ref_agent.human_dynamics)
            color = self._agent_colors[i]
            patch = PolygonPatch(poly, fc=color, ec=color, zorder=2)
            self._update_patch(self._axes['bev'], 'patch:agent_{}'.format(i),
                               patch)

        # Update sensory measurements
        for i, agent in enumerate(self._agents_with_sensors):
            cameras = {
                _v.name: _v for _v in agent.sensors if isinstance(_v, Camera)
            }
            event_cameras = {
                _v.name: _v for _v in agent.sensors if isinstance(_v, EventCamera)
            }
            for j, (obs_name, obs) in enumerate(agent.observations.items()):
                ax_name = 'a{}s{}'.format(i, j)
                if obs_name in cameras.keys():
                    obs = plot_roi(obs.copy(), cameras[obs_name].camera_param.get_roi())
                    obs_render = fit_img_to_ax(self._fig, self._axes[ax_name],
                                               obs[:, :, ::-1])
                    # TODO: draw noodle for curvature visualization
                elif obs_name in event_cameras.keys():
                    event_cam_param = event_cameras[obs_name].camera_param
                    frame_obs = events2frame(obs, event_cam_param.get_height(), 
                                             event_cam_param.get_width())
                    # frame_obs = plot_roi(frame_obs.copy(), event_cam_param.get_roi())
                    rgb = cv2.resize(event_cameras[obs_name].prev_frame[:,:,::-1], frame_obs.shape[:2][::-1]) # DEBUG
                    frame_obs = np.concatenate([rgb, frame_obs], axis=1) # DEBUG
                    obs_render = fit_img_to_ax(self._fig, self._axes[ax_name],
                                               frame_obs[:, :, ::-1])
                    # TODO: obs_render shape changes at the first frame
                    logging.debug('obs_render shape changes at the first frame')
                else:
                    logging.error('Unrecognized observation {}'.format(obs_name))
                    continue
                self._axes[ax_name].set_title('{}: {}'.format(
                    agent.id, obs_name),
                                              color='white',
                                              size=20,
                                              weight='bold')
                self._artists['im:{}'.format(ax_name)].set_data(obs_render)

        # Convert to image
        img = fig2img(self._fig)

        return img

    def _update_patch(self, ax: plt.Axes, name: str,
                      patch: PolygonPatch) -> None:
        if name in self._artists:
            self._artists[name].remove()
        ax.add_patch(patch)
        self._artists[name] = patch

    def _get_timestamp(self, frame_index: int) -> Tuple[float, bool]:
        return self.ref_agent.trace.get_master_timestamp(
            self.ref_agent.segment_index, frame_index, check_end=True)

    @property
    def ref_agent(self) -> Car:
        return self._world.agents[0]


def plot_roi(img, roi, color=(0, 0, 255), thickness=2):
    (i1, j1, i2, j2) = roi
    img = cv2.rectangle(img, (j1, i1), (j2, i2), color, thickness)
    return img


def events2frame(events: List[np.ndarray], cam_h: int, cam_w: int,
                 positive_color: Optional[List] = [255, 255, 255],
                 negative_color: Optional[List] = [212, 188, 114],
                 mode: Optional[int] = 2) -> np.ndarray:
    if mode == 0:
        frame = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
        for color, p_events in zip([positive_color, negative_color], events):
            uv = np.concatenate(p_events)[:,:2]
            frame[uv[:,0], uv[:,1], :] = color
    elif mode == 1:
        frame_acc = np.zeros((cam_h, cam_w), dtype=np.int8)
        for polarity, p_events in zip([1, -1], events):
            for sub_p_events in p_events:
                uv = sub_p_events[:,:2]
                frame_acc[uv[:,0], uv[:,1]] += polarity
        
        frame = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
        frame[frame_acc > 0, :] = positive_color
        frame[frame_acc < 0, :] = negative_color
    elif mode == 2:
        frame_abs_acc = np.zeros((cam_h, cam_w), dtype=np.int8)
        frame = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
        for polarity, p_events in zip([1, -1], events):
            for sub_p_events in p_events:
                uv = sub_p_events[:,:2]
                add_c = np.array(positive_color if polarity > 0 else negative_color)[None,...]
                cnt = frame_abs_acc[uv[:,0], uv[:,1]][:,None]
                frame[uv[:,0], uv[:,1]] = (frame[uv[:,0], uv[:,1]] * cnt + add_c) / (cnt + 1)
                frame_abs_acc[uv[:,0], uv[:,1]] = cnt[:,0] + 1
    else:
        raise NotImplementedError('Unknown mode {}'.format(mode))
    return frame


def fig2img(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    return img


def fit_img_to_ax(fig: plt.Figure, ax: plt.Axes,
                  img: np.ndarray) -> np.ndarray:
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    w, h = bbox.width, bbox.height
    img_h, img_w = img.shape[:2]
    new_img_w = img_h * w / h
    new_img_h = img_w * h / w
    d_img_w = new_img_w - img_w
    d_img_h = new_img_h - img_h
    if d_img_h > 0:
        pad_img = np.zeros((int(d_img_h // 2), img_w, 3), dtype=np.uint8)
        new_img = np.concatenate([pad_img, img, pad_img], axis=0)
    elif d_img_w > 0:
        pad_img = np.zeros((img_h, int(d_img_w // 2), 3), dtype=np.uint8)
        new_img = np.concatenate([pad_img, img, pad_img], axis=1)
    else:
        raise ValueError('Something weird happened.')
    return new_img
