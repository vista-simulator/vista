from threading import Event
from typing import Optional, Dict, Tuple, List, Any
from collections import deque
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors, patches

from shapely.geometry import LineString
from descartes import PolygonPatch
import cv2

from . import World
from ..entities.agents.Dynamics import StateDynamics, update_with_perfect_controller, \
                                       curvature2tireangle
from ..entities.agents import Car
from ..entities.sensors import Camera, EventCamera, Lidar
from ..entities.sensors.camera_utils import CameraParams
from ..entities.sensors.lidar_utils import Pointcloud
from ..utils import logging, transform, misc


class Display:
    """ This is a visualizer of VISTA simulator. It renders an image that contains visualization
    of all sensors from all agents and a top-down view that depicts road and all cars in the scene
    within a predefined range based on the state of the simulator (:class:`World`).

    Args:
        world (vista.core.World): World to be visualized.
        fps (int): Frame per second.
        display_config (Dict): Configuration for the display (visualization).

    Raises:
        AssertionError: Grid spec is inconsistent with maximal number of sensors across agents.

    Example usage::

        >>> display_config = {
            'road_buffer_size': 200,
            'birdseye_map_size': (30, 20), # size of bev map in vertical and horizontal directions
            'gs_bev_w': 2, # grid spec width for the birdseye view block
            'gs_agent_w': 4, # grid spec width for an agent's block
            'gs_h': 6, # grid spec height
            'gui_scale': 1.0, # a global scale that determines the size of the figure
            'vis_full_frame': False, # if Display should not crop/resize camera for visualization purposes
        }
        >>> display = Display(world, )

    """
    DEFAULT_DISPLAY_CONFIG = {
        'road_buffer_size': 200,
        'birdseye_map_size': (30, 20),
        'gs_bev_w': 2,
        'gs_agent_w': 4,
        'gs_h': 6,
        'gui_scale': 1.0,
        'vis_full_frame': False
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
        if n_agents_with_sensors > 0:
            max_n_sensors = max(
                [len(_v.sensors) for _v in self._agents_with_sensors])
        else:
            max_n_sensors = 1

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
        gui_scale = self._config['gui_scale']
        figsize = (6.4 * gui_scale * n_agents_with_sensors + 3.2 * gui_scale,
                   3.2 * gui_scale * max_n_sensors)
        self._fig: plt.Figure = plt.figure(figsize=figsize)
        self._fig.patch.set_facecolor('black')  # use black background
        self._gs = self._fig.add_gridspec(
            self._config['gs_h'],
            self._config['gs_agent_w'] * n_agents_with_sensors +
            self._config['gs_bev_w'])
        assert self._config['gs_h'] % max_n_sensors == 0, \
            (f'Height of grid ({self._config["gs_h"]}) can not be exactly ' + \
            f'divided by max number of sensors ({max_n_sensors})')
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

                if (isinstance(sensor, Camera)
                        or isinstance(sensor, EventCamera)):
                    param = sensor.camera_param
                    img_shape = (param.get_height(), param.get_width(), 3)

                elif isinstance(sensor, Lidar):
                    x_dim, y_dim = sensor.view_synthesis._dims[:, 0]
                    # Cut width in half and stack on-top
                    img_shape = (y_dim * 2, x_dim // 2, 3)

                else:
                    logging.error(f'Unrecognized sensor type {type(sensor)}')
                    continue

                gs_ij = self._gs[gs_agent_h * j:gs_agent_h * (j + 1),
                                 self._config['gs_agent_w'] *
                                 i:self._config['gs_agent_w'] * (i + 1)]
                ax_name = 'a{}s{}'.format(i, j)
                self._axes[ax_name] = self._fig.add_subplot(gs_ij,
                                                            facecolor='black')
                self._axes[ax_name].set_xticks([])
                self._axes[ax_name].set_yticks([])
                self._axes[ax_name].set_title('Init',
                                              color='white',
                                              size=20,
                                              weight='bold')

                placeholder = fit_img_to_ax(
                    self._fig, self._axes[ax_name],
                    np.zeros(img_shape, dtype=np.uint8))
                self._artists['im:{}'.format(
                    ax_name)] = self._axes[ax_name].imshow(placeholder)

        self._fig.tight_layout()

    def reset(self) -> None:
        """ Reset the visualizer. This should be called every time after
        :class:`World` reset. It basically reset the cache of road data
        used in the top-down view visualization.
        """
        # Reset road deque
        self._road.clear()
        self._road.append(self.ref_agent.human_dynamics.numpy()[:3])
        self._road_dynamics = self.ref_agent.human_dynamics.copy()
        self._road_frame_idcs.clear()
        self._road_frame_idcs.append(self.ref_agent.frame_index)

    def render(self):
        """ Render an image that visualizes the simulator. This includes visualization
        of all sensors of every agent and a top-down view that depicts the road and all
        cars in the scene within a certain range. Note that it render visualization based
        on the current status of the world and should be called every time when there is
        any update to the simulator.

        Returns:
            np.ndarray: An image of visualization for the simulator.

        """
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
        logging.debug('Computation of road in reference frame not vectorized')
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
                _v.name: _v
                for _v in agent.sensors if isinstance(_v, Camera)
            }
            event_cameras = {
                _v.name: _v
                for _v in agent.sensors if isinstance(_v, EventCamera)
            }
            lidars = {
                _v.name: _v
                for _v in agent.sensors if isinstance(_v, Lidar)
            }
            for j, (obs_name, obs) in enumerate(agent.observations.items()):
                ax_name = 'a{}s{}'.format(i, j)
                if obs_name in cameras.keys():
                    obs = plot_roi(obs.copy(),
                                   cameras[obs_name].camera_param.get_roi())
                    sensor = cameras[obs_name]
                    noodle = curvature2noodle(self.ref_agent.curvature,
                                              sensor.camera_param,
                                              mode='camera')
                    obs = cv2.polylines(obs, [noodle], False, (255, 0, 0), 2)

                    if not self._config["vis_full_frame"]:
                        # Black out the sides for visualization
                        h, w = obs.shape[:2]
                        h_, w_ = (0.65 * h, 0.65 * w)
                        hs, ws = (int((h - h_) // 2), int((w - w_) // 2))
                        obs = cv2.resize(obs[hs:-hs, ws:-ws], (w, h))

                    obs_render = fit_img_to_ax(self._fig, self._axes[ax_name],
                                               obs[:, :, ::-1])

                elif obs_name in event_cameras.keys():
                    event_cam_param = event_cameras[obs_name].camera_param
                    frame_obs = events2frame(obs, event_cam_param.get_height(),
                                             event_cam_param.get_width())
                    sensor = event_cameras[obs_name]
                    frame_obs = plot_roi(frame_obs.copy(),
                                         sensor.camera_param.get_roi())
                    noodle = curvature2noodle(self.ref_agent.curvature,
                                              sensor.camera_param,
                                              mode='camera')
                    frame_obs = cv2.polylines(frame_obs, [noodle], False,
                                              (0, 0, 255), 2)
                    obs_render = fit_img_to_ax(self._fig, self._axes[ax_name],
                                               frame_obs[:, :, ::-1])

                elif obs_name in lidars.keys():
                    if isinstance(obs, Pointcloud):
                        obs_render = None
                        ax = self._axes[ax_name]
                        ax.clear()
                        obs = obs[::10]  # sub-sample the pointcloud for vis

                        ax, scat = plot_pointcloud(
                            obs,
                            ax=ax,
                            color_by="z",
                            max_dist=20.,
                            car_dims=(self.ref_agent.length,
                                      self.ref_agent.width),
                            cmap="nipy_spectral")

                        # Plot the noodle
                        noodle = curvature2noodle(self.ref_agent.curvature,
                                                  mode='lidar')
                        ax.plot(noodle[:, 0], noodle[:, 1], '-r', linewidth=3)

                    else:  # dense image
                        obs = np.roll(obs, -obs.shape[1] // 4, axis=1)  # shift
                        obs = np.concatenate(np.split(obs, 2, axis=1),
                                             0)  # stack
                        obs = np.clip(4 * obs, 0, 255).astype(np.uint8)  # norm
                        obs = cv2.applyColorMap(obs, cv2.COLORMAP_JET)  # color
                        obs_render = fit_img_to_ax(self._fig,
                                                   self._axes[ax_name], obs)

                else:
                    logging.error(f'Unrecognized observation {obs_name}')
                    continue
                title = '{}: {}'.format(agent.id, obs_name)
                self._axes[ax_name].set_title(title,
                                              color='white',
                                              size=20,
                                              weight='bold')
                if obs_render is not None:
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
        """ Agent as a reference to compute poses of objects (e.g., cars, road)
            in visualization. """
        return self._world.agents[0]


def curvature2noodle(curvature: float,
                     camera_param: Optional[CameraParams] = None,
                     mode: Optional[str] = 'camera') -> np.ndarray:
    """ Construct a curly line (noodle) based on the curvature for visualizing
    steering control command.

    Args:
        curvature (float): Curvature (steering angle control command).
        camera_param (vista.entities.sensors.camera_utils.CameraParams): Camera parameters; used if
        mode is set to camera.
        mode (str): Sensor type for the visualization.

    Returns:
        np.ndarray: A curly line that visualizes the given curvature.

    Raises:
        NotImplementedError: Unrecognized mode to draw the noodle.

    """
    lookaheads = np.linspace(0, 15, 10)  # meters
    if mode == 'camera':
        assert camera_param is not None

        K = camera_param.get_K()
        normal = camera_param.get_ground_plane()[0:3]
        normal = np.reshape(normal, [1, 3])
        d = camera_param.get_ground_plane()[3]
        A, B, C = normal[0]

        radius = 1. / (curvature + 1e-8)

        z_vals = lookaheads
        y_vals = (d - C * z_vals) / B
        x_sq_r = radius**2 - z_vals**2 - (y_vals - d)**2
        x_vals = np.sqrt(x_sq_r[x_sq_r > 0]) - abs(radius)
        y_vals = y_vals[x_sq_r > 0]
        z_vals = z_vals[x_sq_r > 0]

        if radius < 0:
            x_vals *= -1

        world_coords = np.stack((x_vals, y_vals, z_vals))

        theta = camera_param.get_yaw()
        R = np.array([[np.cos(theta), 0.0, -np.sin(theta)], [0.0, 1.0, 0.0],
                      [np.sin(theta), 0.0, np.cos(theta)]])
        tf_world_coords = np.matmul(R, world_coords)
        img_coords = np.matmul(K, tf_world_coords)
        norm = np.divide(img_coords, img_coords[2] + 1e-10)

        valid_inds = np.multiply(norm[0] >= 0,
                                 norm[0] < camera_param.get_width())
        valid_inds = np.multiply(valid_inds, norm[1] >= 0)
        valid_inds = np.multiply(valid_inds,
                                 norm[1] < camera_param.get_height())

        noodle = norm[:2, valid_inds].astype(np.int32).T
    elif mode == 'lidar':
        turning_r = 1 / (curvature + 1e-8)
        shifts = (np.sqrt(turning_r**2 - lookaheads**2) - abs(turning_r))
        shifts = -1 * np.sign(turning_r) * shifts
        noodle = np.stack([lookaheads, shifts], axis=1)
    else:
        raise NotImplementedError(
            'Unrecognized mode {} in drawing noodle'.format(mode))

    return noodle


def plot_roi(img: np.ndarray,
             roi: List[int],
             color: Optional[List[int]] = (0, 0, 255),
             thickness: Optional[int] = 2) -> np.ndarray:
    """ Plot a bounding box that shows ROI on an image.

    Args:
        img (np.ndarray): An image to be plotted.
        roi (List[int]): Region of interest.
        color (List[int]): Color of the bounding box.
        thickness (int): Thickness of the bounding box.

    Returns:
        np.ndarray: An image with the ROI bounding box.

    """
    (i1, j1, i2, j2) = roi
    img = cv2.rectangle(img, (j1, i1), (j2, i2), color, thickness)
    return img


def events2frame(events: List[np.ndarray],
                 cam_h: int,
                 cam_w: int,
                 positive_color: Optional[List] = [255, 255, 255],
                 negative_color: Optional[List] = [212, 188, 114],
                 mode: Optional[int] = 2) -> np.ndarray:
    """ Convert event data to frame representation.

    Args:
        events (List[np.ndarray]): A list with entries as a collection of positive and
                                   negative events.
        cam_h (int): Height of the frame representation.
        cam_w (int): Width of the frame representation.
        positive_color (List): Color of positive events.
        negative_color (List): Color of negative events.
        mode (int): Mode for colorization.

    Returns:
        np.ndarray: Frame representation of event data.

    """
    if mode == 0:
        frame = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
        for color, p_events in zip([positive_color, negative_color], events):
            uv = np.concatenate(p_events)[:, :2]
            frame[uv[:, 0], uv[:, 1], :] = color
    elif mode == 1:
        frame_acc = np.zeros((cam_h, cam_w), dtype=np.int8)
        for polarity, p_events in zip([1, -1], events):
            for sub_p_events in p_events:
                uv = sub_p_events[:, :2]
                frame_acc[uv[:, 0], uv[:, 1]] += polarity

        frame = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
        frame[frame_acc > 0, :] = positive_color
        frame[frame_acc < 0, :] = negative_color
    elif mode == 2:
        frame_abs_acc = np.zeros((cam_h, cam_w), dtype=np.int8)
        frame = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
        for polarity, p_events in zip([1, -1], events):
            for sub_p_events in p_events:
                uv = sub_p_events[:, :2]
                add_c = np.array(
                    positive_color if polarity > 0 else negative_color)[None,
                                                                        ...]
                cnt = frame_abs_acc[uv[:, 0], uv[:, 1]][:, None]
                frame[uv[:, 0], uv[:, 1]] = (frame[uv[:, 0], uv[:, 1]] * cnt +
                                             add_c) / (cnt + 1)
                frame_abs_acc[uv[:, 0], uv[:, 1]] = cnt[:, 0] + 1
    else:
        raise NotImplementedError('Unknown mode {}'.format(mode))
    return frame


def plot_pointcloud(pcd,
                    color_by="z",
                    max_dist=None,
                    cmap="nipy_spectral",
                    car_dims=None,
                    ax=None,
                    scat=None,
                    s=1):
    """ Convert pointcloud to an image for visualization. """
    if ax is None:
        _, ax = plt.subplots()

    if max_dist is not None:
        pcd = pcd[pcd.dist < (max_dist * np.sqrt(2))]

    if color_by == "z":
        c = pcd.z
        vmin, vmax = (-2.5, 4)
    elif color_by == "intensity":
        c = np.log(1 + pcd.intensity)
        vmin, vmax = (1.7, 4.3)
    else:
        raise ValueError(f"unsupported color {color_by}")

    # Plot points
    if scat is None:
        scat = ax.scatter(pcd.x,
                          pcd.y,
                          c=c,
                          s=s,
                          vmin=vmin,
                          vmax=vmax,
                          cmap=cmap)
    else:
        scat.set_offsets(np.stack([pcd.x, pcd.y], axis=1))
        scat.set_clim(vmin, vmax)
        scat.set_color(getattr(plt.cm, cmap)(scat.norm(c)))

    # Plot car
    if car_dims is not None:
        l_car, w_car = car_dims
        ax.add_patch(
            patches.Rectangle(
                (-l_car / 2, -w_car / 2),
                l_car,
                w_car,
                fill=True  # remove background
            ))

    ax.set_xlim(-max_dist, max_dist)
    ax.set_ylim(-max_dist, max_dist)
    return ax, scat


def fig2img(fig: plt.Figure) -> np.ndarray:
    """ Convert a matplotlib figure to a numpy array. """
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    return img


def fit_img_to_ax(fig: plt.Figure, ax: plt.Axes,
                  img: np.ndarray) -> np.ndarray:
    """ Fit an image to an axis in a matplotlib figure. """
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
