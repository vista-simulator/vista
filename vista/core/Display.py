from typing import Optional, Dict, Tuple, List, Any
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from shapely.geometry import LineString
from descartes import PolygonPatch

from . import World
from ..entities.agents.Dynamics import StateDynamics, update_with_perfect_controller, \
                                       curvature2tireangle
from ..entities.agents import Car
from ..entities.sensors import Camera
from ..utils import logging


class Display:
    DEFAULT_DISPLAY_CONFIG = {
        'road_buffer_size': 200,
        'birdseye_map_size': (30, 20),
    }

    def __init__(self, world: World, fps: Optional[float] = 30, 
                 display_config: Optional[Dict] = dict()):
        # Get arguments
        self._world: World = world
        self._fps: float = fps
        display_config.update(self.DEFAULT_DISPLAY_CONFIG)
        self._config: Dict = display_config

        # Initialize data for plotting
        self._road: deque[np.ndarray] = deque(maxlen=self._config['road_buffer_size'])
        self._road_frame_idcs: deque[int] = deque(maxlen=self._config['road_buffer_size'])
        self._road_dynamics: StateDynamics = StateDynamics()

        # Get agents with sensors (to be visualized)
        self._agents_with_sensors: List[Car] = []
        for agent in self.world.agents:
            if len(agent.sensors) > 0:
                is_camera = [isinstance(_v, Camera) for _v in agent.sensors]
                if len(is_camera) > 0:
                    self._agents_with_sensors.append(agent)
                if len(is_camera) < len(agent.sensors):
                    logging.warning('Cannot visualize sensor other than Camera')

        # Specify colors for agents and road
        colors = list(cm.get_cmap('Set1').colors)
        rgba2rgb = lambda rgba: np.clip((1 - rgba[:3]) * rgba[3] + rgba[:3], 0., 1.)
        colors = [np.array(list(c) + [0.6]) for c in colors]
        colors = list(map(rgba2rgb, colors))
        self._agent_colors: List[Tuple] = colors
        self._road_color: Tuple = list(cm.get_cmap('Dark2').colors)[-1]

        # TODO: Initialize figure
        self._artists: Dict[Any] = dict()
        self._axes: Dict[plt.Axes] = dict()

        # TODO: Initialize birds eye view

        # TODO: Initialize plot for sensory measurement

    def reset(self) -> None:
        # Reset road deque
        self._road.clear()
        self._road.append(self.ref_agent.human_dynamics.numpy()[:2])
        self._road_dynamics = self.ref_agent.human_dynamics.copy()
        self._road_frame_idcs.clear()
        self._road_frame_idcs.append(self.ref_agent.frame_index)

    def render(self):
        # Update road (in global coordinate)
        exceed_end = False
        while self._road_frame_idcs[-1] < (self.ref_agent.frame_index + \
            self.road_buffer_size / 2.) and not exceed_end:
            ts, _ = self._get_timestamp(self._road_frame_idcs[-1])
            self._road_frame_idcs.append(self._road_frame_idcs[-1] + 1)
            next_ts, exceed_end = self._get_timestamp(self._road_frame_idcs[-1])

            state = [curvature2tireangle(self.ref_agent.trace.f_curvature(ts), 
                                         self.ref_agent.wheel_base),
                     self.ref_agent.trace.f_speed(ts)]
            update_with_perfect_controller(state, next_ts - ts, self._road_dynamics)

        # TODO: Update road in birds eye map (in reference agent's coordinate)

        # TODO: Update agent in birds eye map

        # TODO: Update sensory measurements

        # Convert to image
        img = fig2img(self._fig)

        return img

    def _update_patch(self, ax: plt.Axes, name: str, patch: PolygonPatch) -> None:
        if name in self._artists:
            self._artists[name].remove()
        ax.add_patch(patch)
        self._artists[name] = patch

    def _get_timestamp(self, frame_index: int) -> Tuple[float, bool]:
        return self.ref_agent.trace.get_master_timestamp( \
            self.ref_agent.segment_index, frame_index, check_end=True)

    @property
    def ref_agent(self) -> Car:
        return self._world.agents[0]


def fig2img(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:,:,:3]
    return img


def fit_img_to_ax(fig: plt.Figure, ax: plt.Axes, img: np.ndarray) -> np.ndarray:
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
