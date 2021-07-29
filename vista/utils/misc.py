from typing import Optional, Dict, Union, List, Tuple
import numpy as np
import cv2
from shapely.geometry import box as Box
from shapely import affinity

from . import transform
from ..entities.agents import Car
from ..entities.agents.Dynamics import StateDynamics

Vec = Union[List, Tuple, np.ndarray]


def agent2poly(agent: Car,
               ref_dynamics: Optional[StateDynamics] = None) -> Box:
    """ Convert Agent object to polygon w.r.t. a reference dynamics. """
    ref_dynamics = agent.human_dynamics if ref_dynamics is None else ref_dynamics
    rel_pose = transform.compute_relative_latlongyaw(
        agent.ego_dynamics.numpy()[:3],
        ref_dynamics.numpy()[:3])
    poly = Box(rel_pose[0] - agent.width / 2., rel_pose[1] - agent.length / 2.,
               rel_pose[0] + agent.width / 2., rel_pose[1] + agent.length / 2.)
    poly = affinity.rotate(poly, np.degrees(rel_pose[2]))
    return poly


def merge_dict(dict1: Dict, dict2: Dict) -> Dict:
    """ Merge two dict, where dict1 has higher priority. """
    return dict(list(dict2.items()) + list(dict1.items()))


def fetch_agent_info(agent: Car):
    """ Get info from agent class. """
    info = dict(
        trace_path=agent.trace.trace_path,
        relative_state=agent.relative_state.numpy(),
        ego_dynamics=agent.ego_dynamics.numpy(),
        human_dynamics=agent.human_dynamics.numpy(),
        length=agent.length,
        width=agent.width,
        wheel_base=agent.wheel_base,
        steering_ratio=agent.steering_ratio,
        speed=agent.speed,
        curvature=agent.curvature,
        steering=agent.steering,
        tire_angle=agent.tire_angle,
        human_speed=agent.human_speed,
        human_curvature=agent.human_curvature,
        human_steering=agent.human_steering,
        human_tire_angle=agent.human_tire_angle,
        timestamp=agent.timestamp,
        frame_number=agent.frame_number,
        trace_index=agent.trace_index,
        segment_index=agent.segment_index,
        frame_index=agent.frame_index,
        trace_done=agent.done,
    )
    return info


def img2flow(img: np.ndarray,
             mag_minmax: Vec,
             flow_size: Optional[Vec] = None) -> np.ndarray:
    """ Convert HSV-encoded flow image to optical flow. """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # assume image is BGR
    if flow_size is not None:
        hsv = cv2.resize(hsv, flow_size[::-1])
    ang = hsv[..., 0] * 2. * np.pi / 180
    mag = hsv[..., 1] / 255. * (mag_minmax[1] - mag_minmax[0]) + mag_minmax[0]
    flow = np.stack(cv2.polarToCart(mag, ang), axis=-1)
    return flow


def biinterp(I0: np.ndarray, I1: np.ndarray, F_0_1: np.ndarray,
             F_1_0: np.ndarray, ts: float, t0: float, t1: float) -> np.ndarray:
    """ Interpolate frame with bidirectional flow. """
    t = (ts - t0) / (t1 - t0)
    temp = -t * (1 - t)
    fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]
    F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
    F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

    g_I0_F_t_0 = flow_backwarp(I0, F_t_0)
    g_I1_F_t_1 = flow_backwarp(I1, F_t_1)

    wCoeff = [1 - t, t]
    out = wCoeff[0] * g_I0_F_t_0 + wCoeff[1] * g_I1_F_t_1
    return out


def flow_backwarp(img: np.ndarray,
                  flow: np.ndarray,
                  use_pytorch: Optional[bool] = False) -> np.ndarray:
    """ Warp image based on optical flow. """
    H, W = img.shape[:2]
    gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))

    u = flow[:, :, 0]
    v = flow[:, :, 1]
    x = gridX + u
    y = gridY + v

    if use_pytorch:
        import torch
        x = 2 * (x / W - 0.5)
        y = 2 * (y / H - 0.5)
        grid = torch.from_numpy(np.stack((x, y), axis=2))

        img = torch.from_numpy(img / 255.).permute(2, 0, 1)
        out = torch.nn.functional.grid_sample(img[None, ...],
                                              grid[None, ...],
                                              align_corners=True)
        out = (out[0].permute(1, 2, 0).cpu().numpy() * 255.).astype(np.uint8)
    else:
        img = img / 255.
        grid = np.stack([x, y], axis=-1)
        out = cv2.remap(img, x.astype(np.float32), y.astype(np.float32),
                        cv2.INTER_LINEAR)
        out = np.clip(out * 255, 0, 255).astype(np.uint8)
    return out
