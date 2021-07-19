from typing import Optional, Dict
import numpy as np
from shapely.geometry import box as Box
from shapely import affinity

from . import transform
from ..entities.agents import Car
from ..entities.agents.Dynamics import StateDynamics


def agent2poly(agent: Car, ref_dynamics: Optional[StateDynamics] = None) -> Box:
    """ Convert Agent object to polygon w.r.t. a reference dynamics. """
    ref_dynamics = agent.human_dynamics if ref_dynamics is None else ref_dynamics
    rel_pose = transform.compute_relative_latlongyaw(agent.ego_dynamics.numpy()[:3],
                                                     ref_dynamics.numpy()[:3])
    poly = Box(rel_pose[0] - agent.width / 2.,
               rel_pose[1] - agent.length / 2.,
               rel_pose[0] + agent.width / 2.,
               rel_pose[1] + agent.length / 2.)
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
