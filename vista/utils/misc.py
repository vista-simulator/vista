from typing import Optional
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
