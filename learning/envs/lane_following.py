import gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from .base_env import BaseEnv


class LaneFollowing(BaseEnv, MultiAgentEnv):
    def __init__(self, trace_paths):
        super(LaneFollowing, self).__init__(trace_paths, n_agents=1)

        # always use curvature only as action
        self.action_space = gym.spaces.Box(
                low=np.array([self.lower_curvature_bound]),
                high=np.array([self.upper_curvature_bound]),
                shape=(1,),
                dtype=np.float64)