import numpy as np

from .base import Base
from ..utils import misc


class LaneFollowing(Base):
    DEFAULT_CONFIG = {
        'reward_type': 'non-crash',
        'maximal_rotation': np.pi / 10.,
    }

    def __init__(self, **kwargs):
        super(LaneFollowing, self).__init__(**kwargs)

    def reset(self):
        self._world.reset()
        agent = self._world.agents[0]
        observations = agent.observations
        return observations

    def step(self, action):
        # Step agent and get observation
        agent = self._world.agents[0]
        agent.step_dynamics(action)
        agent.step_sensors()
        observations = agent.observations

        # Define terminal condition
        lat, long, theta = agent.relative_state.numpy()
        free_width = agent.trace.road_width - agent.width
        out_of_lane = np.abs(lat) > (free_width / 2.)
        exceed_rot = np.abs(theta) > self._config['maximal_rotation']
        done = out_of_lane or exceed_rot

        # Define reward
        reward_type = self._config['reward_type']
        if reward_type == 'non-crash':
            reward = 0. if done else 1.
        else:
            raise NotImplementedError('Unrecognized reward type {}'.format(reward_type))

        # Get info
        info = misc.fetch_agent_info(agent)
        info['out_of_lane'] = out_of_lane
        info['exceed_rot'] = exceed_rot

        return observations, reward, done, info
