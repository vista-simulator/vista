import numpy as np

from .base import Base
from ..utils import misc


class LaneFollowing(Base):
    DEFAULT_CONFIG = {
        'reward_type': 'non-crash',
        'non_crash_reward': 1.,
        'maximal_rotation': np.pi / 10.,
    }

    def __init__(self, **kwargs):
        super(LaneFollowing, self).__init__(**kwargs)

    def reset(self):
        self._world.reset()
        agent = self._world.agents[0]
        observations = self._append_agent_id(agent.observations)
        return observations

    def step(self, action):
        # Step agent and get observation
        agent = self._world.agents[0]
        action = np.array([action[agent.id][0], agent.human_speed])
        agent.step_dynamics(action)
        agent.step_sensors()
        observations = agent.observations

        # Define terminal condition
        lat, long, theta = agent.relative_state.numpy()
        free_width = agent.trace.road_width - agent.width
        out_of_lane = np.abs(lat) > (free_width / 2.)
        exceed_rot = np.abs(theta) > self._config['maximal_rotation']
        done = out_of_lane or exceed_rot or agent.done

        # Define reward
        reward_type = self._config['reward_type']
        if reward_type == 'non-crash':
            reward = 0. if done else self._config['non_crash_reward']
        else:
            raise NotImplementedError(
                'Unrecognized reward type {}'.format(reward_type))

        # Get info
        info = misc.fetch_agent_info(agent)
        info['out_of_lane'] = out_of_lane
        info['exceed_rot'] = exceed_rot

        # Pack output
        observations, reward, done, info = map(self._append_agent_id, 
                                               [observations, reward, done, info])

        return observations, reward, done, info

    def _append_agent_id(self, data):
        agent = self._world.agents[0]
        return {agent.id: data}
