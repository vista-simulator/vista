from typing import Optional, List, Dict, Any
import numpy as np
from shapely.geometry import box as Box
from shapely import affinity

from .. import World
from ..entities.agents.Car import Car
from ..entities.agents.Dynamics import StateDynamics
from ..utils import logging, misc, transform


def default_terminal_condition(task, agent_id):
    """ An example definition of terminal condition. """

    agent = [_a for _a in task.world.agents if _a.id == agent_id][0]

    def _check_out_of_lane():
        road_half_width = agent.trace.road_width / 2.
        return np.abs(agent.relative_state.x) > road_half_width

    def _check_crash():
        other_agents = [_a for _a in task.world.agents if _a.id != agent_id]
        agent2poly = lambda _x: misc.agent2poly(
            _x, ref_dynamics=agent.human_dynamics)
        poly = agent2poly(agent)
        other_polys = list(map(agent2poly, other_agents))
        overlap = compute_overlap(poly, other_polys) / poly.area
        crashed = np.any(overlap > task.config['overlap_threshold'])
        return crashed

    out_of_lane = _check_out_of_lane()
    crashed = _check_crash()

    return out_of_lane or crashed or agent.done


def default_reward_fn(task, agent_id):
    """ An example definition of reward function. """
    done = default_terminal_condition(task, agent_id)
    reward = -1 if done else 0  # simply encourage survival

    return reward


class MultiAgentBase:
    """ TODO """
    DEFAULT_CONFIG = {
        'overlap_threshold': 0.05,
        'max_resample_tries': 10,
        'init_dist_range': [5., 10.],
        'init_lat_noise_range': [-1., 1.],
        'init_yaw_noise_range': [-0.0, 0.0],
        'reward_fn': default_reward_fn,
        'terminal_condition': default_terminal_condition
    }

    def __init__(self,
                 trace_paths: List[str],
                 trace_config: Dict,
                 n_agents: int,
                 car_configs: List[Dict],
                 sensors_configs: List[List[Dict]],
                 task_config: Optional[Dict] = dict(),
                 logging_level: Optional[str] = 'WARNING'):
        logging.setLevel(getattr(logging, logging_level))
        assert len(
            car_configs
        ) == n_agents, 'Number of car config is not consistent with number of agents'
        assert len(
            sensors_configs
        ) == n_agents, 'Number of sensors config is not consistent with number of agents'
        assert car_configs[0][
            'lookahead_road'], '\'lookahead_road\' in the first car config should be set to True'

        self._config = misc.merge_dict(task_config, self.DEFAULT_CONFIG)
        self._world: World = World(trace_paths, trace_config)
        for i in range(n_agents):
            agent = self._world.spawn_agent(car_configs[i])
            for sensor_config in sensors_configs[i]:
                sensor_type = sensor_config.pop('type')
                if sensor_type == 'camera':
                    agent.spawn_camera(sensor_config)
                else:
                    raise NotImplementedError(
                        f'Unrecognized sensor type {sensor_type}')

        self.set_seed(0)

    def reset(self) -> Dict:
        """ TODO """
        # Reset world; all agents are initialized at the same pointer to the trace
        new_trace_index, new_segment_index, new_frame_index = \
            self.world.sample_new_location()
        for agent in self.world.agents:
            agent.reset(new_trace_index,
                        new_segment_index,
                        new_frame_index,
                        step_sensors=False)

        # Randomly initialize ado agents in the front
        ref_dynamics = self.ego_agent.human_dynamics
        polys = [misc.agent2poly(self.ego_agent, ref_dynamics)]
        for agent in self.world.agents:
            if agent == self.ego_agent:
                continue

            collision_free = False
            resample_tries = 0
            while not collision_free and resample_tries < self.config[
                    'max_resample_tries']:
                self._randomly_place_agent(agent)
                poly = misc.agent2poly(agent, ref_dynamics)
                overlap = compute_overlap(poly, polys) / poly.area
                collision_free = np.all(
                    overlap <= self.config['overlap_threshold'])

                resample_tries += 1
            polys.append(poly)

        # Reset mesh library
        if len(self.world.agents) > 1:
            pass  # DEBUG
            #self._reset_meshlib()

        # Get observation
        observations = self._get_observations()

        return observations

    def step(self, actions):
        """ Step the environment. This includes updating agents' states, synthesizing
        agents' observations, checking terminal conditions, and computing rewards.
        TODO: probably want to separate an environment from a RL environment.

        Args:
            actions (Dict[str, np.ndarray]):
                A dictionary with keys as agent IDs and values as actions
                to be executed to interact with the environment and other
                agents.

        Returns:
            Return a tuple (``dict_a``, ``dict_b``, ``dict_c``, ``dict_d``),
            where ``dict_a`` is the observation, ``dict_b`` is the reward,
            ``dict_c`` is whether the episode terminates, ``dict_d`` is additional
            informations for every agents; keys of every dictionary are agent IDs.

        """
        # Update agents' dynamics (state)
        for agent in self.world.agents:
            action = actions[agent.id]
            agent.step_dynamics(action)

        # Get agents' sensory measurement
        observations = self._get_observations()

        # Check terminal conditions
        dones = dict()
        terminal_condition = self.config['terminal_condition']
        for agent in self.world.agents:
            dones[agent.id] = terminal_condition(self, agent.id)

        # Compute reward
        rewards = dict()
        reward_fn = self.config['reward_fn']
        for agent in self.world.agents:
            rewards[agent.id] = reward_fn(self, agent.id)

        # Get info
        infos = dict()

        return observations, rewards, dones, infos

    def set_seed(self, seed) -> None:
        """ Set random seed.

        Args:
            seed (int): Random seed.

        """
        self._seed = seed
        self._rng = np.random.default_rng(self.seed)
        self.world.set_seed(seed)

    def _randomly_place_agent(self, agent: Car):
        # Randomly sampled a pose in the front of ego agent that is still on
        # the road. This can be achieved by,
        # (1) randomly sampling a distance from the ego agent
        # (2) fetch the closest pointer from the road kept by the ego agent
        # (3) slightly perturb the associated pose.
        tgt_dist = self._rng.uniform(*self.config['init_dist_range'])

        road = np.array(self.ego_agent.road)
        dist_from_ego = np.linalg.norm(road[:, :2], axis=1)
        tgt_idx = np.argmin(np.abs(tgt_dist - dist_from_ego))
        tgt_pose = road[tgt_idx].copy()

        lat_noise = self._rng.uniform(*self.config['init_lat_noise_range'])
        tgt_pose[0] += lat_noise * np.cos(tgt_pose[2])
        tgt_pose[1] += lat_noise * np.sin(tgt_pose[2])
        yaw_noise = self._rng.uniform(*self.config['init_yaw_noise_range'])
        tgt_pose[2] += yaw_noise

        # Place agent given the randomly sampled pose
        agent.ego_dynamics.update(*tgt_pose)
        agent.step_dynamics(tgt_pose[-2:], dt=1e-8)

    def _reset_meshlib(self):
        raise NotImplementedError

    def _get_observations(self):
        observations = dict()
        for agent in self.world.agents:
            agent.step_sensors()
            observations[agent.id] = agent.observations
        return observations

    @property
    def config(self) -> Dict:
        """ Configuration of this task. """
        return self._config

    @property
    def ego_agent(self) -> Car:
        """ Ego agent. """
        return self.world.agents[0]

    @property
    def world(self) -> World:
        """ :class:`World` of this task. """
        return self._world

    @property
    def seed(self) -> int:
        """ Random seed for the task and the associated :class:`World`. """
        return self._seed


def compute_overlap(poly: Box, polys: List[Box]) -> List[float]:
    """ Compute overlapping area between 1 polygons and N polygons.

    Args:
        poly (shapely.geometry.Box): A polygon.
        poly (List): A list of polygon.

    Returns:
        List[float]: Intersecting area between polygons.

    """
    n_polys = len(polys)
    overlap = np.zeros((n_polys))
    for i in range(n_polys):
        intersection = polys[i].intersection(poly)
        overlap[i] = intersection.area
    return overlap
