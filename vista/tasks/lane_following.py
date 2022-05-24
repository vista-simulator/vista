from typing import Optional, List, Dict, Any
import numpy as np

from .. import World
from ..utils import logging, misc


def default_terminal_condition(task, agent_id, **kwargs):
    """ An example definition of terminal condition. """

    agent = [_a for _a in task.world.agents if _a.id == agent_id][0]

    def _check_out_of_lane():
        road_half_width = agent.trace.road_width / 2.
        return np.abs(agent.relative_state.x) > road_half_width

    def _check_exceed_max_rot():
        maximal_rotation = np.pi / 10.
        return np.abs(agent.relative_state.theta) > maximal_rotation

    out_of_lane = _check_out_of_lane()
    exceed_max_rot = _check_exceed_max_rot()
    done = out_of_lane or exceed_max_rot or agent.done
    other_info = {
        'done': done,
        'out_of_lane': out_of_lane,
        'exceed_max_rot': exceed_max_rot,
    }

    return done, other_info


def default_reward_fn(task, agent_id, **kwargs):
    """ An example definition of reward function. """
    reward = -1 if kwargs['done'] else 0  # simply encourage survival

    return reward, {}


class LaneFollowing:
    """ This class defines a simple lane following task in Vista. It basically
    handles vehicle state update of the ego car, rendering of specified sensors,
    and determing reward and terminal condition. The default terminal condition
    is set to (1) being out of lane (2) exceed maximal roation (3) reaching the
    end of the trace.

    Args:
        trace_paths (List[str]): A list of trace paths.
        trace_config (Dict): Configuration of the trace.
        car_configs (List[Dict]): Configuration of ``every`` cars.
        sensors_configs (List[Dict]): Configuration of sensors on ``every`` cars.
        task_config (Dict): Configuration of the task, which specifies reward function
                            and terminal condition. For more details, please check the
                            source code.
        logging_level (str): Logging level (``DEBUG``, ``INFO``, ``WARNING``,
                             ``ERROR``, ``CRITICAL``); default set to ``WARNING``.

    """
    DEFAULT_CONFIG = {
        'reward_fn': default_reward_fn,
        'terminal_condition': default_terminal_condition,
    }

    def __init__(self,
                 trace_paths: List[str],
                 trace_config: Dict,
                 car_config: Dict,
                 sensors_configs: Optional[List[Dict]] = [],
                 task_config: Optional[Dict] = dict(),
                 logging_level: Optional[str] = 'WARNING'):
        logging.setLevel(getattr(logging, logging_level))

        self._config = misc.merge_dict(task_config, self.DEFAULT_CONFIG)

        self._world: World = World(trace_paths, trace_config)
        agent = self._world.spawn_agent(car_config)
        for sensor_config in sensors_configs:
            sensor_type = sensor_config.pop('type')
            if sensor_type == 'camera':
                agent.spawn_camera(sensor_config)
            elif sensor_type == 'event_camera':
                agent.spawn_event_camera(sensor_config)
            elif sensor_type == 'lidar':
                agent.spawn_lidar(sensor_config)
            else:
                raise NotImplementedError(
                    f'Unrecognized sensor type {sensor_type}')

        self._distance = 0
        self._prev_xy = np.zeros((2, ))

    def reset(self):
        """ Reset the environment. This basically reset the :class:`World` object
        in Vista, which reset all (the only) agent in the world.

        Returns:
            Dict: A dictionary with keys as agent IDs and values as observation
            for each agent, which is also a dictionary with keys as sensor IDs
            and values as sensory measurements.

        """
        self._world.reset()
        agent = self._world.agents[0]
        observations = self._append_agent_id(agent.observations)
        self._distance = 0
        self._prev_xy = np.zeros((2, ))
        return observations

    def step(self, action, dt=1 / 30.):
        """ Step the environment. This involves updating agent's state based on
        the given actions and determining reward and termination.

        Args:
            actions (Dict[str, np.ndarray]):
                A dictionary with keys as agent IDs and values as actions
                to be executed to interact with the environment and other
                agents.
            dt (float): Elapsed time in second; default set to 1/30.

        Returns:
            Return a tuple (``dict_a``, ``dict_b``, ``dict_c``, ``dict_d``),
            where ``dict_a`` is the observation, ``dict_b`` is the reward,
            ``dict_c`` is whether the episode terminates, ``dict_d`` is additional
            informations for every agents; keys of every dictionary are agent IDs.

        """
        # Step agent and get observation
        agent = self._world.agents[0]
        action = np.array([action[agent.id][0], agent.human_speed])
        agent.step_dynamics(action, dt=dt)
        agent.step_sensors()
        observations = agent.observations

        # Define terminal condition
        done, info_from_terminal_condition = self.config['terminal_condition'](
            self, agent.id)

        # Define reward
        reward, _ = self.config['reward_fn'](self, agent.id,
                                             **info_from_terminal_condition)

        # Get info
        info = misc.fetch_agent_info(agent)
        info['out_of_lane'] = info_from_terminal_condition['out_of_lane']
        info['exceed_rot'] = info_from_terminal_condition['exceed_rot']

        current_xy = agent.ego_dynamics.numpy()[:2]
        self._distance += np.linalg.norm(current_xy - self._prev_xy)
        self._prev_xy = current_xy
        info['distance'] = self._distance

        # Pack output
        observations, reward, done, info = map(
            self._append_agent_id, [observations, reward, done, info])

        return observations, reward, done, info

    def set_seed(self, seed) -> None:
        """ Set random seed.

        Args:
            seed (int): Random seed.

        """
        self._seed = seed
        self._rng = np.random.default_rng(self.seed)
        self.world.set_seed(seed)

    def _append_agent_id(self, data):
        agent = self._world.agents[0]
        return {agent.id: data}

    @property
    def config(self) -> Dict:
        """ Configuration of this task. """
        return self._config

    @property
    def world(self) -> World:
        """ :class:`World` of this task. """
        return self._world

    @property
    def seed(self) -> int:
        """ Random seed for the task and the associated :class:`World`. """
        return self._seed
