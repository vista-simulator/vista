import os
from functools import partial
from collections import deque
import gym
import pickle5 as pickle
import numpy as np
import ray
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import get_trainable_cls

import misc
from .obstacle_avoidance import ObstacleAvoidance


class PlacingObstacle(ObstacleAvoidance, MultiAgentEnv):
    def __init__(self, trace_paths, mesh_dir=None, task_mode='episodic', 
                 model_path=None, respawn_distance=15, **kwargs):
        super(PlacingObstacle, self).__init__(trace_paths, mesh_dir=mesh_dir, 
            task_mode=task_mode, respawn_distance=respawn_distance, **kwargs)

        self.env_agent_id = 'env_agent_0'

        # action space is defined by how we can place obstacles
        # self.action_space = gym.spaces.Box(
        #         low=np.array([-0.05, -self.ref_agent.car_width / 2.]),
        #         high=np.array([0.05, self.ref_agent.car_width / 2.]),
        #         shape=(2,),
        #         dtype=np.float64)
        self.action_space = gym.spaces.Box( # DEBUG
                low=np.array([-self.ref_agent.car_width / 2.]),
                high=np.array([self.ref_agent.car_width / 2.]),
                shape=(1,),
                dtype=np.float64)

        # setup observation space
        self.road_buffer_size = 200 # unit is frame
        self.road = deque(maxlen=self.road_buffer_size)
        self.road_frame_index = deque(maxlen=self.road_buffer_size)

        self.render_observation_space = self.observation_space
        obs_size = self.road_buffer_size * 2 + 3 # road xy and ref agent xytheta
        self.observation_space = gym.spaces.Box(
            low=-50., # NOTE: hardcoded bound for birdseye map range
            high=50.,
            shape=(obs_size,),
            dtype=np.float64)

        # get car agent behavior
        self.agent_behavior = self.load_agent(model_path)
        # self.agent_behavior = lambda _x: np.array([0.0]) # DEBUG

        assert self.n_agents == 2, 'Only support 2 agents for now'
        
    def load_agent(self, model_path):
        # load config
        model_path = os.path.abspath(os.path.expanduser(model_path))
        config_dir = os.path.dirname(model_path)
        config_path = os.path.join(config_dir, 'params.pkl')
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, '../params.pkl')
        
        with open(config_path, 'rb') as f:
            config = pickle.load(f)

        # Overwrite some config with arguments
        config['num_workers'] = 0 # use single-process for each envs
        config['num_gpus'] = 0

        # Register custom model
        agent_env_creator = misc.register_custom_env(config['env'])
        misc.register_custom_model(config['model'])
        
        # load agent (assume ray is already initialized)
        assert ray.is_initialized()
        cls = get_trainable_cls('PPO')
        agent = cls(env=config['env'], config=config)
        agent.restore(model_path)

        # construct agent behavior NOTE: perhaps not good to create env instance here
        obs_wrapper_fns = []
        recur_unwrap_env = agent_env_creator(config['env_config'])
        while hasattr(recur_unwrap_env, 'observation'):
            obs_wrapper_fns.append(recur_unwrap_env.observation)
            recur_unwrap_env = recur_unwrap_env.env
        obs_wrapper_fns = obs_wrapper_fns[::-1]

        policy_mapping_fn = agent.config['multiagent']['policy_mapping_fn']
        def _agent_behavior(_obs, _obs_wrapper_fns):
            for _obs_wrapper_fn in _obs_wrapper_fns:
                _obs = _obs_wrapper_fn(_obs)
            return agent.compute_action(_obs, policy_id=policy_mapping_fn(self.ref_agent_id))
        agent_behavior = partial(_agent_behavior, _obs_wrapper_fns=obs_wrapper_fns)

        return agent_behavior

    def reset(self, **kwargs):
        # NOTE: other car already placed in the scene (this will cause problem for car agent with memory)
        self.agent_observations = super().reset(**kwargs)
        self.passed = [True] * (self.n_agents - 1) # allow env take action at the first step (exclude ref-agent)

        self.reset_scene_state()
        observation = self.wrap_env_data(self.get_scene_state())

        return observation

    def step(self, action):
        # env agent takes action only when the static agent is passed and required to be reinit
        # dtheta, lat_shift = action[self.env_agent_id]
        lat_shift = action[self.env_agent_id][0]; dtheta = 0 # DEBUG
        if lat_shift >= 0:
            lat_shift += self.ref_agent.car_width / 2.
        else:
            lat_shift -= self.ref_agent.car_width / 2.
        other_agents = [_a for _i, _a in enumerate(self.world.agents) if _i != self.ref_agent_idx]
        for agent, passed_this in zip(other_agents, self.passed):
            self.place_agent(agent, self.ref_agent.human_dynamics, self.respawn_distance, dtheta, lat_shift)
        # run simulation for agents
        any_agent_done = False
        agent_succeed = False
        while not agent_succeed and not any_agent_done:
            # augment static action for agents
            agent_actions = dict()
            for agent_id in self.agent_ids:
                if agent_id == self.ref_agent_id:
                    agent_actions[agent_id] = self.agent_behavior(self.agent_observations[agent_id])
                else:
                    agent_actions[agent_id] = self.static_action
            # step environment for agents TODO: need to find a way to use monitor
            self.agent_observations, agent_rewards, agent_dones, agent_infos = \
                map(self.wrap_data, super().step(agent_actions))
            any_agent_done = np.any(list(agent_dones.values()))
            # check ego-agent passing static agents and back to lane center
            in_lane_center = self.check_agent_in_lane_center(self.ref_agent)
            passed = [self.check_agent_pass_other(self.ref_agent, _a) for _a in other_agents]
            agent_succeed = np.all(passed) and in_lane_center
        self.passed = passed # used for reinit static agent in the next step
        # get step results for env agent
        observation = self.wrap_env_data(self.get_scene_state())
        reward = self.wrap_env_data(-np.sum(list(agent_rewards.values())))
        done = self.wrap_env_data(np.any(list(agent_dones.values())))
        done['__all__'] = agent_dones[self.ref_agent_id]
        info = self.wrap_env_data(agent_infos)
        
        return observation, reward, done, info

    def get_scene_state(self):
        # TODO: this should be in BaseEnv
        # update road (in global coordinate)
        while self.road_frame_index[-1] < (self.ref_agent.current_frame_index + self.road_buffer_size / 2):
            current_timestamp = self.get_timestamp_readonly(self.ref_agent, self.road_frame_index[-1])
            self.road_frame_index.append(self.road_frame_index[-1] + 1)
            next_timestamp = self.get_timestamp_readonly(self.ref_agent, self.road_frame_index[-1])
            self.road_dynamics.step(curvature=self.ref_agent.trace.f_curvature(current_timestamp),
                                    velocity=self.ref_agent.trace.f_speed(current_timestamp),
                                    delta_t=next_timestamp - current_timestamp)
            current_timestamp = next_timestamp
            self.road.append(self.road_dynamics.numpy()[:2])

        # update road in birds eye map (in reference agent coordinate)
        ref_x, ref_y, ref_theta = self.ref_agent.human_dynamics.numpy()
        road_in_ref = np.array(self.road) - np.array([ref_x, ref_y])
        c, s = np.cos(ref_theta), np.sin(ref_theta)
        R_T = np.array([[c, -s], [s, c]])
        road_in_ref = np.matmul(road_in_ref, R_T)

        # update agent in birds eye map (in reference agent coordinate)
        agent_xytheta_in_ref = self.compute_relative_transform(
            self.ref_agent.ego_dynamics, self.ref_agent.human_dynamics)

        # get scene state
        aug_road_in_ref = np.concatenate([np.zeros(\
            (self.road_buffer_size-road_in_ref.shape[0],2)), road_in_ref])
        scene_state = np.concatenate([aug_road_in_ref.reshape((-1,)), agent_xytheta_in_ref])

        return scene_state

    def reset_scene_state(self):
        # TODO: should be in BaseEnv
        self.road_frame_index.clear()
        self.road_frame_index.append(self.ref_agent.current_frame_index)
        self.road.clear()
        self.road.append(self.ref_agent.human_dynamics.numpy()[:2])
        self.road_dynamics = self.ref_agent.human_dynamics.copy()

    def get_timestamp_readonly(self, agent, index=0, current=False):
        # TODO: should be in BaseEnv
        index = agent.current_frame_index if current else index
        index = min(len(agent.trace.syncedLabeledTimestamps[
                agent.current_segment_index]) - 1, index)
        return agent.trace.syncedLabeledTimestamps[
            agent.current_segment_index][index]

    def wrap_env_data(self, data):
        return {self.env_agent_id: data}


if __name__ == "__main__":
    import argparse
    import cv2
    from .wrappers import MultiAgentMonitor

    # parse argument
    parser = argparse.ArgumentParser(description='Run wrapper test.')
    parser.add_argument(
        '--trace-paths',
        type=str,
        nargs='+',
        help='Paths to the traces to use for simulation.')
    parser.add_argument(
        '--mesh-dir',
        type=str,
        default=None,
        help='Directory of agents\' meshes.')
    parser.add_argument(
        '--task-mode',
        type=str,
        default='episodic',
        help='Task mode.')
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to agent checkpoint')
    args = parser.parse_args()

    # initialize simulator
    ray.init(
        local_mode=True,
        _temp_dir=os.path.expanduser('~/tmp'),
        include_dashboard=False)
    env = PlacingObstacle(args.trace_paths, args.mesh_dir, 
        args.task_mode, args.model_path, init_agent_range=[6,12])
    #env = MultiAgentMonitor(env, os.path.expanduser('~/tmp/monitor'), video_callable=lambda x: True, force=True)

    # run
    for ep in range(10):
        done = False
        obs = env.reset()
        ep_rew = 0
        ep_steps = 0
        while not done:
            act = dict()
            for k in [env.env_agent_id]:
                act[k] = env.action_space.sample()
            obs, rew, done, info = env.step(act)
            ep_rew += np.mean(list(rew.values()))
            done = np.any(list(done.values()))
            ep_steps += 1
        print('[{}th episodes] {} steps {} reward'.format(ep, ep_steps, ep_rew))
