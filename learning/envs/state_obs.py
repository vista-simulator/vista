import os
from importlib import import_module
from functools import partial
import gym
import pickle5 as pickle
import numpy as np
import ray
from ray.rllib.env.multi_agent_env import MultiAgentEnv

import misc
from . import *


def StateObs(task, **kwargs):
    task = globals()[task]
    class _StateObs(task, MultiAgentEnv):
        def __init__(self, **kwargs):
            self.drop_obs_space_def = True
            super(_StateObs, self).__init__(**kwargs)

            controllable_agent_ids = list(self.controllable_agents.keys())
            assert len(controllable_agent_ids) == 1 and controllable_agent_ids[0] == self.ref_agent_id, \
                'Only support controallable agent being the reference agent'

            self.require_handling_road = not hasattr(self, 'road')
            if self.require_handling_road:
                self.init_scene_state(200)

            self.fake_cam_height, self.fake_cam_width = 10, 15
            self.render_observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(self.fake_cam_height, self.fake_cam_width, 3),
                dtype=np.uint8)
            obs_size = self.road_buffer_size * 2 + 5 * self.n_agents # road xy and agent xytheta + velocity + curvature
            self.observation_space = gym.spaces.Box(
                low=-100., # NOTE: hardcoded bound for birdseye map range
                high=100.,
                shape=(obs_size,),
                dtype=np.float64)

            if not hasattr(self, 'wrap_data'):
                self.wrap_data = lambda _x: _x

            self.vehicle_states = {k: None for k in self.agent_ids}

        def reset(self, **kwargs):
            # reset base env
            observation = super().reset()

            # get state observation
            if self.require_handling_road:
                self.reset_scene_state()
            get_human_speed = lambda _a: _a.trace.f_speed([self.get_timestamp_readonly(_a, current=True)])
            get_human_curvature = lambda _a: _a.trace.f_curvature([self.get_timestamp_readonly(_a, current=True)])
            for agent_id, agent in zip(self.agent_ids, self.world.agents):
                self.vehicle_states[agent_id] = np.array([get_human_curvature(agent)[0], get_human_speed(agent)[0]])
            observation = self.get_state_obs()

            return observation

        def step(self, action):
            step_results = super().step(action)
            done_all = step_results[2]['__all__']
            observation, reward, done, info = map(self.wrap_data, step_results)
            for agent_id, agent in zip(self.agent_ids, self.world.agents):
                self.vehicle_states[agent_id] = np.array([agent.model_curvature, agent.model_velocity])
            observation = self.get_state_obs()
            done['__all__'] = done_all
            return observation, reward, done, info

        def get_state_obs(self):
            # get scene state
            ref_dynamics = self.ref_agent.human_dynamics
            road_in_ref, ego_in_ref = self.get_scene_state(concat=False, ref_dynamics=ref_dynamics)
            ego_in_ref = np.concatenate([ego_in_ref, self.vehicle_states[self.ref_agent_id]])
            others_in_ref = []
            for agent_idx, agent in enumerate(self.world.agents):
                if agent_idx == self.ref_agent_idx:
                    continue
                other_in_ref = self.compute_relative_transform(agent.ego_dynamics, ref_dynamics)
                other_in_ref = np.concatenate([other_in_ref, self.vehicle_states[self.agent_ids[agent_idx]]])
                others_in_ref.append(other_in_ref)

            # prepare observation
            aug_road_in_ref = np.concatenate([np.zeros(\
                (self.road_buffer_size-road_in_ref[:,:2].shape[0],2)), road_in_ref[:,:2]]) # NOTE: drop theta state
            state_obs = np.concatenate([aug_road_in_ref.reshape((-1,)), ego_in_ref] + others_in_ref)
            observation = {self.ref_agent_id: state_obs}

            for k in self.observation_for_render.keys():
                self.observation_for_render[k] = np.zeros(self.render_observation_space.shape,\
                    dtype=self.render_observation_space.dtype)
            self.observation_for_render = self.wrap_data(self.observation_for_render)

            return observation

        def agent_sensors_setup(self, agent_i):
            pass # don't need sensor; use ground truth state

    return _StateObs(**kwargs)


if __name__ == "__main__":
    import os
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
        '--task',
        type=str,
        default=None,
        help='Task')
    args = parser.parse_args()

    # initialize simulator
    env = StateObs(args.task, trace_paths=args.trace_paths, mesh_dir=args.mesh_dir)
    env = MultiAgentMonitor(env, os.path.expanduser('~/tmp/monitor'), video_callable=lambda x: True, force=True)

    # run
    for ep in range(1):
        done = False
        obs = env.reset()
        ep_rew = 0
        ep_steps = 0
        while not done:
            act = dict()
            for _i, k in enumerate(env.controllable_agents.keys()):
                if True: # follow human trajectory
                    ts = env.world.agents[_i].get_current_timestamp()
                    act[k] = env.world.agents[_i].trace.f_curvature(ts), 
                else: # random action
                    act[k] = env.action_space.sample()
            obs, rew, done, info = env.step(act)
            ep_rew += np.mean(list(rew.values()))
            done = np.any(list(done.values()))
            ep_steps += 1
        print('[{}th episodes] {} steps {} reward'.format(ep, ep_steps, ep_rew))
