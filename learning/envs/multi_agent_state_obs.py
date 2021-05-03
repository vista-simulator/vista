import os
from importlib import import_module
from functools import partial
import gym
import pickle5 as pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from shapely.geometry import LineString
from descartes import PolygonPatch
import ray
from ray.rllib.env.multi_agent_env import MultiAgentEnv

import misc
from . import *


def MultiAgentStateObs(task, **kwargs):
    task = globals()[task]
    class _MultiAgentStateObs(task, MultiAgentEnv):
        def __init__(self, aug_extra_obs=False, to_bev_map=False, with_vis_obs=False, **kwargs):
            self.with_vis_obs = with_vis_obs
            self.drop_obs_space_def = True and not self.with_vis_obs
            super(_MultiAgentStateObs, self).__init__(**kwargs)

            self.aug_extra_obs = aug_extra_obs
            self.to_bev_map = to_bev_map

            self.require_handling_road = not hasattr(self, 'road')
            if self.require_handling_road:
                self.init_scene_state(200)

            # define observation space
            self.fake_cam_height, self.fake_cam_width = 10, 15
            self.render_observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(self.fake_cam_height, self.fake_cam_width, 3),
                dtype=np.uint8)
            if self.to_bev_map:
                figsize_in_pix = (80, 120) # 4x smaller

                state_obs_space = gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(figsize_in_pix[1], figsize_in_pix[0], 3),
                    dtype=np.uint8)
                if self.with_vis_obs:
                    self.observation_space = gym.spaces.Tuple([self.observation_space, state_obs_space])
                else:
                    self.observation_space = state_obs_space
            else:
                obs_size = self.road_buffer_size * 2 + 5 * self.n_agents # road xy and agent xytheta + velocity + curvature
                if self.aug_extra_obs and hasattr(self, 'extra_obs'):
                    obs_size += self.extra_obs.shape[0]
                state_obs_space = gym.spaces.Box(
                    low=-100., # NOTE: hardcoded bound for birdseye map range
                    high=100.,
                    shape=(obs_size,),
                    dtype=np.float64)
                if self.with_vis_obs:
                    self.observation_space = gym.spaces.Tuple([self.observation_space, state_obs_space])
                else:
                    self.observation_space = state_obs_space

            if not hasattr(self, 'wrap_data'):
                self.wrap_data = lambda _x: _x

            self.vehicle_states = {k: None for k in self.agent_ids}

            if self.to_bev_map:
                # initialize plot
                colors = list(cm.get_cmap('Set1').colors)
                rgba2rgb = lambda rgba: np.clip((1 - rgba[:3]) * rgba[3] + rgba[:3], 0., 1.)
                colors = [np.array(list(c) + [0.6]) for c in colors]
                colors = list(map(rgba2rgb, colors))
                self.road_color = list(cm.get_cmap('Dark2').colors)[-1]
                self.agent_colors = colors

                self.artists = dict()
                n_agents_with_sensor = 0
                px = 1/plt.rcParams['figure.dpi']  # pixel in inches
                figsize = (figsize_in_pix[0]*px, figsize_in_pix[1]*px)
                self.fig = plt.figure(figsize=figsize)
                self.fig.subplots_adjust(left=0.01, right=0.98, bottom=0.02, top=0.90, wspace=0.03, hspace=0.03)
                self.fig.patch.set_facecolor('black') # use black background
                self.gs = self.fig.add_gridspec(5, 4 * n_agents_with_sensor + 2)

                # birds eye map
                self.ax_birdseye = self.fig.add_subplot(self.gs[:,-2:])
                self.ax_birdseye.set_facecolor('black')
                self.ax_birdseye.set_xticks([])
                self.ax_birdseye.set_yticks([])
                self.birdseye_map_size = (30, 20)

        def reset(self, **kwargs):
            # reset base env
            observation = super().reset()

            if self.to_bev_map:
                # set birds eye map size
                self.ax_birdseye.set_xlim(-self.birdseye_map_size[1]/2., self.birdseye_map_size[1]/2.)
                self.ax_birdseye.set_ylim(-self.birdseye_map_size[0]/2., self.birdseye_map_size[0]/2.)

            # get state observation
            if self.require_handling_road:
                self.reset_scene_state()
            get_human_speed = lambda _a: _a.trace.f_speed([self.get_timestamp_readonly(_a, current=True)])
            get_human_curvature = lambda _a: _a.trace.f_curvature([self.get_timestamp_readonly(_a, current=True)])
            for agent_id, agent in zip(self.agent_ids, self.world.agents):
                self.vehicle_states[agent_id] = np.array([get_human_curvature(agent)[0], get_human_speed(agent)[0]])
            state_obs = self.get_state_obs()
            if not self.to_bev_map and (self.aug_extra_obs and hasattr(self, 'extra_obs')):
                for k in self.controllable_agents.keys():
                    state_obs[k] = np.concatenate([state_obs[k], self.extra_obs[k]])

            # prepare observation
            if self.with_vis_obs:
                for k in self.controllable_agents.keys():
                    observation[k] = [observation[k], state_obs[k]]
            else:
                observation = state_obs

            return observation

        def step(self, action):
            step_results = super().step(action)
            done_all = step_results[2]['__all__']
            observation, reward, done, info = map(self.wrap_data, step_results)
            for agent_id, agent in zip(self.agent_ids, self.world.agents):
                self.vehicle_states[agent_id] = np.array([agent.model_curvature, agent.model_velocity])
            
            state_obs = self.get_state_obs()
            if not self.to_bev_map and (self.aug_extra_obs and hasattr(self, 'extra_obs')):
                for k in self.controllable_agents.keys():
                    state_obs[k] = np.concatenate([state_obs[k], self.extra_obs[k]])

            if self.with_vis_obs:
                for k in self.controllable_agents.keys():
                    observation[k] = [observation[k], state_obs[k]]
            else:
                observation = state_obs

            done['__all__'] = done_all
            return observation, reward, done, info

        def get_state_obs(self):
            update = True
            observation = dict()
            for ref_agent_id, ref_agent in self.controllable_agents.items():
                ref_agent_idx = self.agent_ids.index(ref_agent_id)
                ref_dynamics = ref_agent.human_dynamics

                road_in_ref, ego_in_ref = self.get_scene_state(concat=False, ref_dynamics=ref_dynamics, update=update)
                update = False

                if self.to_bev_map:
                    # update road in bev map
                    patch = LineString(road_in_ref).buffer(ref_agent.trace.road_width / 2.)
                    patch = PolygonPatch(patch, fc=self.road_color, ec=self.road_color, zorder=1)
                    self.update_patch(self.ax_birdseye, 'patch:road', patch)

                    # update agent in birds eye map (in reference agent coordinate)
                    for i, (agent_id, agent) in enumerate(zip(self.agent_ids, self.world.agents)):
                        poly = self.agent2poly(agent, ref_dynamics)
                        patch = PolygonPatch(poly, fc=self.agent_colors[i], ec=self.agent_colors[i], zorder=2)
                        self.update_patch(self.ax_birdseye, 'patch:{}'.format(agent_id), patch)

                    # get observation
                    observation[ref_agent_id] = self.fig2img(self.fig)
                else:
                    # get scene state
                    ego_in_ref = np.concatenate([ego_in_ref, self.vehicle_states[ref_agent_id]])
                    others_in_ref = []
                    for agent_idx, agent in enumerate(self.world.agents):
                        if agent_idx == ref_agent_idx:
                            continue
                        other_in_ref = self.compute_relative_transform(agent.ego_dynamics, ref_dynamics)
                        other_in_ref = np.concatenate([other_in_ref, self.vehicle_states[self.agent_ids[agent_idx]]])
                        others_in_ref.append(other_in_ref)

                    # prepare observation
                    aug_road_in_ref = np.concatenate([np.zeros(\
                        (self.road_buffer_size-road_in_ref[:,:2].shape[0],2)), road_in_ref[:,:2]]) # NOTE: drop theta state
                    state_obs = np.concatenate([aug_road_in_ref.reshape((-1,)), ego_in_ref] + others_in_ref)
                    observation[ref_agent_id] = state_obs

            for k in self.observation_for_render.keys():
                self.observation_for_render[k] = np.zeros(self.render_observation_space.shape,\
                    dtype=self.render_observation_space.dtype)
            self.observation_for_render = self.wrap_data(self.observation_for_render)

            return observation

        def agent_sensors_setup(self, agent_i):
            if self.with_vis_obs:
                super().agent_sensors_setup(agent_i)
            else:
                pass # don't need sensor; use ground truth state

        def fig2img(self, fig):
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            img = np.asarray(buf)[:,:,:3].copy()
            return img

        def update_patch(self, ax, name, patch):
            if name in self.artists:
                self.artists[name].remove()
            ax.add_patch(patch)
            self.artists[name] = patch

        def obs_for_render(self, obs):
            return obs

    return _MultiAgentStateObs(**kwargs)


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
    parser.add_argument(
        '--to-bev-map',
        default=False,
        action='store_true',
        help='Convert state observation to bev map.')
    parser.add_argument(
        '--with-vis-obs',
        default=False,
        action='store_true',
        help='With visual observation.')
    args = parser.parse_args()

    # initialize simulator
    env = MultiAgentStateObs(args.task, trace_paths=args.trace_paths, mesh_dir=args.mesh_dir, 
        to_bev_map=args.to_bev_map, with_vis_obs=args.with_vis_obs)
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
