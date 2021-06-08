import gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from .base_env import BaseEnv


class CarFollowingv2(BaseEnv, MultiAgentEnv):
    def __init__(self, trace_paths, mesh_dir=None,
                 respawn_distance=15, soft_collision=0., soft_collision_ub=0.05, 
                 init_agent_range=[6., 8.], following_reward_coeff=0.001, task_mode='nominal',
                 free_width_mul=1.0, lane_change_freq=300, **kwargs):
        super(CarFollowingv2, self).__init__(trace_paths, n_agents=2, free_width_mul=free_width_mul,
            mesh_dir=mesh_dir, init_agent_range=init_agent_range, **kwargs)

        self.respawn_distance = respawn_distance
        self.soft_collision = soft_collision
        self.soft_collision_ub = soft_collision_ub
        self.following_reward_coeff = following_reward_coeff
        self.task_mode = task_mode
        self.lane_change_freq = lane_change_freq

        # use curvature only or with velocity as action
        self.action_space = gym.spaces.Box(
                low=np.array([self.lower_curvature_bound]),
                high=np.array([self.upper_curvature_bound]),
                shape=(1,),
                dtype=np.float64)

        assert self.n_agents == 2, 'Only support 2 agents for now'

        self.perturb_heading_in_random_init = False # otherwise nominal traj following will fail

        self.controllable_agents = {self.ref_agent_id: self.ref_agent}

        if self.task_mode == 'lane_change':
            self.init_scene_state(200)

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        observations = self.wrap_data(observations)
        self.observation_for_render = self.wrap_data(self.observation_for_render) # for render
        self.step_cnt = 0
        if self.task_mode == 'lane_change':
            other_agent = [_a for _i, _a in enumerate(self.world.agents) if _i != self.ref_agent_idx][0]
            self.lat_lane_target = other_agent.relative_state.translation_x
            self.reset_scene_state()
        return observations

    def step(self, action):
        # augment nominal velocity
        for agent_id, agent in zip(self.agent_ids, self.world.agents):
            current_timestamp = agent.get_current_timestamp()
            human_velocity = agent.trace.f_speed(current_timestamp)
            if agent_id == self.ref_agent_id:
                if action[agent_id].shape == ():
                    action[agent_id] = np.array([action[agent_id], human_velocity])
                else:
                    action[agent_id] = np.array([action[agent_id][0], human_velocity])
            else:
                if self.task_mode == 'nominal':
                    # nominal curvature and the same speed
                    human_curvature = agent.trace.f_curvature(current_timestamp)
                    action[agent_id] = np.array([human_curvature, human_velocity])
                elif self.task_mode == 'lane_change':
                    lookahead_dist = 5
                    dt = 1 / 30.
                    Kp = 0.8

                    if (self.step_cnt % self.lane_change_freq) == 0:
                        self.reset_lat_lane_target()

                    # get road vectors
                    road_in_agent, _ = self.get_scene_state(agent.ego_dynamics, concat=False)
                    road_in_agent = road_in_agent[road_in_agent[:,1] > 0] # drop road behind

                    if road_in_agent.shape[0] > 0:
                        # get target xy; use lookahead distance and apply later shift
                        dist = np.linalg.norm(road_in_agent[:,:2], axis=1)
                        tgt_idx = np.argmin(np.abs(dist - lookahead_dist))
                        dx, dy, dtheta = road_in_agent[tgt_idx]

                        lat_shift = self.lat_lane_target - agent.relative_state.translation_x
                        dx += lat_shift * np.cos(dtheta)
                        dy += lat_shift * np.sin(dtheta)
                        
                        # compute curvature
                        arc_len = human_velocity * dt
                        curvature = (Kp * np.arctan2(-dx, dy) * dt) / arc_len
                        curvature = np.clip(curvature, self.lower_curvature_bound, self.upper_curvature_bound)
                    else: # cutting off agent out-of-view from road
                        current_timestamp = agent.get_current_timestamp()
                        curvature = agent.trace.f_curvature(current_timestamp)
                    action[agent_id] = np.array([curvature, human_velocity])
                else:
                    raise NotImplementedError('Unrecognized task mode {} for car following'.format(self.task_mode))
        # step environment
        observation, reward, done, info = map(self.wrap_data, super().step(action))
        self.observation_for_render = self.wrap_data(self.observation_for_render)
        self.step_cnt += 1
        # modify reward function and done
        other_agent = [_a for _i, _a in enumerate(self.world.agents) if _i != self.ref_agent_idx][0]
        lat_shift = np.abs(self.ref_agent.relative_state.translation_x - other_agent.relative_state.translation_x)
        max_lat_shift = (self.ref_agent.trace.road_width - self.ref_agent.car_width) * self.free_width_mul * 2
        following_reward = 1. - (lat_shift / max_lat_shift) ** 2
        reward[self.ref_agent_id] = following_reward * self.following_reward_coeff
        info[self.ref_agent_id]['following_lat_shift'] = lat_shift
        if self.rigid_body_collision:
            for agent_id in reward.keys():
                reward[agent_id] -= self.rigid_body_collision_coef * float(info[agent_id]['collide'])
        if self.soft_collision > 0.:
            for i, agent_id in enumerate(reward.keys()):
                reward[agent_id] -= self.soft_collision * self.overlap_ratio[i]
                done[agent_id] = done[agent_id] or self.overlap_ratio[i] >= self.soft_collision_ub
        other_agent_off_lane, _ = self.check_agent_off_lane_or_max_rot(other_agent)
        done['__all__'] = done[self.ref_agent_id] or other_agent_off_lane
        # add other agent info
        other_agent_id = [v for v in self.agent_ids if v != self.ref_agent_id]
        assert len(other_agent_id) == 1
        other_agent_id = other_agent_id[0]
        other_agent = self.world.agents[self.agent_ids.index(other_agent_id)]
        info[self.ref_agent_id]['other_velocity'] = other_agent.model_velocity
        info[self.ref_agent_id]['other_translation'] = other_agent.relative_state.translation_x
        return observation, reward, done, info

    def reset_lat_lane_target(self):
        free_width = (self.ref_agent.trace.road_width - self.ref_agent.car_width) * 0.5
        self.lat_lane_target = np.random.uniform(-free_width, free_width)

    def wrap_data(self, data):
        return {self.ref_agent_id: data[self.ref_agent_id]}

    def agent_sensors_setup(self, agent_i, rendering_config):
        # static agents don't need sensors
        if agent_i == self.ref_agent_idx:
            agent = self.world.agents[agent_i]
            camera = agent.spawn_camera(rendering_config)


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
        '--preprocess',
        action='store_true',
        default=False,
        help='Use image preprocessor.')
    parser.add_argument(
        '--task-mode',
        type=str,
        default='nominal',
        help='Task mode.')
    args = parser.parse_args()

    # initialize simulator
    env = CarFollowingv2(args.trace_paths, args.mesh_dir, task_mode=args.task_mode)
    if args.preprocess:
        from .wrappers import PreprocessObservation
        env = PreprocessObservation(env)
    from .wrappers import BasicManeuverReward # DEBUG
    env = BasicManeuverReward(env, center_coeff=0.0, jitter_coeff=0.0001, inherit_reward=True) # DEBUG
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
                    act[k] = env.world.agents[_i].trace.f_curvature(ts)
                else: # random action
                    act[k] = env.action_space.sample()
            obs, rew, done, info = env.step(act)
            ep_rew += np.mean(list(rew.values()))
            done = np.any(list(done.values()))
            ep_steps += 1
        print('[{}th episodes] {} steps {} reward'.format(ep, ep_steps, ep_rew))
