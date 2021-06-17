import gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from .base_env import BaseEnv


class Overtaking(BaseEnv, MultiAgentEnv):
    def __init__(self, trace_paths, mesh_dir=None, task_mode='episodic', 
                 respawn_distance=15, speed_scale_range=[0.0, 0.8], 
                 motion_model='random_speed', constant_speed_range=[0., 6.], 
                 with_velocity=False, target_velocity=None, 
                 soft_collision=0., soft_collision_ub=0.05, 
                 ego_constant_speed_range=None, **kwargs):
        super(Overtaking, self).__init__(trace_paths, n_agents=2, 
            mesh_dir=mesh_dir, **kwargs)

        assert task_mode in ['episodic', 'infinite_horizon_dense', 'infinite_horizon_sparse']
        assert motion_model in ['random_speed', 'constant_speed']
        self.task_mode = task_mode
        self.respawn_distance = respawn_distance
        self.speed_scale_range = speed_scale_range
        self.motion_model = motion_model
        self.constant_speed_range = constant_speed_range
        self.with_velocity = with_velocity
        self.target_velocity = target_velocity
        self.soft_collision = soft_collision
        self.soft_collision_ub = soft_collision_ub
        self.ego_constant_speed_range = ego_constant_speed_range

        # use curvature only or with velocity as action
        if self.with_velocity:
            self.action_space = gym.spaces.Box(
                low=np.array([self.lower_curvature_bound, self.lower_velocity_bound]),
                high=np.array([self.upper_curvature_bound, self.upper_velocity_bound]),
                shape=(2,),
                dtype=np.float64)
        else:
            self.action_space = gym.spaces.Box(
                    low=np.array([self.lower_curvature_bound]),
                    high=np.array([self.upper_curvature_bound]),
                    shape=(1,),
                    dtype=np.float64)

        assert self.n_agents == 2, 'Only support 2 agents for now'

        self.perturb_heading_in_random_init = False # otherwise nominal traj following will fail

        self.controllable_agents = {self.ref_agent_id: self.ref_agent}

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        observations = self.wrap_data(observations)
        self.observation_for_render = self.wrap_data(self.observation_for_render) # for render
        if self.ego_constant_speed_range is not None:
            self.ego_constant_speed = np.random.uniform(*self.ego_constant_speed_range)
        if self.motion_model == 'constant_speed':
            low = self.constant_speed_range[0]
            high = self.constant_speed_range[1]
            if hasattr(self, 'ego_constant_speed'):
                high = min(high, self.ego_constant_speed)
            self.constant_speed = np.random.uniform(low, high) # make sure not faster than ego car
        return observations

    def step(self, action):
        # augment nominal action
        for agent_id, agent in zip(self.agent_ids, self.world.agents):
            if agent_id == self.ref_agent_id:
                if self.ego_constant_speed_range is not None:
                    ego_velocity = self.ego_constant_speed
                else:
                    ego_velocity = agent.trace.f_speed(agent.get_current_timestamp())
                if action[agent_id].shape == ():
                    action[agent_id] = np.array([action[agent_id], ego_velocity])
                else:
                    action[agent_id] = np.array([action[agent_id][0], ego_velocity])
                continue
            # nominal curvature but slower speed
            current_timestamp = agent.get_current_timestamp()
            human_curvature = agent.trace.f_curvature(current_timestamp)
            human_velocity = agent.trace.f_speed(current_timestamp)
            if self.motion_model == 'random_speed':
                speed = np.random.uniform(*self.speed_scale_range) * human_velocity
            else:
                speed = self.constant_speed
            action[agent_id] = np.array([human_curvature, speed])
        # step environment
        observation, reward, done, info = map(self.wrap_data, super().step(action))
        self.observation_for_render = self.wrap_data(self.observation_for_render)
        # modify reward function and done
        other_agents = [_a for _i, _a in enumerate(self.world.agents) if _i != self.ref_agent_idx]
        passed = [self.check_agent_pass_other(self.ref_agent, _a) for _a in other_agents]
        if self.task_mode == 'episodic': # success when passed all cars and back to lane center
            in_lane_center = self.check_agent_in_lane_center(self.ref_agent)
            good_terminal_cond = self.get_agent_dist_diff(self.ref_agent, other_agents[0]) > 10 or \
                (self.get_agent_dist_diff(self.ref_agent, other_agents[0]) > 5 and self.ref_agent.trace_done)
            if good_terminal_cond: # already pass
                if in_lane_center:
                    reward[self.ref_agent_id] = 2
                else:
                    reward[self.ref_agent_id] = 1
            else:
                reward[self.ref_agent_id] = 0
            done[self.ref_agent_id] = good_terminal_cond or done[self.ref_agent_id]
            info[self.ref_agent_id]['success'] = in_lane_center and good_terminal_cond
            info[self.ref_agent_id]['passed_cars'] = np.sum(passed) * float(good_terminal_cond)
        else:
            for agent, passed_this in zip(other_agents, passed):
                if passed_this:
                    self.random_init_agent_in_the_front(agent, self.respawn_distance, 
                        self.respawn_distance, self.ref_agent.human_dynamics)
                    self.reset_mesh_lib() # NOTE: reset all agents
                    # NOTE: no collision check as we only consider 2 agents
            if self.task_mode == 'infinite_horizon_sparse':
                reward[self.ref_agent_id] = np.sum(passed)
            else:
                pass # use non-crash reward
        if self.with_velocity:
            # terminate episode if too far away behind
            origin_dist = self.ref_agent.trace.f_distance(self.ref_agent.first_time)
            dist = self.ref_agent.trace.f_distance(self.ref_agent.get_current_timestamp()) - origin_dist
            fail_to_catch_up = []
            not_succeed_after_pass = []
            for other_agent in other_agents:
                other_dist = other_agent.trace.f_distance(other_agent.get_current_timestamp()) - origin_dist
                too_far_behind = (other_dist - dist) > (10 * (other_agent.car_length + self.ref_agent.car_length) / 2.)
                fail_to_catch_up.append(too_far_behind)
                too_far_beyond = (dist - other_dist) > (10 * (other_agent.car_length + self.ref_agent.car_length) / 2.)
                not_succeed_after_pass.append(too_far_beyond)
            done[self.ref_agent_id] = done[self.ref_agent_id] or np.any(fail_to_catch_up) or np.any(too_far_beyond)

            # reward to track target speed
            if self.target_velocity is not None:
                ref_agent_speed = info[self.ref_agent_id]['model_velocity']
                velo_rew = 1 - (self.target_velocity - ref_agent_speed) / self.target_velocity
                velo_rew = np.clip(velo_rew, 0., 1.) * 0.001
                reward[self.ref_agent_id] += velo_rew
        if self.rigid_body_collision:
            for agent_id in reward.keys():
                reward[agent_id] -= self.rigid_body_collision_coef * float(info[agent_id]['collide'])
        if self.soft_collision > 0.:
            for i, agent_id in enumerate(reward.keys()):
                reward[agent_id] -= self.soft_collision * self.overlap_ratio[i]
                done[agent_id] = done[agent_id] or self.overlap_ratio[i] >= self.soft_collision_ub
        done['__all__'] = done[self.ref_agent_id]
        # add other agent info
        other_agent_id = [v for v in self.agent_ids if v != self.ref_agent_id]
        assert len(other_agent_id) == 1
        other_agent_id = other_agent_id[0]
        other_agent = self.world.agents[self.agent_ids.index(other_agent_id)]
        info[self.ref_agent_id]['other_velocity'] = other_agent.model_velocity
        info[self.ref_agent_id]['other_translation'] = other_agent.relative_state.translation_x
        info[self.ref_agent_id]['pose_wrt_others'] = self.compute_relative_transform(self.ref_agent.ego_dynamics, other_agent.ego_dynamics)
        info[self.ref_agent_id]['other_dim'] = [other_agent.car_length, other_agent.car_width]
        return observation, reward, done, info

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
        '--task-mode',
        type=str,
        default='episodic',
        help='Task mode.')
    parser.add_argument(
        '--with-velocity',
        default=False,
        action='store_true',
        help='Include velocity in action space.')
    parser.add_argument(
        '--target-velocity',
        default=None,
        type=float,
        help='Target velocity.')
    parser.add_argument(
        '--preprocess',
        action='store_true',
        default=False,
        help='Use image preprocessor.')
    args = parser.parse_args()

    # initialize simulator
    env = Overtaking(args.trace_paths, args.mesh_dir, args.task_mode, init_agent_range=[8,5],
        with_velocity=args.with_velocity, target_velocity=args.target_velocity, 
        curv_reset_mode='segment_start', init_lat_shift_range=[1.5,2.0],
        collision_overlap_threshold=0.5, soft_collision_ub=0.05,
        soft_collision=0.01, dilate_ref_agent=[1.0,0.4])# DEBUG, rendering_config={'use_lighting': False})
    if args.preprocess:
        from .wrappers import PreprocessObservation
        env = PreprocessObservation(env, standardize=False, color_jitter=[0.5,0.7,0.5,0.0], randomize_at_episode=True)
    from .wrappers import BasicManeuverReward
    env = BasicManeuverReward(env, center_coeff=0.001, jitter_coeff=0.001, inherit_reward=True)
    tmp_dir = os.path.join(os.environ['TMPDIR'], 'monitor')
    env = MultiAgentMonitor(env, tmp_dir, video_callable=lambda x: True, force=True)

    # run
    for ep in range(10):
        done = False
        obs = env.reset()
        ep_rew = 0
        ep_steps = 0
        while not done:
            act = dict()
            for _i, k in enumerate(env.agent_ids):
                if False: # follow human trajectory
                    ts = env.world.agents[_i].get_current_timestamp()
                    act[k] = env.world.agents[_i].trace.f_curvature(ts)
                else: # random action
                    act[k] = env.action_space.sample()
            obs, rew, done, info = env.step(act)
            ep_rew += np.mean(list(rew.values()))
            done = np.any(list(done.values()))
            ep_steps += 1
        print('[{}th episodes] {} steps {} reward'.format(ep, ep_steps, ep_rew))
