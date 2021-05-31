import gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from .base_env import BaseEnv


class Overtaking(BaseEnv, MultiAgentEnv):
    def __init__(self, trace_paths, mesh_dir=None, task_mode='episodic', 
                 respawn_distance=15, speed_scale_range=[0.0, 0.8], 
                 motion_model='random_speed', with_velocity=False, 
                 target_velocity=None, **kwargs):
        super(Overtaking, self).__init__(trace_paths, n_agents=2, 
            mesh_dir=mesh_dir, **kwargs)

        assert task_mode in ['episodic', 'infinite_horizon_dense', 'infinite_horizon_sparse']
        assert motion_model in ['random_speed', 'constant_speed']
        self.task_mode = task_mode
        self.respawn_distance = respawn_distance
        self.speed_scale_range = speed_scale_range
        self.motion_model = motion_model
        self.with_velocity = with_velocity
        self.target_velocity = target_velocity

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
        if self.motion_model == 'constant_speed':
            self.constant_speed = np.random.uniform(0., 6.) # make sure not faster than ego car
        return observations

    def step(self, action):
        # augment nominal action
        for agent_id, agent in zip(self.agent_ids, self.world.agents):
            if agent_id == self.ref_agent_id:
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
            reward[self.ref_agent_id] = 1 if in_lane_center and np.all(passed) else 0
            done[self.ref_agent_id] = (in_lane_center and np.all(passed)) or done[self.ref_agent_id]
            info[self.ref_agent_id]['success'] = in_lane_center and np.all(passed)
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
        done['__all__'] = done[self.ref_agent_id]
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
    env = Overtaking(args.trace_paths, args.mesh_dir, args.task_mode, init_agent_range=[6,10],
        with_velocity=args.with_velocity, target_velocity=args.target_velocity)
    if args.preprocess:
        from .wrappers import PreprocessObservation
        env = PreprocessObservation(env)
    env = MultiAgentMonitor(env, os.path.expanduser('~/tmp/monitor'), video_callable=lambda x: True, force=True)

    # run
    for ep in range(1):
        done = False
        obs = env.reset()
        ep_rew = 0
        ep_steps = 0
        while not done:
            act = dict()
            for _i, k in enumerate(env.agent_ids):
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
