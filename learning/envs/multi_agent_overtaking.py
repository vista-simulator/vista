import gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from .base_env import BaseEnv


class MultiAgentOvertaking(BaseEnv, MultiAgentEnv):
    def __init__(self, trace_paths, mesh_dir=None, task_mode='episodic',
                 with_velocity=False, velocity_range=[5., 15.], min_velocity_diff=4., 
                 pass_reward=1., svo_theta=0, passed_dist=10., agent_failure_reward=0., 
                 **kwargs):
        super(MultiAgentOvertaking, self).__init__(trace_paths, n_agents=2,
            mesh_dir=mesh_dir, free_width_mul=1.0, **kwargs) # NOTE: otherwise episode end during init
        assert task_mode in ['episodic', 'infinite_horizon']
        self.task_mode = task_mode
        self.with_velocity = with_velocity
        self.velocity_range = velocity_range
        self.min_velocity_diff = min_velocity_diff
        self.pass_reward = pass_reward
        self.svo_theta = np.deg2rad(svo_theta)
        self.passed_dist = passed_dist
        self.agent_failure_reward = agent_failure_reward

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
        self.perturb_heading_in_random_init = False # TODO: not sure whether to keep this

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self._sample_velocity()
        self.initial_agent_order = self.get_agent_order()
        self.n_passed = {k: 0 for k in self.agent_ids}
        return observations

    def step(self, action):
        # set lower velocity for the front car; only when not using velocity
        if not self.with_velocity:
            for i, speed in zip(self.initial_agent_order, self.sampled_velocity):
                agent_id = self.agent_ids[i]
                if isinstance(action[agent_id], float):
                    action[agent_id] = np.array([action[agent_id], speed])
                else:
                    action[agent_id] = np.concatenate([action[agent_id], np.array([speed])])
        # step environment
        observation, reward, done, info = super().step(action)
        # determine success and terminal condition (and reinit agent)
        agent_front = self.world.agents[self.initial_agent_order[0]]
        agent_behind_idx = self.initial_agent_order[1]
        agent_behind_id = self.agent_ids[agent_behind_idx]
        agent_behind = self.world.agents[agent_behind_idx]
        passed = [self.check_agent_pass_other(agent_behind, _a) for _i, _a in \
            enumerate(self.world.agents) if _i != agent_behind_idx][0]
        in_lane_center = self.check_agent_in_lane_center(agent_behind)
        passed_long_enough = self.get_agent_dist(agent_behind, agent_front) > self.passed_dist
        success = passed and in_lane_center and passed_long_enough
        if success:
            if self.task_mode == 'infinite_horizon':
                self._sample_velocity()
                self.initial_agent_order = self.get_agent_order()
            else:
                done = {k: v or success for k, v in done.items()}
        too_far_behind = self.check_agent_too_far_behind(agent_behind, agent_front)
        done = {k: v or too_far_behind for k, v in done.items()} # TODO: not sure if ending both agents is good
        done['__all__'] = np.any(list(done.values()))
        # modify reward 
        reward = {k: 0 for k in reward.keys()}
        if success:
            # SVO reward
            for agent_id in reward.keys():
                if agent_id == agent_behind_id:
                    reward[agent_id] = self.pass_reward
                else:
                    reward[agent_id] = self.pass_reward * np.cos(self.svo_theta)
        if self.rigid_body_collision:
            for agent_id in reward.keys():
                reward[agent_id] -= self.rigid_body_collision_coef * float(info[agent_id]['collide'])
        if self.agent_failure_reward:
            for agent_id in reward.keys():
                off_lane_or_max_rot = info[agent_id]['off_lane'] or info[agent_id]['max_rot']
                collide_key = 'collide' if 'collide' in info[agent_id].keys() else 'has_collided'
                collide = False if self.rigid_body_collision else info[agent_id][collide_key]
                reward[agent_id] -= self.agent_failure_reward * (float(off_lane_or_max_rot) + float(collide))
        # add info
        for agent_id in info.keys():
            info[agent_id]['success'] = success

        return observation, reward, done, info

    def get_agent_dist(self, agent, other_agent):
        xy1 = agent.ego_dynamics.numpy()[:2]
        xy2 = other_agent.ego_dynamics.numpy()[:2]
        return np.linalg.norm(xy1 - xy2)

    def check_agent_too_far_behind(self, agent, other_agent):
        origin_dist = agent.trace.f_distance(agent.first_time)
        dist = agent.trace.f_distance(agent.get_current_timestamp()) - origin_dist
        other_dist = other_agent.trace.f_distance(other_agent.get_current_timestamp()) - origin_dist
        too_far_behind = (other_dist - dist) > (10 * (agent.car_length + other_agent.car_length) / 2.)
        return too_far_behind

    def _sample_velocity(self):
        self.sampled_velocity = np.random.uniform(*self.velocity_range, self.n_agents)
        self.sampled_velocity = np.sort(self.sampled_velocity)
        # make sure speed difference is at leat some value
        self.sampled_velocity[1] = max(self.sampled_velocity[1], 
            self.sampled_velocity[0]+self.min_velocity_diff)


if __name__ == '__main__':
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
    args = parser.parse_args()
    
    # initialize simulator
    env = MultiAgentOvertaking(args.trace_paths, args.mesh_dir, args.task_mode,
        init_agent_range=[6,10], with_velocity=args.with_velocity)
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