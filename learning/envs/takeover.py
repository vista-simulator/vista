import gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from .base_env import BaseEnv


class Takeover(BaseEnv, MultiAgentEnv):
    def __init__(self, trace_paths, mesh_dir=None, task_mode='episodic', 
                 respawn_distance=15, speed_scale_range=[0.0, 0.8], **kwargs):
        super(Takeover, self).__init__(trace_paths, n_agents=2, 
            mesh_dir=mesh_dir, **kwargs)

        assert task_mode in ['episodic', 'infinite_horizon_dense', 'infinite_horizon_sparse']
        self.task_mode = task_mode
        self.respawn_distance = respawn_distance
        self.speed_scale_range = speed_scale_range

        # always use curvature only as action
        self.action_space = gym.spaces.Box(
                low=np.array([self.lower_curvature_bound]),
                high=np.array([self.upper_curvature_bound]),
                shape=(1,),
                dtype=np.float64)

        assert self.n_agents == 2, 'Only support 2 agents for now'

        self.perturb_heading_in_random_init = False # otherwise nominal traj following will fail

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        observations = self.wrap_data(observations)
        self.observation_for_render = self.wrap_data(self.observation_for_render) # for render
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
            action[agent_id] = np.array([human_curvature, \
                np.random.uniform(*self.speed_scale_range) * human_velocity])
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
        else:
            for agent, passed_this in zip(other_agents, passed):
                if passed_this:
                    self.random_init_agent_in_the_front(agent, self.respawn_distance, 
                        self.respawn_distance, self.ref_agent.human_dynamics)
                    # NOTE: no collision check as we only consider 2 agents
            if self.task_mode == 'infinite_horizon_sparse':
                reward[self.ref_agent_id] = np.sum(passed)
            else:
                pass # use non-crash reward
        done['__all__'] = done[self.ref_agent_id]
        return observation, reward, done, info

    def wrap_data(self, data):
        return {self.ref_agent_id: data[self.ref_agent_id]}

    def agent_sensors_setup(self, agent_i):
        # static agents don't need sensors
        if agent_i == self.ref_agent_idx:
            agent = self.world.agents[agent_i]
            camera = agent.spawn_camera()


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
    args = parser.parse_args()

    # initialize simulator
    env = Takeover(args.trace_paths, args.mesh_dir, args.task_mode, init_agent_range=[6,10])
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
