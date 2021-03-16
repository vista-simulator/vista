import gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from .base_env import BaseEnv


class ObstacleAvoidance(BaseEnv, MultiAgentEnv):
    def __init__(self, trace_paths, mesh_dir=None, **kwargs):
        super(ObstacleAvoidance, self).__init__(trace_paths, n_agents=2, 
            mesh_dir=mesh_dir, **kwargs)

        # always use curvature only as action
        self.action_space = gym.spaces.Box(
                low=np.array([self.lower_curvature_bound]),
                high=np.array([self.upper_curvature_bound]),
                shape=(1,),
                dtype=np.float64)

        self.static_action = np.array([0, 0])

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        observations = self.wrap_data(observations)
        self.observation = self.wrap_data(self.observation) # for render
        return observations

    def step(self, action):
        # augment static action
        for agent_id in self.agent_ids:
            if agent_id == self.ref_agent_id:
                continue
            action[agent_id] = self.static_action
        # step environment
        observation, reward, done, info = map(self.wrap_data, super().step(action))
        self.observation = self.wrap_data(self.observation)
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
    args = parser.parse_args()

    # initialize simulator
    env = ObstacleAvoidance(args.trace_paths, args.mesh_dir, init_agent_range=[6,10])
    env = MultiAgentMonitor(env, os.path.expanduser('~/tmp/monitor'), video_callable=lambda x: True, force=True)

    # run
    for ep in range(5):
        done = False
        obs = env.reset()
        ep_rew = 0
        ep_steps = 0
        while not done:
            act = dict()
            for k in env.agent_ids:
                act[k] = env.action_space.sample()
            obs, rew, done, info = env.step(act)
            ep_rew += np.mean(list(rew.values()))
            done = np.any(list(done.values()))
            ep_steps += 1
        print('[{}th episodes] {} steps {} reward'.format(ep, ep_steps, ep_rew))
