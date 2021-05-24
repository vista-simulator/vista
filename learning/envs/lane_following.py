import gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from .base_env import BaseEnv


class LaneFollowing(BaseEnv, MultiAgentEnv):
    def __init__(self, trace_paths, **kwargs):
        super(LaneFollowing, self).__init__(trace_paths, n_agents=1, **kwargs)

        # always use curvature only as action
        self.action_space = gym.spaces.Box(
                low=np.array([self.lower_curvature_bound]),
                high=np.array([self.upper_curvature_bound]),
                shape=(1,),
                dtype=np.float64)


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
        '--preprocess',
        action='store_true',
        default=False,
        help='Use image preprocessor.')
    parser.add_argument(
        '--random-action',
        action='store_true',
        default=False,
        help='Use random action.')
    args = parser.parse_args()

    # initialize simulator
    env = LaneFollowing(args.trace_paths)
    if args.preprocess:
        from .wrappers import PreprocessObservation
        env = PreprocessObservation(env)
    env = MultiAgentMonitor(env, os.path.expanduser('~/tmp/monitor'), video_callable=lambda x: True, force=True)

    # run
    for ep in range(2):
        done = False
        obs = env.reset()
        ep_rew = 0
        ep_steps = 0
        while not done:
            act = dict()
            for _i, k in enumerate(env.agent_ids):
                if not args.random_action: # follow human trajectory
                    ts = env.world.agents[_i].get_current_timestamp()
                    act[k] = env.world.agents[_i].trace.f_curvature(ts)
                else: # random action
                    act[k] = env.action_space.sample() / 3.
            obs, rew, done, info = env.step(act)
            ep_rew += np.mean(list(rew.values()))
            done = np.any(list(done.values()))
            ep_steps += 1
        print('[{}th episodes] {} steps {} reward'.format(ep, ep_steps, ep_rew))