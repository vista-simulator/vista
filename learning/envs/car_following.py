import gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv


from .takeover import Takeover


class CarFollowing(Takeover, MultiAgentEnv):
    def __init__(self, trace_paths, mesh_dir=None, respawn_distance=15,
                 motion_model='random_speed', speed_scale_range=[0.0, 0.8], **kwargs):
        super(CarFollowing, self).__init__(trace_paths, mesh_dir=mesh_dir, 
            task_mode='episodic', respawn_distance=respawn_distance, 
            speed_scale_range=speed_scale_range, motion_model=motion_model,
            with_velocity=True, **kwargs)

    def step(self, action):
        observation, reward, done, info = super().step(action)
        other_agent = [agent for i, agent in enumerate(self.world.agents) if i != self.ref_agent_idx][0]

        in_lane_center = self.check_agent_in_lane_center(self.ref_agent)
        passed = self.check_agent_pass_other(self.ref_agent, other_agent)

        dist_to_other = np.linalg.norm(other_agent.ego_dynamics.numpy()[:2]-self.ref_agent.ego_dynamics.numpy()[:2])
        min_dist = (other_agent.car_length + self.ref_agent.car_length) / 2.
        in_range = dist_to_other < (min_dist * 1.5) and dist_to_other > (min_dist * 1.1) and not passed

        reward[self.ref_agent_id] = 0.01 if in_range and in_lane_center else 0.
        done[self.ref_agent_id] = done[self.ref_agent_id] or passed
        done['__all__'] = done[self.ref_agent_id]

        return observation, reward, done, info


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
    args = parser.parse_args()

    # initialize simulator
    env = CarFollowing(args.trace_paths, args.mesh_dir, init_agent_range=[6,12], respawn_distance=15)
    # env = MultiAgentMonitor(env, os.path.expanduser('~/tmp/monitor'), video_callable=lambda x: True, force=True)

    # run
    for ep in range(10):
        done = False
        obs = env.reset()
        ep_rew = 0
        ep_steps = 0
        while not done:
            act = dict()
            for k, a in env.controllable_agents.items():
                if True: # follow human trajectory
                    ts = a.get_current_timestamp()
                    act[k] = np.array([a.trace.f_curvature(ts), a.trace.f_speed(ts)])
                else:
                    act[k] = env.action_space.sample()
            obs, rew, done, info = env.step(act)
            ep_rew += np.mean(list(rew.values()))
            done = np.any(list(done.values()))
            ep_steps += 1
        print('[{}th episodes] {} steps {} reward'.format(ep, ep_steps, ep_rew))