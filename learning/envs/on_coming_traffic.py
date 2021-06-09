import gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from .base_env import BaseEnv


class OnComingTraffic(BaseEnv, MultiAgentEnv):
    def __init__(self, trace_paths, mesh_dir=None,
                 respawn_distance=15, soft_collision=0., soft_collision_ub=0.05, 
                 init_agent_range=[20., 30.], **kwargs):
        super(OnComingTraffic, self).__init__(trace_paths, n_agents=2, 
            mesh_dir=mesh_dir, init_agent_range=init_agent_range, **kwargs)

        self.respawn_distance = respawn_distance
        self.soft_collision = soft_collision
        self.soft_collision_ub = soft_collision_ub

        # use curvature only or with velocity as action
        self.action_space = gym.spaces.Box(
                low=np.array([self.lower_curvature_bound]),
                high=np.array([self.upper_curvature_bound]),
                shape=(1,),
                dtype=np.float64)

        assert self.n_agents == 2, 'Only support 2 agents for now'

        self.perturb_heading_in_random_init = False # otherwise nominal traj following will fail

        self.controllable_agents = {self.ref_agent_id: self.ref_agent}

        for i, agent in enumerate(self.world.agents):
            if i != self.ref_agent_idx:
                agent.direction = 'backward'

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        observations = self.wrap_data(observations)
        self.observation_for_render = self.wrap_data(self.observation_for_render) # for render
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
                # nominal curvature and the same speed but backward
                human_curvature = agent.trace.f_curvature(current_timestamp)
                action[agent_id] = np.array([human_curvature, -human_velocity])
        # step environment
        observation, reward, done, info = map(self.wrap_data, super().step(action))
        self.observation_for_render = self.wrap_data(self.observation_for_render)
        # modify reward function and done
        other_agents = [_a for _i, _a in enumerate(self.world.agents) if _i != self.ref_agent_idx]
        passed = [self.check_agent_pass_other(self.ref_agent, _a) for _a in other_agents]
        in_lane_center = self.check_agent_in_lane_center(self.ref_agent)
        reward[self.ref_agent_id] = 1 if in_lane_center and np.all(passed) else 0
        done[self.ref_agent_id] = (in_lane_center and np.all(passed)) or done[self.ref_agent_id]
        info[self.ref_agent_id]['success'] = in_lane_center and np.all(passed)
        info[self.ref_agent_id]['passed_cars'] = np.sum(passed)
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

    def reset_lat_lane_target(self):
        free_width = (self.ref_agent.trace.road_width - self.ref_agent.car_width) * 0.5
        self.lat_lane_target = np.random.uniform(-free_width, free_width)

    def check_agent_pass_other(self, agent, other_agent): # NOTE: override
        origin_dist = agent.trace.f_distance(agent.first_time)
        dist = agent.trace.f_distance(agent.get_current_timestamp()) - origin_dist
        other_dist = other_agent.trace.f_distance(other_agent.get_current_timestamp()) - origin_dist
        print(dist, other_dist, agent.current_frame_index, other_agent.current_frame_index)
        passed = (dist - other_dist) > ((other_agent.car_length + agent.car_length) / 2.)
        return passed

    def convert_to_scene_node(self, ego_agent, other_agents): # NOTE: override
        other_agents_nodes = []
        for i, agent in other_agents.items():
            # compute relative pose to ego agent
            agent_dynamics = agent.ego_dynamics.copy()
            # agent_dynamics.theta_state *= -1
            trans_x, trans_y, theta = self.compute_relative_transform( \
                agent_dynamics, ego_agent.human_dynamics)
            rot = np.array([0, 1, 0, theta])
            rot = rot / np.linalg.norm(rot) # unit vector for quaternion
            trans = np.array([trans_x, 0, trans_y]) 
            # compensate for camera position
            camera_offset = np.array(self.camera_offset)
            # camera_offset[2] *= -1
            trans = trans - camera_offset
            # compensate for camera rotation, pitch and yaw (check RIG.xml)
            # NOTE: this is different from the car heading theta
            long_dist = trans[2]
            trans[0] += long_dist * np.sin(self.camera_rotation[2])
            trans[1] += long_dist * np.sin(self.camera_rotation[1])
            trans[2] = long_dist * np.cos(self.camera_rotation[1])
            # convert to scene node
            agent_node = self.mesh_lib.get_mesh_node(i, trans, rot)
            other_agents_nodes.append(agent_node)

        return other_agents_nodes

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
    args = parser.parse_args()

    # initialize simulator
    env = OnComingTraffic(args.trace_paths, args.mesh_dir)
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
