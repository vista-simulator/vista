import gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from .base_env import BaseEnv


class MultiAgentCuttingOff(BaseEnv, MultiAgentEnv):
    def __init__(self, trace_paths, mesh_dir=None, respawn_distance=15, 
                 reward_mode=0, **kwargs):
        super(MultiAgentCuttingOff, self).__init__(trace_paths, n_agents=3, 
            mesh_dir=mesh_dir, **kwargs)

        self.respawn_distance = respawn_distance
        self.reward_mode = reward_mode

        # include velocity
        self.action_space = gym.spaces.Box(
            low=np.array([self.lower_curvature_bound, self.lower_velocity_bound]),
            high=np.array([self.upper_curvature_bound, self.upper_velocity_bound]),
            shape=(2,),
            dtype=np.float64)

        assert self.n_agents == 3, 'Only support 3 agents for now'

        self.controllable_agents = {'agent_0': self.world.agents[0], 'agent_1': self.world.agents[1]}
        self.special_agent_ids = {'cutting_off': 'agent_1', 'nominal': 'agent_2'}
        self.init_scene_state(200)

    def reset(self, **kwargs):
        # reset reference agent
        self.reset_agent(self.ref_agent, sync_with_ref=False)

        # sample situation
        nominal_agent = self.world.agents[self.agent_ids.index(self.special_agent_ids['nominal'])]
        cutting_off_agent = self.world.agents[self.agent_ids.index(self.special_agent_ids['cutting_off'])]

        self.situation = dict(nominal=dict(), cutting_off=dict())
        self.situation['nominal']['base_speed'] = np.random.uniform(0., 6.)
        
        nominal_car_len = nominal_agent.car_length
        cutting_off_car_len = cutting_off_agent.car_length
        self.situation['cutting_off']['spawn_dist'] = self.respawn_distance
        spawn_mul = np.random.uniform(2.0, 3.0)
        self.situation['nominal']['spawn_dist'] = self.situation['cutting_off']['spawn_dist'] + \
            spawn_mul * ((nominal_car_len + cutting_off_car_len) / 2.)

        left_or_right = np.random.choice([-1, 1])
        self.situation['nominal']['lat_shift'] = left_or_right * \
            np.random.uniform(nominal_agent.car_width / 2., nominal_agent.car_width)
        self.situation['cutting_off']['lat_shift'] = left_or_right * \
            np.random.uniform(0., cutting_off_agent.car_width / 2.)

        # reset non-reference agent
        polys = [self.agent2poly(self.ref_agent)]
        for i, agent in enumerate(self.world.agents):
            # skip reference agent
            if i == self.ref_agent_idx:
                continue            

            # init non-reference agents in the front of reference agent
            self.reset_agent(agent, ref_agent=self.ref_agent)

            key_by_val = [_k for _k, _v in self.special_agent_ids.items() if _v == self.agent_ids[i]]
            assert len(key_by_val) != 0, 'Cannot find matched agent ID {}'.format(self.agent_ids[i])
            sp_agent_id = key_by_val[0]
            spawn_dist = self.situation[sp_agent_id]['spawn_dist']
            lat_shift = self.situation[sp_agent_id]['lat_shift']
            self.place_agent(agent, self.ref_agent.human_dynamics, spawn_dist, 0, lat_shift)
            self.update_trace_and_first_time(agent)

        # reset sensors (should go after agent dynamics reset, which affects video stream reset)
        for agent in self.world.agents:
            for sensor in agent.sensors:
                sensor.reset()

        # reset mesh library (this assigns mesh to each agents)
        if self.n_agents > 1:
            self.reset_mesh_lib()

        # get sensor measurement
        observation = dict()
        for i, agent in enumerate(self.world.agents):
            other_agents = {_ai: self.world.agents[_ai] for _ai in range(self.n_agents) if _ai != i}
            other_agents = self.convert_to_scene_node(agent, other_agents)
            # NOTE: only support one sensor now
            if len(agent.sensors) == 1:
                # TODO: ValueError: Mesh is already bound to a context
                obs = agent.sensors[0].capture(agent.first_time, other_agents=other_agents)
            else:
                obs = None
            observation[self.agent_ids[i]] = obs

        # wrap data
        observation = self.wrap_data(observation)
        self.observation_for_render = observation # for render

        # for rigid body collision
        if self.rigid_body_collision:
            self.rigid_body_info = {
                'crash': np.zeros((self.n_agents, self.n_agents), dtype=bool),
                'overlap': np.zeros((self.n_agents, self.n_agents)),
            }

        # horizon count
        self.horizon_cnt = 0

        # reset road
        self.reset_scene_state()

        return observation

    def step(self, action):
        # augment action
        for agent_id, agent in zip(self.agent_ids, self.world.agents):
            if agent_id in self.controllable_agents.keys():
                continue
            else:
                # nominal curvature but slower speed
                current_timestamp = agent.get_current_timestamp()
                human_curvature = agent.trace.f_curvature(current_timestamp)
                speed = self.situation['nominal']['base_speed']
                action[agent_id] = np.array([human_curvature, speed])
        
        # step environment
        observation, reward, done, info = map(self.wrap_data, super().step(action))
        self.observation_for_render = self.wrap_data(observation)

        # define reward and terminal condition (passing nominal agent and back to lane)
        nominal_agent = self.world.agents[self.agent_ids.index(self.special_agent_ids['nominal'])]

        pass_nominal = dict()
        in_lane_center = dict()
        for agent_id, agent in self.controllable_agents.items():
            pass_nominal[agent_id] = self.check_agent_pass_other(agent, nominal_agent)
            in_lane_center[agent_id] = self.check_agent_in_lane_center(agent)
            success = pass_nominal[agent_id] and in_lane_center[agent_id]

            info[agent_id]['success'] = success

            # terminate episode if too far away behind
            for other_agent_idx, other_agent_id in enumerate(self.agent_ids):
                if other_agent_id == agent_id:
                    continue
                other_agent = self.world.agents[other_agent_idx]
                dist_to_other = np.linalg.norm(agent.ego_dynamics.numpy()[:2] - other_agent.ego_dynamics.numpy()[:2])
                too_far_behind = dist_to_other > (10 * (other_agent.car_length + agent.car_length) / 2.)
                done[agent_id] = too_far_behind or done[agent_id]

        if self.reward_mode == 0: 
            # done whenever one of the agents succeed or crash
            # reward of an agent is given when the agent succeed at the end of the episode
            for agent_id in self.controllable_agents.keys():
                success = info[agent_id]['success']
                done[agent_id] = success or done[agent_id]
                reward[agent_id] = float(done[agent_id] and success)
            done['__all__'] = np.any(list(done.values()))

        elif self.reward_mode == 1: 
            # done when one of the agents crash or all agents succeed
            # reward of all agents is given as the number of succeeded agents at the end of the episodes
            all_success = [info[aid]['success'] for aid in self.controllable_agents.keys()]
            for agent_id in self.controllable_agents.keys():
                done[agent_id] = np.all(all_success) or done[agent_id]
            done['__all__'] = np.any(list(done.values()))
            for agent_id in self.controllable_agents.keys():
                reward[agent_id] = float(done['__all__']) * np.sum(all_success)

        elif self.reward_mode == 2:
            # done when one of the agents crash or all agents succeed
            # reward of an agent is given if the agent succeed
            all_success = [info[aid]['success'] for aid in self.controllable_agents.keys()]
            for agent_id in self.controllable_agents.keys():
                done[agent_id] = np.all(all_success) or done[agent_id]
            done['__all__'] = np.any(list(done.values()))
            for agent_id in self.controllable_agents.keys():
                reward[agent_id] = float(info[agent_id]['success'])

        elif self.reward_mode == 3:
            # done whenever one of the agents succeed or crash
            # reward of an agent is given when the agent succeed at the end of the episode
            # with extra punishment on agents that crash
            for agent_id in self.controllable_agents.keys():
                success = info[agent_id]['success']
                crash_done = done[agent_id]
                done[agent_id] = success or done[agent_id]
                reward[agent_id] = float(done[agent_id]) * (float(success) - float(crash_done))
            done['__all__'] = np.any(list(done.values()))

        else:
            raise NotImplementedError('Unrecognized reward mode {}'.format(self.reward_mode))

        if self.rigid_body_collision:
            for agent_id in reward.keys():
                reward[agent_id] -= self.rigid_body_collision_coef * float(info[agent_id]['collide'])

        return observation, reward, done, info

    def wrap_data(self, data):
        new_data = dict()
        for aid in self.controllable_agents.keys():
            new_data[aid] = data[aid]
        return new_data

    def agent_sensors_setup(self, agent_i):
        controllable_agent_idx = [0, 1] # NOTE: hardcoded
        if agent_i in controllable_agent_idx:
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
    
    args = parser.parse_args()

    # initialize simulator
    env = MultiAgentCuttingOff(args.trace_paths, args.mesh_dir, init_agent_range=[6,12], respawn_distance=10)
    env = MultiAgentMonitor(env, os.path.expanduser('~/tmp/monitor'), video_callable=lambda x: True, force=True)

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
