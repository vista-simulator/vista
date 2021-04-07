import gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from .base_env import BaseEnv


class CuttingOff(BaseEnv, MultiAgentEnv):
    def __init__(self, trace_paths, mesh_dir=None, respawn_distance=15, **kwargs):
        super(CuttingOff, self).__init__(trace_paths, n_agents=3, 
            mesh_dir=mesh_dir, **kwargs)

        self.respawn_distance = respawn_distance

        # include velocity
        self.action_space = gym.spaces.Box(
            low=np.array([self.lower_curvature_bound, self.lower_velocity_bound]),
            high=np.array([self.upper_curvature_bound, self.upper_velocity_bound]),
            shape=(2,),
            dtype=np.float64)

        assert self.n_agents == 3, 'Only support 3 agents for now'

        self.controllable_agents = {self.ref_agent_id: self.ref_agent}
        self.special_agent_ids = dict(nominal='agent_1', cutting_off='agent_2')
        self.constant_speed = dict(nominal=0., cutting_off=0.)

    def reset(self, **kwargs):
        # reset reference agent
        self.reset_agent(self.ref_agent, sync_with_ref=False)

        # sample situation
        nominal_agent = self.world.agents[self.agent_ids.index(self.special_agent_ids['nominal'])]
        cutting_off_agent = self.world.agents[self.agent_ids.index(self.special_agent_ids['cutting_off'])]

        self.situation = dict(nominal=dict(), cutting_off=dict())
        self.situation['cutting_off']['base_speed'] = np.random.uniform(3., 6.) # slower than ego-agent but still moving
        self.situation['nominal']['base_speed'] = np.random.uniform(0., # slower than cutting-off agent
            0.9 * self.situation['cutting_off']['base_speed'])
        
        nominal_car_len = nominal_agent.car_length
        cutting_off_car_len = cutting_off_agent.car_length
        self.situation['cutting_off']['spawn_dist'] = self.respawn_distance
        spawn_mul = np.random.uniform(2.0, 5.0)
        spawn_mul = 2.0 # DEBUG
        self.situation['nominal']['spawn_dist'] = self.situation['cutting_off']['spawn_dist'] + \
            spawn_mul * ((nominal_car_len + cutting_off_car_len) / 2.)
        self.situation['cutting_off']['cutting_off_dist'] = np.random.uniform(1.0, spawn_mul) * \
            ((nominal_car_len + cutting_off_car_len) / 2.)
        self.situation['cutting_off']['cutting_off_dist'] = self.situation['nominal']['spawn_dist'] - self.situation['cutting_off']['spawn_dist'] # DEBUG

        left_or_right = np.random.choice([-1, 1])
        self.situation['nominal']['lat_shift'] = left_or_right * \
            np.random.uniform(nominal_agent.car_width / 2., nominal_agent.car_width)
        self.situation['cutting_off']['lat_shift'] = left_or_right * \
            np.random.uniform(cutting_off_agent.car_width / 2., cutting_off_agent.car_width)

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
        observation = []
        for i, agent in enumerate(self.world.agents):
            other_agents = {_ai: self.world.agents[_ai] for _ai in range(self.n_agents) if _ai != i}
            other_agents = self.convert_to_scene_node(agent, other_agents)
            # NOTE: only support one sensor now
            if len(agent.sensors) == 1:
                obs = agent.sensors[0].capture(agent.first_time, other_agents=other_agents)
            else:
                obs = None
            observation.append(obs)

        # wrap data
        observation = {self.ref_agent_id: observation[self.ref_agent_idx]}
        self.observation_for_render = observation # for render

        return observation

    def step(self, action):
        # augment action
        for agent_id, agent in zip(self.agent_ids, self.world.agents):
            if agent_id == self.ref_agent_id:
                continue
            elif agent_id == self.special_agent_ids['nominal']:
                # nominal curvature but slower speed
                current_timestamp = agent.get_current_timestamp()
                human_curvature = agent.trace.f_curvature(current_timestamp)
                speed = self.situation['nominal']['base_speed']
                action[agent_id] = np.array([human_curvature, speed])
            elif agent_id == self.special_agent_ids['cutting_off']:
                nominal_agent = self.world.agents[self.agent_ids.index(self.special_agent_ids['nominal'])]
                dist_to_nominal = np.linalg.norm(agent.ego_dynamics.numpy()[:2]\
                    - nominal_agent.ego_dynamics.numpy()[:2])
                perform_cutting_off = dist_to_nominal <= self.situation['cutting_off']['cutting_off_dist']

                speed = self.situation['cutting_off']['base_speed']
                if perform_cutting_off:
                    # get target dynamics (right next to nominal agent)
                    tgt_dynamics = nominal_agent.human_dynamics.copy()
                    left_or_right = -1 if self.situation['nominal']['lat_shift'] > 0 else 1
                    lat_shift = left_or_right * agent.car_width / 2.
                    long_shift = 0#-agent.car_length / 2.
                    tgt_theta = tgt_dynamics.numpy()[2]
                    tgt_dynamics.x_state += lat_shift * np.cos(tgt_theta) + long_shift * np.sin(tgt_theta)
                    tgt_dynamics.y_state += lat_shift * np.sin(tgt_theta) + long_shift * np.cos(tgt_theta)

                    # compute relative transform (in cutting-off agent coordinate)
                    dx, dy, dtheta = self.compute_relative_transform(\
                        agent.ego_dynamics, tgt_dynamics)
                    theta = agent.ego_dynamics.numpy()[2]
                    print(dx, dy, dtheta) # DEBUG

                    if -dy < 1:
                    #if self.check_agent_pass_other(agent, nominal_agent): # switch to nominal after passed
                        print('passed')
                        current_timestamp = agent.get_current_timestamp()
                        curvature = agent.trace.f_curvature(current_timestamp)
                        perform_cutting_off = False
                    else:
                        # compute rotation
                        Kp_alpha, Kp_beta = 10, 3
                        dt = 1 / 30.
                        to_pos_neg_pi = lambda _x: (_x + np.pi) % (2 * np.pi) - np.pi
                        alpha = to_pos_neg_pi(np.arctan2(dx, -dy) - theta)
                        beta = to_pos_neg_pi(tgt_theta - theta - alpha)
                        w = (Kp_alpha * alpha + Kp_beta * beta) * dt

                        # compute curvature
                        arc_len = speed * dt
                        curvature = w / arc_len
                        curvature = np.clip(curvature, self.lower_curvature_bound, self.upper_curvature_bound)
                else: # follow nominal trajectory
                    current_timestamp = agent.get_current_timestamp()
                    curvature = agent.trace.f_curvature(current_timestamp)
                    too_close = dist_to_nominal <= 1.1 * (agent.car_length + nominal_agent.car_length) / 2.
                    if too_close: # slow down when too close to the front car
                        speed = self.situation['nominal']['base_speed']
                action[agent_id] = np.array([curvature, speed])
            else:
                raise ValueError('Invalid agent ID {}'.format(agent_id))
        # step environment
        observation, reward, done, info = map(self.wrap_data, super().step(action))
        self.observation_for_render = self.wrap_data(self.observation_for_render)
        # define reward and terminal condition (passing nominal agent and back to lane)
        other_agents = [_a for _i, _a in enumerate(self.world.agents) if \
            self.agent_ids[_i] == self.special_agent_ids['nominal']]
        passed = [self.check_agent_pass_other(self.ref_agent, _a) for _a in other_agents]
        in_lane_center = self.check_agent_in_lane_center(self.ref_agent)

        reward[self.ref_agent_id] = 1 if in_lane_center and np.all(passed) else 0
        done[self.ref_agent_id] = (in_lane_center and np.all(passed)) or done[self.ref_agent_id]
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
    args = parser.parse_args()

    # initialize simulator
    env = CuttingOff(args.trace_paths, args.mesh_dir, init_agent_range=[6,12], respawn_distance=5)
    env = MultiAgentMonitor(env, os.path.expanduser('~/tmp/monitor'), video_callable=lambda x: True, force=True)

    # run
    for ep in range(1):
        done = False
        obs = env.reset()
        ep_rew = 0
        ep_steps = 0
        while not done:
            act = dict()
            for k, a in env.controllable_agents.items():
                if True: # follow human trajectory
                    ts = a.get_current_timestamp()
                    act[k] = np.array([a.trace.f_curvature(ts), 0.3 * a.trace.f_speed(ts)]) # DEBUG
                else:
                    act[k] = env.action_space.sample()
            obs, rew, done, info = env.step(act)
            ep_rew += np.mean(list(rew.values()))
            done = np.any(list(done.values()))
            ep_steps += 1
            # print(ep_steps) # DEBUG
        print('[{}th episodes] {} steps {} reward'.format(ep, ep_steps, ep_rew))