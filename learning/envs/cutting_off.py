import gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from .base_env import BaseEnv


class CuttingOff(BaseEnv, MultiAgentEnv):
    def __init__(self, trace_paths, mesh_dir=None, respawn_distance=15, 
                 target_velocity=None, n_passed_reward=True, car_following_bonus=0., 
                 cutoff_immediately=False, cutoff_at_reset_prob=None, 
                 give_pass_reward_immediately=False, **kwargs):
        super(CuttingOff, self).__init__(trace_paths, n_agents=3, 
            mesh_dir=mesh_dir, **kwargs)

        self.respawn_distance = respawn_distance
        self.target_velocity = target_velocity
        self.n_passed_reward = n_passed_reward
        self.car_following_bonus = car_following_bonus
        self.cutoff_immediately = cutoff_immediately
        self.cutoff_at_reset_prob = cutoff_at_reset_prob
        self.give_pass_reward_immediately = give_pass_reward_immediately
        if self.cutoff_at_reset_prob is not None:
            assert self.cutoff_immediately
        self.extra_obs = np.zeros((3,))

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

        self.init_scene_state(200)

    def reset(self, **kwargs):
        # reset reference agent
        self.reset_agent(self.ref_agent, sync_with_ref=False)

        # sample situation
        nominal_agent = self.world.agents[self.agent_ids.index(self.special_agent_ids['nominal'])]
        cutting_off_agent = self.world.agents[self.agent_ids.index(self.special_agent_ids['cutting_off'])]

        self.situation = dict(nominal=dict(), cutting_off=dict())
        self.situation['cutting_off']['base_speed'] = np.random.uniform(5., 7.) # moving
        self.situation['nominal']['base_speed'] = np.random.uniform(0., # slower than cutting-off agent
            0.9 * self.situation['cutting_off']['base_speed'])
        
        nominal_car_len = nominal_agent.car_length
        cutting_off_car_len = cutting_off_agent.car_length
        self.situation['cutting_off']['spawn_dist'] = self.respawn_distance
        spawn_mul = np.random.uniform(2.0, 3.0)
        self.situation['nominal']['spawn_dist'] = self.situation['cutting_off']['spawn_dist'] + \
            spawn_mul * ((nominal_car_len + cutting_off_car_len) / 2.)
        self.situation['cutting_off']['cutting_off_dist'] = np.random.uniform(1.5, spawn_mul) * \
            ((nominal_car_len + cutting_off_car_len) / 2.) * np.random.choice([-1, 1], p=[0.2,0.8])
        if self.cutoff_at_reset_prob is not None:
            self.sampled_perform_cutoff = np.random.choice([True, False], \
                p=[self.cutoff_at_reset_prob,1-self.cutoff_at_reset_prob])
        else:
            self.sampled_perform_cutoff = False

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
        self.extra_obs = np.array([float(False), float(False), float(False)])

        # reset road
        self.reset_scene_state()

        # counter for cutting off
        self.cutting_off_cnt = 0
        if self.give_pass_reward_immediately:
            self.passed_car = []

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
                if self.cutoff_immediately:
                    if self.cutoff_at_reset_prob is not None:
                        perform_cutting_off = self.sampled_perform_cutoff
                    else:
                        perform_cutting_off = True
                else:
                    perform_cutting_off = (dist_to_nominal <= self.situation['cutting_off']['cutting_off_dist']) \
                        and (self.situation['cutting_off']['cutting_off_dist'] > 0)

                pose_to_ref = self.compute_relative_transform(self.ref_agent.ego_dynamics, agent.ego_dynamics)
                yield_to_ref = pose_to_ref[1] > (-agent.car_length / 2.)
                commit_to_cutting_off = (self.cutting_off_cnt * speed) > 50

                self.extra_obs = np.array([float(perform_cutting_off), float(yield_to_ref), float(commit_to_cutting_off)])

                speed = self.situation['cutting_off']['base_speed']
                too_close = dist_to_nominal <= 1.2 * (agent.car_length + nominal_agent.car_length) / 2.
                if perform_cutting_off:
                    lookahead_dist = 5
                    dt = 1 / 30.
                    Kp = 3

                    # get road vectors
                    road_in_agent, _ = self.get_scene_state(agent.ego_dynamics, False)
                    road_in_agent = road_in_agent[road_in_agent[:,1] > 0] # drop road behind

                    if road_in_agent.shape[0] > 0:
                        # get target xy; use lookahead distance and apply later shift
                        dist = np.linalg.norm(road_in_agent, axis=1)
                        tgt_idx = np.argmin(np.abs(dist - lookahead_dist))
                        dx, dy, dtheta = road_in_agent[tgt_idx]

                        if yield_to_ref and not commit_to_cutting_off: # back to original trajectory
                            lat_shift = self.situation['cutting_off']['lat_shift']
                            if too_close: # slow down when too close to the front car
                                speed = self.situation['nominal']['base_speed']
                        else:
                            left_or_right = -1 if self.situation['nominal']['lat_shift'] > 0 else 1
                            lat_shift = left_or_right * agent.car_width / 2.

                            self.cutting_off_cnt += 1
                        dx += lat_shift * np.cos(dtheta)
                        dy += lat_shift * np.sin(dtheta)
                        
                        # compute curvature
                        arc_len = speed * dt
                        curvature = (Kp * np.arctan2(-dx, dy) * dt) / arc_len
                        curvature = np.clip(curvature, self.lower_curvature_bound, self.upper_curvature_bound)
                    else: # cutting off agent out-of-view from road
                        current_timestamp = agent.get_current_timestamp()
                        curvature = agent.trace.f_curvature(current_timestamp)
                else: # follow nominal trajectory
                    current_timestamp = agent.get_current_timestamp()
                    curvature = agent.trace.f_curvature(current_timestamp)
                    if too_close: # slow down when too close to the front car
                        speed = self.situation['nominal']['base_speed']

                    self.cutting_off_cnt = 0
                action[agent_id] = np.array([curvature, speed])
            else:
                raise ValueError('Invalid agent ID {}'.format(agent_id))
        
        # step environment
        observation, reward, done, info = map(self.wrap_data, super().step(action))
        self.observation_for_render = self.wrap_data(self.observation_for_render)

        # define reward and terminal condition (passing nominal agent and back to lane)
        other_agents = [_a for _i, _a in enumerate(self.world.agents) if _i != self.ref_agent_idx]
        passed = [self.check_agent_pass_other(self.ref_agent, _a) for _a in other_agents]
        in_lane_center = self.check_agent_in_lane_center(self.ref_agent)

        nominal_agent_idx = self.agent_ids.index(self.special_agent_ids['nominal'])
        cutting_off_agent_idx = self.agent_ids.index(self.special_agent_ids['cutting_off'])
        passed_nominal = passed[nominal_agent_idx-1]
        passed_cutting_off = passed[cutting_off_agent_idx-1]
        success = in_lane_center and passed_nominal
        info[self.ref_agent_id]['success'] = success
        info[self.ref_agent_id]['passed_cars'] = np.sum(passed)
        done[self.ref_agent_id] = success or done[self.ref_agent_id]
        if self.give_pass_reward_immediately:
            reward[self.ref_agent_id] = 0
            if passed_nominal and 'nominal' not in self.passed_car:
                self.passed_car.append('nominal')
                reward[self.ref_agent_id] += 1
            if passed_cutting_off and 'cutting_off' not in self.passed_car and self.n_passed_reward:
                self.passed_car.append('cutting_off')
                reward[self.ref_agent_id] += 1
            if success:
                reward[self.ref_agent_id] += 10
        else:
            reward[self.ref_agent_id] = np.sum(passed) if passed_nominal and in_lane_center and done[self.ref_agent_id] else 0
            if not self.n_passed_reward:
                reward[self.ref_agent_id] = float(passed_nominal) if done[self.ref_agent_id] else 0
        if self.car_following_bonus and (perform_cutting_off and not yield_to_ref or commit_to_cutting_off):
            in_range_mul = [1.1, 1.5]
            cutting_off_agent = self.world.agents[cutting_off_agent_idx]
            dist_to_other = np.linalg.norm(cutting_off_agent.ego_dynamics.numpy()[:2]-self.ref_agent.ego_dynamics.numpy()[:2])
            min_dist = (cutting_off_agent.car_length + self.ref_agent.car_length) / 2.
            in_range = dist_to_other < (min_dist * in_range_mul[1]) and \
                dist_to_other > (min_dist * in_range_mul[0]) and not passed_cutting_off
            bonus = 1 if in_range else 0.
            reward[self.ref_agent_id] += self.car_following_bonus * bonus

        # terminate episode if too far away behind
        # origin_dist = self.ref_agent.trace.f_distance(self.ref_agent.first_time)
        # dist = self.ref_agent.trace.f_distance(self.ref_agent.get_current_timestamp()) - origin_dist
        fail_to_catch_up = []
        for other_agent in other_agents:
            # other_dist = other_agent.trace.f_distance(other_agent.get_current_timestamp()) - origin_dist
            # too_far_behind = (other_dist - dist) > (10 * (other_agent.car_length + self.ref_agent.car_length) / 2.)
            dist_to_other = np.linalg.norm(self.ref_agent.ego_dynamics.numpy()[:2] - other_agent.ego_dynamics.numpy()[:2])
            too_far_behind = dist_to_other > (10 * (other_agent.car_length + self.ref_agent.car_length) / 2.)
            fail_to_catch_up.append(too_far_behind)
        done[self.ref_agent_id] = done[self.ref_agent_id] or np.any(fail_to_catch_up)

        # reward to track target speed
        if self.target_velocity is not None:
            ref_agent_speed = info[self.ref_agent_id]['model_velocity']
            velo_rew = 1 - (self.target_velocity - ref_agent_speed) / self.target_velocity
            velo_rew = np.clip(velo_rew, 0., 1.) * 0.001
            reward[self.ref_agent_id] += velo_rew

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
        '--target-velocity',
        default=None,
        type=float,
        help='Target velocity.')
    args = parser.parse_args()

    # initialize simulator
    env = CuttingOff(args.trace_paths, args.mesh_dir, init_agent_range=[6,12], 
        respawn_distance=10, target_velocity=args.target_velocity)
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