import gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from .base_env import BaseEnv


class CuttingOff(BaseEnv, MultiAgentEnv):
    def __init__(self, trace_paths, mesh_dir=None, speed_scale_range=[0.0, 0.8], **kwargs):
        super(CuttingOff, self).__init__(trace_paths, n_agents=3, 
            mesh_dir=mesh_dir, **kwargs)

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

        # reset non-reference agent
        polys = [self.agent2poly(self.ref_agent)]
        for i, agent in enumerate(self.world.agents):
            # skip reference agent
            if i == self.ref_agent_idx:
                continue            

            # randomly init non-reference agents in the front of reference agent
            collision_free = False
            max_resample_tries = 10
            resample_tries = 0
            while not collision_free and resample_tries < max_resample_tries:
                self.reset_agent(agent, ref_agent=self.ref_agent)
                if self.agent_ids[i] == self.special_agent_ids['nominal']:
                    raise NotImplementedError # TODO: define spawn_dist and lat_shift
                elif self.agent_ids[i] == self.special_agent_ids['nominal']:
                    raise NotImplementedError # TODO: define spawn_dist and lat_shift
                else:
                    raise ValueError('Invalid agent ID {}'.format(self.agent_ids[i]))
                self.place_agent(agent, self.ref_agent.human_dynamics, spawn_dist, 0, lat_shift)
                self.update_trace_and_first_time(agent)

                poly = self.agent2poly(agent, self.ref_agent.human_dynamics)
                collision_free = not np.any(self.check_collision(polys + [poly]))
                resample_tries += 1
            polys.append(poly)

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
        observations = self.wrap_data(observations)
        self.observation_for_render = self.wrap_data(self.observation_for_render) # for render

        # sample constant speed for other agents
        self.constant_speed['cutting_off'] = np.random.uniform(1., 6.) # slower than ego-agent but still moving
        self.constant_speed['nominal'] = np.random.uniform(0., # slower than cutting-off agent
            0.9 * self.constant_speed['cutting_off'])

        return observations

    def step(self, action):
        # augment action
        for agent_id, agent in zip(self.agent_ids, self.world.agents):
            if agent_id == self.ref_agent_id:
                continue
            elif agent_id == self.special_agent_ids['nominal']:
                # nominal curvature but slower speed
                current_timestamp = agent.get_current_timestamp()
                human_curvature = agent.trace.f_curvature(current_timestamp)
                speed = self.constant_speed['nominal']
                action[agent_id] = np.array([human_curvature, speed])
            elif agent_id == self.special_agent_ids['cutting_off']:
                raise NotImplementedError # TODO: define curvature for agent cutting off
                speed = self.constant_speed['cutting_off']
                action[agent_id] = np.array([curvature, speed])
            else:
                raise ValueError('Invalid agent ID {}'.format(agent_id))
        # step environment
        observation, reward, done, info = map(self.wrap_data, super().step(action))
        self.observation_for_render = self.wrap_data(self.observation_for_render)
        # define reward and terminal condition (passing nominal agent and back to lane)
        other_agents = [_a for _i, _a in enumerate(self.world.agents) if _i == self.special_agent_ids['nominal']]
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
    ray.init(
        local_mode=True,
        _temp_dir=os.path.expanduser('~/tmp'),
        include_dashboard=False)
    env = PlacingObstacle(args.trace_paths, args.mesh_dir, 
        args.task_mode, args.model_path, init_agent_range=[6,12])
    #env = MultiAgentMonitor(env, os.path.expanduser('~/tmp/monitor'), video_callable=lambda x: True, force=True)

    # run
    for ep in range(10):
        done = False
        obs = env.reset()
        ep_rew = 0
        ep_steps = 0
        while not done:
            act = dict()
            for k in [env.env_agent_id]:
                act[k] = env.action_space.sample()
            obs, rew, done, info = env.step(act)
            ep_rew += np.mean(list(rew.values()))
            done = np.any(list(done.values()))
            ep_steps += 1
        print('[{}th episodes] {} steps {} reward'.format(ep, ep_steps, ep_rew))