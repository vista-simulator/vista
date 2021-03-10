import numpy as np
import gym
from shapely.geometry import box as Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv

import vista


class BaseEnv(MultiAgentEnv):
    def __init__(self, trace_paths, n_agents=1):
        self.world = vista.World(trace_paths)
        for i in range(n_agents):
            agent = self.world.spawn_agent()
            camera = agent.spawn_camera()
        self.n_agents = len(self.world.agents)
        self.ref_agent_idx = 0
        self.ref_agent = self.world.agents[self.ref_agent_idx]
        self.agent_ids = ['agent_{}'.format(i) for i in range(self.n_agents)]

        self.collision_overlap_threshold = 0.2

        # TODO: add action space

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
            while not collision_free:
                self.reset_agent(agent, ref_agent=self.ref_agent)
                self.random_init_agent_in_the_front(agent, 8, 20)
                self.update_trace_and_first_time(agent)

                poly = self.agent2poly(agent, self.ref_agent.human_dynamics)
                collision_free = not np.any(self.check_collision(polys + [poly]))
            polys.append(poly)

        # reset sensors (should go after agent dynamics reset, which affects video stream reset)
        for agent in self.world.agents:
            for sensor in agent.sensors:
                sensor.reset()

        # get sensor measurement
        observations = []
        for agent in self.world.agents:
            obs = agent.sensors[0].capture(agent.first_time) # NOTE: only support one sensor now
            observations.append(obs)

        # wrap data
        observations = self.wrap_data(observations)

        return observations

    def step(self, action):
        raise NotImplementedError

    def render(self, mode='rgb_array'):
        raise NotImplementedError

    def wrap_data(self, data):
        return {k: v for k, v in zip(self.agent_ids, data)}

    def reset_agent(self, agent, sync_with_ref=True, ref_agent=None):
        agent.relative_state.reset()
        agent.ego_dynamics.reset()
        agent.human_dynamics.reset()

        if sync_with_ref: # for non-reference agent, sync with reference
            agent.current_trace_index = ref_agent.current_trace_index
            agent.current_segment_index = ref_agent.current_segment_index
            agent.current_frame_index = ref_agent.current_frame_index
        else: # for reference agent, sample in world
            (agent.current_trace_index, agent.current_segment_index, \
                agent.current_frame_index) = agent.world.sample_new_location()

        self.update_trace_and_first_time(agent)

        agent.trace_done = False
        agent.isCrashed = False

    def update_trace_and_first_time(self, agent):
        agent.trace = agent.world.traces[agent.current_trace_index]
        agent.first_time = agent.trace.masterClock.get_time_from_frame_num(
            agent.trace.which_camera,
            agent.trace.syncedLabeledFrames[agent.current_trace_index][
                agent.trace.which_camera][agent.current_frame_index])  # MODIFIED

    def random_init_agent_in_the_front(self, agent, min_dist, max_dist):
        """ Randomly initialize agent in the front (w.r.t. the first/reference 
            agent, which is initialized with zeros) within the range of min and 
            max distance. """
        # sample between min and max distance
        dist = np.random.uniform(min_dist, max_dist)

        # find the closest frame/index with the sampled distance
        index = agent.current_frame_index
        time = agent.get_timestamp(index)
        dist_match = False
        while not dist_match:
            next_time = agent.get_timestamp(index)
            agent.human_dynamics.step(
                curvature=agent.trace.f_curvature(time),
                velocity=agent.trace.f_speed(time),
                delta_t=next_time-time
            )
            time = next_time
            dist_match = (np.linalg.norm(agent.human_dynamics.numpy()[:2]) - dist) >= 0
            index += 1
        agent.current_frame_index = index - 1

        # randomly shift in lateral direction and rotate agent's heading
        dtheta = np.random.uniform(-0.05, 0.05)
        agent.ego_dynamics.theta_state = agent.human_dynamics.theta_state + dtheta

        lat_bound = (agent.trace.road_width - agent.car_width) / 2.
        lat_shift = np.random.uniform(-lat_bound, lat_bound)
        dx = lat_shift * np.cos(agent.ego_dynamics.theta_state)
        dy = lat_shift * np.sin(agent.ego_dynamics.theta_state)
        agent.ego_dynamics.x_state = agent.human_dynamics.x_state + dx
        agent.ego_dynamics.y_state = agent.human_dynamics.y_state + dy

        # update relative transform
        translation_x, translation_y, theta = agent.compute_relative_transform()
        agent.relative_state.update(translation_x, translation_y, theta)

    def agent2poly(self, agent, ref_dynamics=None):
        """ Convert agent to polygon w.r.t. reference dynamics. """
        ref_dynamics = agent.human_dynamics if ref_dynamics is None else ref_dynamics
        x, y, theta = self.compute_relative_transform(agent.ego_dynamics, ref_dynamics)
        car_length = agent.car_length
        car_width = agent.car_width
        poly = Box(x-car_width/2., y-car_length/2., x+car_width/2., y+car_length/2.)

        return poly

    def check_collision(self, polys):
        """ Given a set of polygons, check if there is collision. """
        n_polys = len(polys)
        crash = [False] * n_polys
        for i in range(n_polys):
            for j in range(i+1, n_polys):
                intersect = polys[i].intersection(polys[j])
                overlap_ratio = intersect.area / polys[i].area
                crash[i] = crash[i] or (overlap_ratio >= self.collision_overlap_threshold)
        return crash

    def compute_relative_transform(self, dynamics, ref_dynamics):
        """ Relative x, y, and yaw w.r.t. reference dynamics. """
        x_state, y_state, theta_state = dynamics.numpy()
        ref_x_state, ref_y_state, ref_theta_state = ref_dynamics.numpy()

        c = np.cos(ref_theta_state)
        s = np.sin(ref_theta_state)
        R_2 = np.array([[c, -s], [s, c]])
        xy_global_centered = np.array([[x_state - ref_x_state],
                                        [ref_y_state - y_state]])
        [[translation_x], [translation_y]] = np.matmul(R_2, xy_global_centered)
        translation_y *= -1  # negate the longitudinal translation (due to VS setup)

        theta = theta_state - ref_theta_state

        return translation_x, translation_y, theta


if __name__ == "__main__":
    import os
    import argparse

    # parse argument
    parser = argparse.ArgumentParser(description='Run wrapper test.')
    parser.add_argument(
        '--trace-paths',
        type=str,
        nargs='+',
        help='Paths to the traces to use for simulation.')
    parser.add_argument(
        '--n-agents',
        type=int,
        default=1,
        help='Number of agents in the scene.')
    args = parser.parse_args()

    # initialize simulator
    env = BaseEnv(args.trace_paths, args.n_agents)

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
            ep_steps += 1
        print('[{}th episodes] {} steps {} reward'.format(ep, ep_steps, ep_rew))
