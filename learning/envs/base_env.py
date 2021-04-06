import os
import numpy as np
import gym
import cv2
from shapely.geometry import box as Box
from shapely import affinity
from ray.rllib.env.multi_agent_env import MultiAgentEnv

import vista
from .mesh_lib import MeshLib


class BaseEnv(gym.Env, MultiAgentEnv):
    lower_curvature_bound = -1 / 3.
    upper_curvature_bound = 1 / 3.
    metadata = {
        'render.modes': ['rgb_array'],
        'video.frames_per_second': 10
    }

    def __init__(self, trace_paths, n_agents=1, mesh_dir=None, 
                 collision_overlap_threshold=0.2,
                 init_agent_range=[8, 20]):
        trace_paths = [os.path.abspath(os.path.expanduser(tp)) for tp in trace_paths]
        self.world = vista.World(trace_paths)
        self.ref_agent_idx = 0
        for i in range(n_agents):
            agent = self.world.spawn_agent()
            self.agent_sensors_setup(i)
        self.n_agents = len(self.world.agents)
        self.agent_ids = ['agent_{}'.format(i) for i in range(self.n_agents)]
        self.ref_agent = self.world.agents[self.ref_agent_idx]
        self.ref_agent_id = self.agent_ids[self.ref_agent_idx]

        self.collision_overlap_threshold = collision_overlap_threshold
        self.init_agent_range = init_agent_range
        self.perturb_heading_in_random_init = True # set False for car following nominal traj 

        if self.n_agents > 1:
            assert mesh_dir is not None, "Specify mesh_dir if n_agents > 1"
            self.mesh_lib = MeshLib(mesh_dir)

        self.crash_to_others = [False] * self.n_agents

        # NOTE: only support the same observation space across all agents now
        cam = self.world.agents[self.ref_agent_idx].sensors[0].camera
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(cam.get_height(), cam.get_width(), 3),
            dtype=np.uint8)

        # TODO: include velocity/acceleration
        self.action_space = gym.spaces.Box(
                low=np.array([self.lower_curvature_bound]),
                high=np.array([self.upper_curvature_bound]),
                shape=(1,),
                dtype=np.float64)

        # NOTE: check how this affects learning
        self.reward_range = [0., 100.]

        self.controllable_agents = {k:v for k, v in zip(self.agent_ids, self.world.agents)}

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
                self.random_init_agent_in_the_front(agent, *self.init_agent_range, self.ref_agent.human_dynamics)
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
        observation = {k: v for k, v in zip(self.agent_ids, observation)}
        self.observation_for_render = observation

        return observation

    def step(self, action):
        # update agents' dynamics (take action in the environment)
        observation, reward, done, info = dict(), dict(), dict(), dict()
        next_valid_timestamp_list = []
        for agent_id, agent in zip(self.agent_ids, self.world.agents):
            act = action[agent_id]
            rew, d, info_, next_valid_timestamp = agent.step_dynamics(act)
            reward[agent_id] = rew
            done[agent_id] = d
            info[agent_id] = info_
            next_valid_timestamp_list.append(next_valid_timestamp)
        self.info_for_render = info
        # get agents' sensory measurement
        for i, agent_id in enumerate(self.agent_ids):
            agent = self.world.agents[i]
            next_valid_timestamp = next_valid_timestamp_list[i]
            other_agents = {_ai: self.world.agents[_ai] for _ai in range(self.n_agents) if _ai != i}
            other_agents = self.convert_to_scene_node(agent, other_agents)
            obs = agent.step_sensors(next_valid_timestamp, other_agents=other_agents)
            # NOTE: only support one sensor now
            if len(agent.sensors) == 1:
                observation[agent_id] = obs[agent.sensors[0].id]
            else:
                observation[agent_id] = None
        self.observation_for_render = observation
        # check agents' collision
        polys = [self.agent2poly(a, self.ref_agent.human_dynamics) for a in self.world.agents]
        self.crash_to_others = self.check_collision(polys)
        done = {k: v or c for (k, v), c in zip(done.items(), self.crash_to_others)}
        done['__all__'] = np.any(list(done.values()))
        
        return observation, reward, done, info

    def render(self, mode='rgb_array'):
        obs_show = []
        for i, obs in enumerate(self.observation_for_render.values()):
            if self.crash_to_others[i]:
                text = 'Crash to others'
            elif self.world.agents[i].isCrashed:
                text = 'Exceed Max Trans/Rot'
            else:
                text = 'Running'
            img = cv2.putText(obs, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2).get()
            obs_show.append(img)
        img = np.concatenate(obs_show, axis=1)
        img = img[:,:,::-1]
        return img

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

    def random_init_agent_in_the_front(self, agent, min_dist, max_dist, ref_dynamics):
        """ Randomly initialize agent in the front (w.r.t. the first/reference 
            agent, which is initialized with zeros) within the range of min and 
            max distance. """
        # sample distance, delta heading, and lateral shift
        dist = np.random.uniform(min_dist, max_dist)

        if self.perturb_heading_in_random_init:
            dtheta = np.random.uniform(-0.05, 0.05)
        else:
            dtheta = 0

        lat_shift = np.random.choice([-1, 1]) * np.random.uniform(agent.car_width / 2., agent.car_width)

        # place agent
        self.place_agent(agent, ref_dynamics, dist, dtheta, lat_shift)

    def place_agent(self, agent, ref_dynamics, dist, dtheta, lat_shift):
        # find the closest frame/index with the sampled distance
        index = agent.current_frame_index
        time = agent.get_timestamp(index)
        dist_match = False
        while not dist_match and not agent.trace_done:
            next_time = agent.get_timestamp(index)
            agent.human_dynamics.step(
                curvature=agent.trace.f_curvature(time),
                velocity=agent.trace.f_speed(time),
                delta_t=next_time-time
            )
            time = next_time
            rel_state = self.compute_relative_transform(agent.human_dynamics, ref_dynamics)
            dist_match = (np.linalg.norm(rel_state[:2]) - dist) >= 0
            index += 1
        agent.current_frame_index = index - 1

        # shift in lateral direction and rotate agent's heading
        agent.ego_dynamics.theta_state = agent.human_dynamics.theta_state + dtheta

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
        poly = affinity.rotate(poly, np.degrees(theta))

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

    def convert_to_scene_node(self, ego_agent, other_agents):
        other_agents_nodes = []
        for i, agent in other_agents.items():
            # compute relative pose to ego agent
            trans_x, trans_y, theta = self.compute_relative_transform( \
                agent.ego_dynamics, ego_agent.human_dynamics)
            rot = np.array([0, 1, 0, theta])
            rot = rot / np.linalg.norm(rot) # unit vector for quaternion
            trans = np.array([trans_x, 0, trans_y])
            # convert to scene node
            agent_node = self.mesh_lib.get_mesh_node(i, trans, rot)
            other_agents_nodes.append(agent_node)

        return other_agents_nodes

    def agent_sensors_setup(self, agent_i):
        agent = self.world.agents[agent_i]
        camera = agent.spawn_camera()

    def check_agent_in_lane_center(self, agent):
        dx = agent.relative_state.translation_x 
        dy = agent.relative_state.translation_y
        dtheta = agent.relative_state.theta
        human_theta = agent.human_dynamics.numpy()[2]
        d = np.sqrt(dx**2 + dy**2)
        lat_shift = d * np.cos(human_theta)
        in_lane_center = (np.abs(dtheta) <= 0.05) and (np.abs(lat_shift) <= (agent.car_width / 8))
        return in_lane_center

    def check_agent_pass_other(self, agent, other_agent):
        origin_dist = agent.trace.f_distance(agent.first_time)
        dist = agent.trace.f_distance(agent.get_current_timestamp()) - origin_dist
        other_dist = other_agent.trace.f_distance(other_agent.get_current_timestamp()) - origin_dist
        passed = (dist - other_dist) > ((other_agent.car_length + agent.car_length) / 2.)
        return passed

    def reset_mesh_lib(self):
        self.mesh_lib.reset(self.n_agents)
        # assign car width and length based on mesh size
        for i, agent in enumerate(self.world.agents):
            agent.car_width = self.mesh_lib.agents_meshes_dim[i][0]
            agent.car_length = self.mesh_lib.agents_meshes_dim[i][1]

    def init_scene_state(self, road_buffer_size):
        self.road_buffer_size = road_buffer_size # unit is frame
        self.road = deque(maxlen=self.road_buffer_size)
        self.road_frame_index = deque(maxlen=self.road_buffer_size)

    def get_scene_state(self):
        # update road (in global coordinate)
        while self.road_frame_index[-1] < (self.ref_agent.current_frame_index + self.road_buffer_size / 2):
            current_timestamp = self.get_timestamp_readonly(self.ref_agent, self.road_frame_index[-1])
            self.road_frame_index.append(self.road_frame_index[-1] + 1)
            next_timestamp = self.get_timestamp_readonly(self.ref_agent, self.road_frame_index[-1])
            self.road_dynamics.step(curvature=self.ref_agent.trace.f_curvature(current_timestamp),
                                    velocity=self.ref_agent.trace.f_speed(current_timestamp),
                                    delta_t=next_timestamp - current_timestamp)
            current_timestamp = next_timestamp
            self.road.append(self.road_dynamics.numpy()[:2])

        # update road in birds eye map (in reference agent coordinate)
        ref_x, ref_y, ref_theta = self.ref_agent.human_dynamics.numpy()
        road_in_ref = np.array(self.road) - np.array([ref_x, ref_y])
        c, s = np.cos(ref_theta), np.sin(ref_theta)
        R_T = np.array([[c, -s], [s, c]])
        road_in_ref = np.matmul(road_in_ref, R_T)

        # update agent in birds eye map (in reference agent coordinate)
        agent_xytheta_in_ref = self.compute_relative_transform(
            self.ref_agent.ego_dynamics, self.ref_agent.human_dynamics)

        # get scene state
        aug_road_in_ref = np.concatenate([np.zeros(\
            (self.road_buffer_size-road_in_ref.shape[0],2)), road_in_ref])
        scene_state = np.concatenate([aug_road_in_ref.reshape((-1,)), agent_xytheta_in_ref])

        return scene_state

    def reset_scene_state(self):
        self.road_frame_index.clear()
        self.road_frame_index.append(self.ref_agent.current_frame_index)
        self.road.clear()
        self.road.append(self.ref_agent.human_dynamics.numpy()[:2])
        self.road_dynamics = self.ref_agent.human_dynamics.copy()

    def get_timestamp_readonly(self, agent, index=0, current=False):
        index = agent.current_frame_index if current else index
        index = min(len(agent.trace.syncedLabeledTimestamps[
                agent.current_segment_index]) - 1, index)
        return agent.trace.syncedLabeledTimestamps[
            agent.current_segment_index][index]

    def close(self):
        for agent in self.world.agents:
            agent.viewer = None
            for sensor in agent.sensors:
                if hasattr(sensor, 'stream'):
                    sensor.stream.close()


if __name__ == "__main__":
    import argparse
    from .wrappers import MultiAgentMonitor

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
    parser.add_argument(
        '--mesh-dir',
        type=str,
        default=None,
        help='Directory of agents\' meshes.')
    args = parser.parse_args()

    # initialize simulator
    env = BaseEnv(args.trace_paths, args.n_agents, args.mesh_dir)
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
