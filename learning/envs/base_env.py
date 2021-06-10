import os
from itertools import combinations
import numpy as np
import gym
import cv2
from collections import deque
from shapely.geometry import box as Box
from shapely import affinity
from ray.rllib.env.multi_agent_env import MultiAgentEnv

import vista
from .mesh_lib import MeshLib


class BaseEnv(gym.Env, MultiAgentEnv):
    lower_curvature_bound = -0.07
    upper_curvature_bound = 0.07
    lower_velocity_bound = 0.
    upper_velocity_bound = 15.
    metadata = {
        'render.modes': ['rgb_array'],
        'video.frames_per_second': 10
    }
    drop_obs_space_def = False
    camera_offset = [0., 1.7653, 0.2] # x, z (camera height = 1.7653), y 
    camera_rotation = [0., np.deg2rad(5), 0.04] # raw pitch yaw
    standard_car_area = 5 * 2 # car length x car width

    def __init__(self, trace_paths, n_agents=1, mesh_dir=None, 
                 collision_overlap_threshold=0.2, init_agent_range=[8, 20],
                 max_horizon=500, rigid_body_collision=False,
                 rigid_body_collision_coef=0.0, rigid_body_collision_repulsive_coef=0.9,
                 rendering_config=None, free_width_mul=0.5, max_rot_mul=0.1,
                 curv_reset_mode='default', dilate_ref_agent=[0., 0.]):
        trace_paths = [os.path.abspath(os.path.expanduser(tp)) for tp in trace_paths]
        self.world = vista.World(trace_paths)
        self.ref_agent_idx = 0
        for i in range(n_agents):
            agent = self.world.spawn_agent()
            self.agent_sensors_setup(i, rendering_config)
        self.n_agents = len(self.world.agents)
        self.agent_ids = ['agent_{}'.format(i) for i in range(self.n_agents)]
        self.ref_agent = self.world.agents[self.ref_agent_idx]
        self.ref_agent_id = self.agent_ids[self.ref_agent_idx]

        self.collision_overlap_threshold = collision_overlap_threshold
        self.init_agent_range = init_agent_range
        self.max_horizon = max_horizon
        self.rigid_body_collision = rigid_body_collision
        self.rigid_body_collision_coef = rigid_body_collision_coef
        self.rigid_body_collision_repulsive_coef = rigid_body_collision_repulsive_coef
        self.rendering_config = rendering_config
        self.free_width_mul = free_width_mul
        self.max_rot_mul = max_rot_mul
        self.soft_collision = 0.
        self.perturb_heading_in_random_init = True # set False for car following nominal traj 

        self.curv_reset_mode = curv_reset_mode
        for trace in self.world.traces:
            trace.reset_mode = self.curv_reset_mode

        self.dilate_ref_agent = dilate_ref_agent
        self.ref_agent.car_length += dilate_ref_agent[0]
        self.ref_agent.car_width += dilate_ref_agent[1]

        if self.n_agents > 1:
            assert mesh_dir is not None, "Specify mesh_dir if n_agents > 1"
            self.mesh_lib = MeshLib(mesh_dir)

        self.crash_to_others = [False] * self.n_agents

        # NOTE: only support the same observation space across all agents now
        if not self.drop_obs_space_def:
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
            # NOTE: only support one sensor now
            if len(agent.sensors) == 1:
                other_agents = {_ai: self.world.agents[_ai] for _ai in range(self.n_agents) if _ai != i}
                other_agents = self.convert_to_scene_node(agent, other_agents)
                obs = agent.sensors[0].capture(agent.first_time, other_agents=other_agents)
            else:
                obs = None
            observation.append(obs)

        # wrap data
        observation = {k: v for k, v in zip(self.agent_ids, observation)}
        self.observation_for_render = observation

        # for rigid body collision
        if self.rigid_body_collision or self.soft_collision > 0.:
            self.rigid_body_info = {
                'crash': np.zeros((self.n_agents, self.n_agents), dtype=bool),
                'overlap': np.zeros((self.n_agents, self.n_agents)),
                'cum_collide': np.zeros((self.n_agents,)),
            }

        # horizon count
        self.horizon_cnt = 0

        return observation

    def step(self, action):
        # modify action for rigid body collision
        if self.rigid_body_collision:
            # assert np.all([v.shape[0] == 2 for v in action.values()]), 'Need to include velocity for rigid body collision'
            # get pairwise projected speed
            agent_order = self.get_agent_order()
            proj_speed = 9999 * np.ones((self.n_agents, self.n_agents))
            for i in range(self.n_agents):
                for j in range(self.n_agents):
                    i_in_front_of_j = np.where(agent_order == i)[0][0] < np.where(agent_order == j)[0][0]
                    if (i == j) or i_in_front_of_j: # only slow down the behind car
                        proj_speed[i,j] = action[self.agent_ids[i]][1]
                    else:
                        crash = self.rigid_body_info['crash'][i,j]
                        if not crash:
                            continue
                        _, _, dtheta = self.compute_relative_transform(
                            self.world.agents[j].ego_dynamics, self.world.agents[i].ego_dynamics)
                        proj_speed[i,j] = np.cos(dtheta) * action[self.agent_ids[j]][1]
            # assign modified speed
            min_agent_speed = 9999. # consider modified speed of other agents
            for i, agent_idx in enumerate(agent_order):
                agent_id = self.agent_ids[agent_idx]
                crash_to_front_cars = np.any(self.rigid_body_info['crash'][agent_idx][agent_order][:i+1])
                if crash_to_front_cars:
                    action[agent_id][1] = min(proj_speed[agent_idx].min(), min_agent_speed)
                    action[agent_id][1] *= self.rigid_body_collision_repulsive_coef
                    min_agent_speed = min(min_agent_speed, action[agent_id][1])
                else:
                    min_agent_speed = 9999.
        # update agents' dynamics (take action in the environment)
        info = dict()
        for agent_id, agent in zip(self.agent_ids, self.world.agents):
            act = action[agent_id]
            agent.step_dynamics(act)
            info[agent_id] = {
                'model_curvature': agent.model_curvature,
                'model_velocity': agent.model_velocity,
                'human_curvature': agent.trace.f_curvature(agent.get_current_timestamp()),
                'model_angle': agent.curvature_to_steering(agent.model_curvature),
                'distance': agent.trace.f_distance(agent.timestamp) - \
                            agent.trace.f_distance(agent.first_time),
                'rotation': agent.relative_state.theta,
                'translation': agent.relative_state.translation_x,
                # NOTE: not saving trace index
                'segment_index': agent.current_segment_index,
                'frame_index': agent.current_frame_index,
            }
        self.info_for_render = info
        # get agents' sensory measurement
        observation = dict()
        for i, agent_id in enumerate(self.agent_ids):
            agent = self.world.agents[i]
            # NOTE: only support one sensor now
            if len(agent.sensors) == 1:
                other_agents = {_ai: self.world.agents[_ai] for _ai in range(self.n_agents) if _ai != i}
                other_agents = self.convert_to_scene_node(agent, other_agents)
                agent.step_sensors(other_agents=other_agents)
                obs = agent.observations
                observation[agent_id] = obs[agent.sensors[0].id]
            else:
                observation[agent_id] = None
        self.observation_for_render = observation
        # check agent off lane or exceed maximal rotation
        done, reward = dict(), dict()
        for agent_id in self.agent_ids:
            agent = self.world.agents[self.agent_ids.index(agent_id)]
            off_lane, max_rot = self.check_agent_off_lane_or_max_rot(agent)
            done[agent_id] = off_lane or max_rot or agent.trace_done
            info[agent_id]['off_lane'] = off_lane
            info[agent_id]['max_rot'] = max_rot
            info[agent_id]['trace_done'] = np.any([_a.trace_done for _a in self.world.agents])
            reward[agent_id] = 1 if not agent.isCrashed else 0 # default reward
        # check agents' collision
        polys = [self.agent2poly(a, self.ref_agent.human_dynamics) for a in self.world.agents]
        crash, overlap = self.check_collision(polys, return_overlap=True)
        self.crash_to_others, self.overlap_ratio = np.any(crash, axis=1), np.sum(overlap, axis=1) # legacy code
        if self.rigid_body_collision or self.soft_collision > 0.:
            self.rigid_body_info['crash'] = crash
            self.rigid_body_info['overlap'] = overlap
            for i, agent_id in enumerate(info.keys()):
                info[agent_id]['collide'] = np.any(self.rigid_body_info['crash'][i])
                self.rigid_body_info['cum_collide'][i] += float(info[agent_id]['collide'])
                info[agent_id]['cum_collide'] = self.rigid_body_info['cum_collide'][i]
                info[agent_id]['has_collided'] = info[agent_id]['cum_collide'] > 0
            # NOTE: don't end due to collision unless pass hard overlap threshold
        else:
            done = {k: v or c for (k, v), c in zip(done.items(), self.crash_to_others)}
            for i, agent_id in enumerate(info.keys()):
                info[agent_id]['has_collided'] = self.crash_to_others[i]

        self.horizon_cnt += 1
        if self.horizon_cnt >= self.max_horizon:
            done = {k: True for k, v in done.items()}
        done['__all__'] = np.any(list(done.values()))

        if done['__all__']:
            for i, agent_id in enumerate(info.keys()):
                info[agent_id]['done_out_of_lane_or_max_rot'] = self.world.agents[i].isCrashed
        
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
            agent.trace.syncedLabeledFrames[agent.current_segment_index][
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

    def check_collision(self, polys, return_overlap=False):
        """ Given a set of polygons, check if there is collision. """
        n_polys = len(polys)
        crash = np.zeros((n_polys, n_polys), dtype=bool)
        overlap = np.zeros((n_polys, n_polys))
        for i in range(n_polys):
            for j in range(n_polys):
                if j == i:
                    crash[i,j] = False
                    overlap[i,j] = 0.
                else:
                    intersect = polys[i].intersection(polys[j])
                    overlap_ratio = intersect.area / self.standard_car_area
                    crash[i,j] = overlap_ratio >= self.collision_overlap_threshold
                    overlap[i,j] = overlap_ratio
        if return_overlap:
            return crash, overlap
        else:
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
            theta = np.pi - theta
            rot = np.array([0, np.sin(theta/2.), 0, np.cos(theta/2.)])
            trans = np.array([trans_x, 0, trans_y]) 
            # compensate for camera position
            trans = trans - np.array(self.camera_offset)
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

    def agent_sensors_setup(self, agent_i, rendering_config):
        agent = self.world.agents[agent_i]
        camera = agent.spawn_camera(rendering_config)

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

    def get_agent_order(self):
        ref_agent = self.ref_agent
        origin_dist = ref_agent.trace.f_distance(ref_agent.first_time)
        dists = []
        for agent in self.world.agents:
            dist = agent.trace.f_distance(agent.get_current_timestamp()) - origin_dist
            dists.append(dist)
        return np.argsort(dists)[::-1] # front to behind

    def check_agent_off_lane_or_max_rot(self, agent):
        tx = agent.relative_state.translation_x
        theta = agent.relative_state.theta
        if agent == self.ref_agent: # don't use dilation while determining off-lane
            free_width = agent.trace.road_width - (agent.car_width - self.dilate_ref_agent[1])
        else:
            free_width = agent.trace.road_width - agent.car_width
        off_lane = abs(tx) > (free_width * self.free_width_mul)
        max_rot = abs(theta) > (np.pi * self.max_rot_mul)
        return off_lane, max_rot

    def reset_mesh_lib(self):
        self.mesh_lib.reset(self.n_agents)
        # assign car width and length based on mesh size
        for i, agent in enumerate(self.world.agents):
            if i != self.ref_agent_idx:
                agent.car_width = self.mesh_lib.agents_meshes_dim[i][0]
                agent.car_length = self.mesh_lib.agents_meshes_dim[i][1]

    def init_scene_state(self, road_buffer_size):
        self.road_buffer_size = road_buffer_size # unit is frame
        self.road = deque(maxlen=self.road_buffer_size)
        self.road_frame_index = deque(maxlen=self.road_buffer_size)

    def get_scene_state(self, ref_dynamics=None, agent=None, concat=True, update=True):
        # update road (in global coordinate)
        if update:
            while self.road_frame_index[-1] < (self.ref_agent.current_frame_index + self.road_buffer_size / 2):
                current_timestamp = self.get_timestamp_readonly(self.ref_agent, self.road_frame_index[-1])
                self.road_frame_index.append(self.road_frame_index[-1] + 1)
                next_timestamp = self.get_timestamp_readonly(self.ref_agent, self.road_frame_index[-1])
                self.road_dynamics.step(curvature=self.ref_agent.trace.f_curvature(current_timestamp),
                                        velocity=self.ref_agent.trace.f_speed(current_timestamp),
                                        delta_t=next_timestamp - current_timestamp)
                current_timestamp = next_timestamp
                self.road.append(self.road_dynamics.numpy())

        # update road in birds eye map (in reference agent coordinate)
        # NOTE: custom ref_dynamics is only used for road_in_ref; self.road still use ref_agent as reference
        ref_dynamics = self.ref_agent.human_dynamics if ref_dynamics is None else ref_dynamics
        ref_x, ref_y, ref_theta = ref_dynamics.numpy()
        road_in_ref = np.array(self.road)
        road_in_ref[:,:2] -= np.array([ref_x, ref_y])
        road_in_ref[:,2] -= ref_theta
        c, s = np.cos(ref_theta), np.sin(ref_theta)
        R_T = np.array([[c, -s], [s, c]])
        road_in_ref[:,:2] = np.matmul(road_in_ref[:,:2], R_T)

        # update agent in birds eye map (in reference agent coordinate)
        agent = self.ref_agent if agent is None else agent
        agent_xytheta_in_ref = self.compute_relative_transform(
            agent.ego_dynamics, ref_dynamics)

        # get scene state
        aug_road_in_ref = np.concatenate([np.zeros(\
            (self.road_buffer_size-road_in_ref[:,:2].shape[0],2)), road_in_ref[:,:2]]) # NOTE: drop theta state
        scene_state = np.concatenate([aug_road_in_ref.reshape((-1,)), agent_xytheta_in_ref])

        if concat:
            return scene_state
        else:
            return road_in_ref, agent_xytheta_in_ref

    def reset_scene_state(self):
        self.road_frame_index.clear()
        self.road_frame_index.append(self.ref_agent.current_frame_index)
        self.road.clear()
        self.road.append(self.ref_agent.human_dynamics.numpy().astype(np.float))
        self.road_dynamics = self.ref_agent.human_dynamics.copy()

    def get_timestamp_readonly(self, agent, index=0, current=False):
        index = agent.current_frame_index if current else index
        index = min(len(agent.trace.syncedLabeledTimestamps[
                agent.current_segment_index]) - 1, index)
        return agent.trace.syncedLabeledTimestamps[
            agent.current_segment_index][index]

    def obs_for_render(self, obs):
        # for monitor wrapper
        return obs

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
