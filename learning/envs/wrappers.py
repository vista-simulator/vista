import os
from collections import deque
import gym
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
from shapely.geometry import LineString
from descartes import PolygonPatch
import torch
import torchvision
from ray.rllib.env.multi_agent_env import MultiAgentEnv

import vista


class _MultiAgentMonitor(gym.Wrapper):
    def __init__(self, env):
        super(_MultiAgentMonitor, self).__init__(env)

        # get ids of agents with sensors
        self.agents_with_sensor = dict()
        for agent_id, agent in zip(self.agent_ids, self.world.agents):
            has_camera = len(agent.sensors) > 0 # NOTE: assume the only sensor is camera
            if has_camera:
                self.agents_with_sensor[agent_id] = agent

        # initialize data for plotting
        self.road_buffer_size = 200 # unit is frame
        self.road = deque(maxlen=self.road_buffer_size)
        self.road_frame_index = deque(maxlen=self.road_buffer_size)

        # initialize plot
        colors = list(cm.get_cmap('Set1').colors)
        rgba2rgb = lambda rgba: np.clip((1 - rgba[:3]) * rgba[3] + rgba[:3], 0., 1.)
        colors = [np.array(list(c) + [0.6]) for c in colors]
        colors = list(map(rgba2rgb, colors))
        self.road_color = list(cm.get_cmap('Dark2').colors)[-1]
        self.agent_colors = colors

        self.artists = dict()
        n_agents_with_sensor = len(self.agents_with_sensor)
        figsize = (6.4 * n_agents_with_sensor + 3.2, 4.8)
        self.fig = plt.figure(figsize=figsize)
        self.fig.subplots_adjust(left=0.01, right=0.98, bottom=0.02, top=0.90, wspace=0.03, hspace=0.03)
        self.fig.patch.set_facecolor('black') # use black background
        self.gs = self.fig.add_gridspec(5, 4 * n_agents_with_sensor + 2)

        # birds eye map
        self.ax_birdseye = self.fig.add_subplot(self.gs[:,-2:])
        self.ax_birdseye.set_facecolor('black')
        self.ax_birdseye.set_xticks([])
        self.ax_birdseye.set_yticks([])
        self.ax_birdseye.set_title('Top-down View', color='white', size=20, weight='bold')
        self.birdseye_map_size = (30, 20)

        # observation, speedometer and steering wheel
        resource_dir = os.path.join(os.path.dirname(vista.__file__), 'resources', 'img')
        self.img_steering_wheel = cv2.imread(os.path.join(resource_dir, 'mit.jpg'))[:,:,::-1]
        self.img_speedometer = cv2.imread(os.path.join(resource_dir, 'speed.jpg'))[:,:,::-1]

        self.ax_obs = dict()
        self.ax_car_states = dict()
        obs_space = self.render_observation_space if hasattr(self, 'render_observation_space')\
            else self.observation_space
        for i, agent_id in enumerate(self.agents_with_sensor.keys()):
            this_gs = self.gs[:4, 4*i:4*(i+1)]
            self.ax_obs[agent_id] = self.fig.add_subplot(this_gs)
            self.artists['im:{}'.format(agent_id)] = self.ax_obs[agent_id].imshow(
                self.fit_img_to_ax(self.ax_obs[agent_id], \
                    np.zeros(list(obs_space.shape[:2]) + [3], \
                        dtype=obs_space.dtype)))
            self.ax_obs[agent_id].set_xticks([])
            self.ax_obs[agent_id].set_yticks([])
            self.ax_obs[agent_id].set_title('Init', color='white', size=20, weight='bold')

            self.ax_car_states[agent_id] = self.fig.add_subplot(self.gs[-1,4*i:4*i+4])
            self.ax_car_states[agent_id].set_xticks([])
            self.ax_car_states[agent_id].set_yticks([])
            self.artists['im:car_states_{}'.format(agent_id)] = \
                self.ax_car_states[agent_id].imshow(self.fit_img_to_ax(self.ax_car_states[agent_id], \
                    self.cat_speedometer_steering_wheel(self.img_steering_wheel, self.img_speedometer)))

    def reset(self, **kwargs):
        # regular reset
        observation = super().reset(**kwargs)

        # initialize road deque
        self.road_frame_index.clear()
        self.road_frame_index.append(self.ref_agent.current_frame_index)
        self.road.clear()
        self.road.append(self.ref_agent.human_dynamics.numpy()[:2])
        self.road_dynamics = self.ref_agent.human_dynamics.copy()

        # set birds eye map size
        self.ax_birdseye.set_xlim(-self.birdseye_map_size[1]/2., self.birdseye_map_size[1]/2.)
        self.ax_birdseye.set_ylim(-self.birdseye_map_size[0]/2., self.birdseye_map_size[0]/2.)

        return observation

    def render(self, mode='rgb_array'):
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
        patch = LineString(road_in_ref).buffer(self.ref_agent.trace.road_width / 2.)
        patch = PolygonPatch(patch, fc=self.road_color, ec=self.road_color, zorder=1)
        self.update_patch(self.ax_birdseye, 'patch:road', patch)

        # update agent in birds eye map (in reference agent coordinate)
        for i, (agent_id, agent) in enumerate(zip(self.agent_ids, self.world.agents)):
            poly = self.agent2poly(agent, self.ref_agent.human_dynamics)
            patch = PolygonPatch(poly, fc=self.agent_colors[i], ec=self.agent_colors[i], zorder=2)
            self.update_patch(self.ax_birdseye, 'patch:{}'.format(agent_id), patch)
        
        # update observation
        for agent_id in self.agents_with_sensor.keys():
            i = self.agent_ids.index(agent_id)
            obs = self.observation_for_render[agent_id]
            if self.crash_to_others[i]:
                text = 'Crash Into Another Car'
            elif self.world.agents[i].isCrashed:
                text = 'Out-of-lane / Exceed Max Rotation'
            else:
                text = 'Running'
            self.ax_obs[agent_id].set_title(text, color='white', size=20, weight='bold')
            self.artists['im:{}'.format(agent_id)].set_data(self.fit_img_to_ax(
                self.ax_obs[agent_id], obs[:,:,-3:][:,:,::-1])) # handle stacked frames

        # add speed and steering wheel
        for agent_id, agent in self.agents_with_sensor.items():
            current_timestamp = self.get_timestamp_readonly(agent, current=True)
            if hasattr(self, 'info_for_render'):
                steering_angle = self.info_for_render[agent_id]['model_angle']
                speed = self.info_for_render[agent_id]['model_velocity']
            else:
                curvature = agent.trace.f_curvature(current_timestamp)
                steering_angle = agent.curvature_to_steering(curvature)
                speed = agent.trace.f_speed(current_timestamp)

            rows, cols = self.img_steering_wheel.shape[:2]
            M = cv2.getRotationMatrix2D((cols/2, rows/2), steering_angle, 1)
            img_steering_wheel_rotated = cv2.warpAffine(self.img_steering_wheel, M, (cols, rows))
            img_speedometer = self.img_speedometer.copy()
            self.plot_speedometer_pointer(img_speedometer, speed)
            self.artists['im:car_states_{}'.format(agent_id)].set_data(\
                self.fit_img_to_ax(self.artists['im:car_states_{}'.format(agent_id)],
                self.cat_speedometer_steering_wheel(img_steering_wheel_rotated, img_speedometer)))

        # convert to image
        img = self.fig2img(self.fig)

        return img

    def update_patch(self, ax, name, patch):
        if name in self.artists:
            self.artists[name].remove()
        ax.add_patch(patch)
        self.artists[name] = patch

    def fig2img(self, fig):
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:,:,:3]
        return img

    def plot_speedometer_pointer(self, image, speed):
        speedometer_max = 100.
        origin = (image.shape[1] // 2, image.shape[0])
        R = int(image.shape[0] * 0.8)
        angle = -np.pi * (speed / speedometer_max + 1)
        end = (origin[0] + int(R * np.cos(angle)), origin[1] - int(R * np.sin(angle)))
        cv2.line(image, tuple(origin), tuple(end), (255, 255, 255), image.shape[0] // 80)

    def cat_speedometer_steering_wheel(self, img_sw, img_sp):
        return np.concatenate([img_sw, np.zeros((\
            img_sw.shape[0], 200, 3), dtype=np.uint8), img_sp], axis=1)

    def fit_img_to_ax(self, ax, img):
        bbox = ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        w, h = bbox.width, bbox.height
        img_h, img_w = img.shape[:2]
        new_img_w = img_h * w / h
        new_img_h = img_w * h / w
        d_img_w = new_img_w - img_w
        d_img_h = new_img_h - img_h
        if d_img_h > 0:
            pad_img = np.zeros((int(d_img_h // 2), img_w, 3), dtype=np.uint8)
            new_img = np.concatenate([pad_img, img, pad_img], axis=0)
        elif d_img_w > 0:
            pad_img = np.zeros((img_h, int(d_img_w // 2), 3), dtype=np.uint8)
            new_img = np.concatenate([pad_img, img, pad_img], axis=1)
        else:
            raise ValueError('Something weird happened.')
        return new_img


class MultiAgentMonitor(gym.wrappers.Monitor, MultiAgentEnv):
    def __init__(self, env, directory, video_callable=None, force=False, resume=False,
                 write_upon_reset=False, uid=None, mode=None):
        self._env = env
        env = _MultiAgentMonitor(env)
        super(MultiAgentMonitor, self).__init__(env, directory, video_callable, 
            force, resume, write_upon_reset, uid, mode)

    def step(self, action):
        self._before_step(action)
        observation, reward, done, info = self.env.step(action)
        self._after_step(observation['agent_0'], reward['agent_0'], done['__all__'], info['agent_0'])

        return observation, reward, done, info
    
    def close(self):
        self._env.close() # to avoid max recursion in __del__


class PreprocessObservation(gym.ObservationWrapper, MultiAgentEnv):
    def __init__(self, env, fx=1.0, fy=1.0, custom_roi_crop=None, standardize=False, 
                 imagenet_normalize=False, random_gamma=None, color_jitter=None):
        super(PreprocessObservation, self).__init__(env)
        self.fx, self.fy = fx, fy
        self.standardize = standardize
        self.imagenet_normalize = imagenet_normalize
        self.random_gamma = random_gamma
        if self.random_gamma is not None:
            assert len(random_gamma) == 2, 'Random gamma requires 2 params: min_gamma and max_gamma'
        if color_jitter is not None:
            assert len(color_jitter) == 4, 'Color jitter requires 4 params: brightness / contrast / saturation / hue'
            self.color_jitter = torchvision.transforms.ColorJitter(
                brightness=color_jitter[0],
                contrast=color_jitter[1],
                saturation=color_jitter[2],
                hue=color_jitter[3]
            )
        self.roi = env.world.agents[0].sensors[0].camera.get_roi() \
            if custom_roi_crop is None else custom_roi_crop # NOTE: use sensor config from the first agent
        (i1, j1, i2, j2) = self.roi
        new_h, new_w = int((i2 - i1) * self.fy), int((j2 - j1) * self.fx)
        if self.standardize:
            low, high, dtype = -10., 10., np.float
        else:
            low, high, dtype = 0, 255, np.uint8
        cropped_obs_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=(new_h, new_w, 3),
            dtype=dtype
        )
        if isinstance(env.observation_space, gym.spaces.Tuple):
            # NOTE: assume the first observation is alway visual input of the agent
            self.observation_space = gym.spaces.Tuple([
                cropped_obs_space,
                *env.observation_space[1:]
            ])
        else:
            self.observation_space = cropped_obs_space
    
    def observation(self, observation):
        (i1, j1, i2, j2) = self.roi
        if isinstance(observation, dict):
            out = dict()
            for k, v in observation.items():
                obs = v[0] if isinstance(v, list) else v
                pp_obs = Image.fromarray(obs[:,:,::-1]) # PIL follows RGB
                if self.random_gamma:
                    gamma = np.random.uniform(*self.random_gamma)
                    pp_obs = torchvision.transforms.functional.adjust_gamma(pp_obs, gamma)
                if self.color_jitter is not None:
                    pp_obs = self.color_jitter(pp_obs)
                pp_obs = np.array(pp_obs)[:,:,::-1]

                pp_obs = cv2.resize(pp_obs[i1:i2, j1:j2], None, fx=self.fx, fy=self.fy)

                if self.imagenet_normalize:
                    pp_obs = self._imagenet_normalize(pp_obs)
                if self.standardize:
                    pp_obs = self._standardize(pp_obs)
                out[k] = [pp_obs] + v[1:] if isinstance(v, list) else pp_obs
    
                cropped_obs = cv2.resize(obs[i1:i2, j1:j2], None, fx=self.fx, fy=self.fy)
                self.observation_for_render[k] = cropped_obs
        elif isinstance(observation, list):
            raise NotImplementedError
            out = []
            for v in observation:
                out.append(cv2.resize(v[i1:i2, j1:j2], None, fx=self.fx, fy=self.fy))
        else:
            raise NotImplementedError
            out = cv2.resize(observation[i1:i2, j1:j2], None, fx=self.fx, fy=self.fy)
        return out

    def _imagenet_normalize(self, x):
        assert x.dtype == np.uint8
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        return (x/255. - mean) / std

    def _standardize(self, x):
        """ follow https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization """
        mean, stddev = x.mean(), x.std()
        adjusted_stddev = max(stddev, 1.0/np.sqrt(np.prod(x.shape)))
        return (x - mean) / adjusted_stddev

    def _random_gamma(self, x):
        """ follow https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/ """
        gamma = np.random.uniform(*self.random_gamma)
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(x, table)


class StackObservation(gym.ObservationWrapper, MultiAgentEnv):
    def __init__(self, env, frame_stack=5):
        super(StackObservation, self).__init__(env)
        self.frame_stack = frame_stack
        self.frame_deque = dict()
        if isinstance(env.observation_space, gym.spaces.Tuple):
            obs_space_list = []
            self.stack_obs_idcs = []
            for oi, obs_space in enumerate(env.observation_space):
                if len(obs_space.shape) == 3: # only stack image observation
                    ori_c = obs_space.shape[2]
                    obs_space_list.append(gym.spaces.Box(
                        low=np.dstack([obs_space.low] * frame_stack),
                        high=np.dstack([obs_space.high] * frame_stack),
                        dtype=obs_space.dtype,
                        shape=list(obs_space.shape[:2]) + [ori_c * frame_stack]
                    ))
                    self.stack_obs_idcs.append(oi)
                else:
                    obs_space_list.append(obs_space)
            self.observation_space = gym.spaces.Tuple(obs_space_list)

            for agent_id in self.controllable_agents.keys():
                self.frame_deque[agent_id] = [deque(maxlen=frame_stack) for oi in \
                    range(len(self.observation_space)) if oi in self.stack_obs_idcs]
        else:
            assert len(env.observation_space.shape) == 3
            ori_c = env.observation_space.shape[2]
            self.observation_space = gym.spaces.Box(
                low=np.dstack([env.observation_space.low] * frame_stack),
                high=np.dstack([env.observation_space.high] * frame_stack),
                dtype=env.observation_space.dtype,
                shape=list(env.observation_space.shape[:2]) + [ori_c * frame_stack]
            )

            for agent_id in self.controllable_agents.keys():
                self.frame_deque[agent_id] = deque(maxlen=frame_stack)

    def observation(self, observation):
        for k, obs in observation.items():
            if isinstance(self.observation_space, gym.spaces.Tuple):
                for oi, oobs in enumerate(obs):
                    if oi in self.stack_obs_idcs:
                        self.frame_deque[k][oi].append(oobs)
                while len(self.frame_deque[k][0]) < self.frame_deque[k][0].maxlen:
                    for oi, oobs in enumerate(obs):
                        if oi in self.stack_obs_idcs:
                            self.frame_deque[k][oi].append(oobs)
                for oi in range(len(observation[k])):
                    if oi in self.stack_obs_idcs:
                        observation[k][oi] = np.concatenate([v for v in self.frame_deque[k][oi]], axis=2)
            else:
                self.frame_deque[k].append(obs)
                while len(self.frame_deque[k]) < self.frame_deque[k].maxlen:
                    self.frame_deque[k].append(obs)
                observation[k] = np.concatenate([v for v in self.frame_deque[k]], axis=2)
        return observation


class ContinuousKinematic(gym.Wrapper, MultiAgentEnv):
    def __init__(self, env, d_curvature_bound=[-5.,5.], d_velocity_bound=[-15.,15.]):
        # NOTE: should be outter wrapper
        super(ContinuousKinematic, self).__init__(env)

        # define action space
        if self.env.action_space.shape[0] == 1:
            self.action_space = gym.spaces.Box(
                    low=np.array([d_curvature_bound[0]]),
                    high=np.array([d_curvature_bound[1]]),
                    shape=(1,),
                    dtype=np.float64)
        elif self.env.action_space.shape[0] == 2:
            self.action_space = gym.spaces.Box(
                    low=np.array([d_curvature_bound[0], d_velocity_bound[0]]),
                    high=np.array([d_curvature_bound[1], d_velocity_bound[1]]),
                    shape=(2,),
                    dtype=np.float64)
        else:
            raise ValueError('Action space of shape {} is not supported'.format(self.env.action_space.shape))
        self.d_curvature_bound = d_curvature_bound
        self.d_velocity_bound = d_velocity_bound

        # define augmented observation space
        vec_obs_space = gym.spaces.Box( # normalized to -1~1
            low=-2*np.ones((2,)), # NOTE: make sure pass boundary check
            high=2*np.ones((2,)),
            shape=(2,),
            dtype=np.float64
        )
        if isinstance(self.env.observation_space, gym.spaces.Tuple):
            self.observation_space = gym.spaces.Tuple(list(self.env.observation_space) + [vec_obs_space])
        else:
            self.observation_space = gym.spaces.Tuple([self.env.observation_space, vec_obs_space])
        self.render_observation_space = self.observation_space[0]

        # track vehicle state
        self.vehicle_state = {k: [None, None] for k in self.controllable_agents.keys()}
        self.delta_t = 1 / 30. # TODO: hardcoded

    def reset(self, **kwargs):
        # regular reset
        observation = super().reset(**kwargs)

        # augment vehicle state (didn't check if agent is controllable)
        for agent_id, obs in observation.items():
            agent_idx = self.agent_ids.index(agent_id)
            agent = self.world.agents[agent_idx]

            current_ts = self.get_timestamp_readonly(agent, current=True)
            self.vehicle_state[agent_id][0] = 0. 
            self.vehicle_state[agent_id][1] = agent.trace.f_speed(current_ts) # start with non-zero velocity

            veh_state_obs = self._standardize_veh_state(*self.vehicle_state[agent_id])

            if isinstance(obs, list):
                observation[agent_id] = obs + [veh_state_obs]
            else:
                observation[agent_id] = [obs, veh_state_obs]

        return observation

    def step(self, action):
        for agent_id, act in action.items():
            agent = self.controllable_agents[agent_id]
            action[agent_id] = self._integrator(act, agent_id)

        observation, reward, done, info = super().step(action)
        
        # augment vehicle state
        for agent_id, obs in observation.items():
            self.vehicle_state[agent_id][0] = info[agent_id]['model_curvature']
            self.vehicle_state[agent_id][1] = info[agent_id]['model_velocity']

            veh_state_obs = self._standardize_veh_state(*self.vehicle_state[agent_id])

            if isinstance(obs, list):
                observation[agent_id] = obs + [veh_state_obs]
            else:
                observation[agent_id] = [obs, veh_state_obs]

        return observation, reward, done, info

    def _integrator(self, act, agent_id):
        self.vehicle_state[agent_id] += act * self.delta_t
        self.vehicle_state[agent_id] = np.clip(
            self.vehicle_state[agent_id],
            np.array([self.lower_curvature_bound, self.lower_velocity_bound]),
            np.array([self.upper_curvature_bound, self.upper_velocity_bound])
        )
        return self.vehicle_state[agent_id]

    def _standardize_veh_state(self, curvature, velocity):
        def _standardize(_x, _lb, _ub):
            _midpt = (_lb + _ub) / 2.
            _norm = (_ub - _lb) / 2.
            return (_x - _midpt) / (_norm + 1e-8)
        curvature = _standardize(curvature, self.lower_curvature_bound, self.upper_curvature_bound)
        velocity = _standardize(velocity, self.lower_velocity_bound, self.upper_velocity_bound)
        return np.array([curvature, velocity])


class DistanceReward(gym.Wrapper, MultiAgentEnv):
    def __init__(self, env, reward_coef=1.0, scale_with_dist=True, cutoff_dist=None, sparse=False):
        super(DistanceReward, self).__init__(env)
        self.prev_distance = None
        self.reward_coef = reward_coef
        self.scale_with_dist = scale_with_dist
        self.cutoff_dist = cutoff_dist
        self.sparse = sparse
        self.step_cnt = None

    def reset(self, **kwargs):
        self.prev_distance = {k: 0. for k in self.controllable_agents.keys()}
        self.step_cnt = 0
        return super().reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = super().step(action)
        self.step_cnt += 1
        for k, v in info.items():
            if self.sparse and done['__all__']:
                if self.scale_with_dist:
                    distance = v['distance']
                else:
                    distance = 0.1 * self.step_cnt
                if self.cutoff_dist is not None:
                    distance = min(self.cutoff_dist, distance)
                if self.reward_coef in ['inf', np.inf]:
                    reward[k] = distance
                else:
                    reward[k] += self.reward_coef * distance
            else:
                if self.cutoff_dist is None or (self.cutoff_dist is not None and self.prev_distance[k] <= self.cutoff_dist):
                    delta_distance = v['distance'] - self.prev_distance[k]
                    if not self.scale_with_dist:
                        delta_distance = 0.1 if delta_distance > 0 else 0.
                    if self.reward_coef in ['inf', np.inf]:
                        reward[k] = delta_distance
                    else:
                        reward[k] += self.reward_coef * delta_distance
                    assert delta_distance >= 0 # sanity check
                    self.prev_distance[k] = v['distance']
        return observation, reward, done, info


class RandomPermuteAgent(gym.Wrapper, MultiAgentEnv):
    def __init__(self, env, permute_prob=0.0):
        super(RandomPermuteAgent, self).__init__(env)
        self.permute_prob = permute_prob

    def reset(self, **kwargs):
        self.do_permute = self.permute_prob > np.random.uniform(0., 1.)
        observation = super().reset(**kwargs)
        if self.do_permute:
            self.permuted_keys = np.random.permutation(list(observation.keys()))
            observation = self.permute_data(observation)
        return observation

    def step(self, action):
        if self.do_permute:
            action = self.permute_data(action)
        observation, reward, done, info = super().step(action)
        if self.do_permute:
            done_all = done.pop('__all__')
            observation, reward, done, info = map(self.permute_data, [observation, reward, done, info])
            done['__all__'] = done_all
        return observation, reward, done, info

    def permute_data(self, data):
        new_data = {k: v for k, v in zip(self.permuted_keys, list(data.values()))}
        return new_data


class BasicManeuverReward(gym.Wrapper, MultiAgentEnv):
    def __init__(self, env, center_coeff=0.01, jitter_coeff=0.0):
        super(BasicManeuverReward, self).__init__(env)
        self.center_coeff = center_coeff
        self.jitter_coeff = jitter_coeff
        self.curvature_deque = deque(maxlen=30)

    def reset(self, **kwargs):
        self.curvature_deque.clear()
        self.curvature_deque.append(0.)
        return super().reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = super().step(action)
        for agent_id in reward.keys():
            agent = self.world.agents[self.agent_ids.index(agent_id)]
            # compute center reward
            free_width = (agent.trace.road_width - agent.car_width) * self.free_width_mul
            center_rew = 1. - (agent.relative_state.translation_x / free_width) ** 2
            # compute jitter reward
            self.curvature_deque.append(agent.model_curvature)
            dcurvature = np.array(self.curvature_deque) / (self.upper_curvature_bound - self.lower_curvature_bound)
            dcurvature = dcurvature[1:] - dcurvature[:-1]
            jitter_rew = dcurvature.std()
            # assign reward
            reward[agent_id] = self.center_coeff * center_rew + self.jitter_coeff * jitter_rew
        return observation, reward, done, info
