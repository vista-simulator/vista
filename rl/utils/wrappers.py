import os
from collections import deque
import gym
import numpy as np
import cv2
from PIL import Image
import torchvision
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from vista.entities.sensors import Camera
from vista.entities.agents.Dynamics import tireangle2curvature


class ToRllib(MultiAgentEnv):
    def __init__(self, env):
        super(ToRllib, self).__init__()
        ref_agent = env._world.agents[0]

        obs_space = []
        for sensor in ref_agent.sensors:
            if isinstance(sensor, Camera):
                cam = sensor.camera_param
                obs_space.append(gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(cam.get_height(), cam.get_width(), 3),
                    dtype=np.uint8
                ))
            else:
                raise ValueError(f'Sensor type {type(sensor)} not supported')
        if len(obs_space) > 1:
            assert False, 'Tuple observation not supported yet'
            self.observation_space = gym.spaces.Tuple(obs_space)
        else:
            self.observation_space = obs_space

        self._curvature_bound = [tireangle2curvature(_v, ref_agent.wheel_base) \
            for _v in ref_agent.ego_dynamics._steering_bound]
        self.action_space = gym.spaces.Box(
            low=np.array([self._curvature_bound[0]]),
            high=np.array([self._curvature_bound[1]]),
            shape=(1,),
            dtype=np.float32)

        self.reward_range = [0., 1000.]

        self.metadata = {
            'render.modes': ['rgb_array'],
            'video.frames_per_second': 10
        }

        self._env = env

    def reset(self):
        obs = self._env.reset()
        return self.parse_observation(obs)

    def step(self, action):
        obs, rew, done, info = self._env.step(action)
        obs = self.parse_observation(obs)
        done['__all__'] = np.all(list(done.values()))
        return obs, rew, done, info

    def parse_observation(self, obs):
        # TODO support tuple observation
        sensor_name = self._env._world.agents[0].sensors[0].name
        return {k: v[sensor_name] for k, v in obs.items()}

    @property
    def world(self):
        return self._env._world

    @property
    def agent_ids(self):
        return [v.id for v in self.world.agents]

    @property
    def curvature_bound(self):
        return self._curvature_bound

    def close(self):
        # TODO: close simulation
        pass


class PreprocessObservation(gym.Wrapper, MultiAgentEnv):
    def __init__(self, env, fx=1.0, fy=1.0, custom_roi_crop=None, standardize=False, 
                 imagenet_normalize=False, random_gamma=None, color_jitter=None,
                 randomize_at_episode=False):
        super(PreprocessObservation, self).__init__(env)
        self.fx, self.fy = fx, fy
        self.standardize = standardize
        self.imagenet_normalize = imagenet_normalize
        self.random_gamma = random_gamma
        self.color_jitter = color_jitter
        self.randomize_at_episode = randomize_at_episode
        if self.random_gamma is not None:
            assert len(random_gamma) == 2, 'Random gamma requires 2 params: min_gamma and max_gamma'
        if self.color_jitter is not None:
            assert len(color_jitter) == 4, 'Color jitter requires 4 params: brightness / contrast / saturation / hue'
        self.roi = env.world.agents[0].sensors[0].camera_param.get_roi() \
            if custom_roi_crop is None else custom_roi_crop # NOTE: use sensor config from the first agent
        (i1, j1, i2, j2) = self.roi
        new_h, new_w = int(np.round((i2 - i1) * self.fy)), int(np.round((j2 - j1) * self.fx))
        if self.standardize or self.imagenet_normalize:
            low, high, dtype = -100., 100., np.float
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

    def reset(self, **kwargs):
        # regular reset
        observation = super().reset(**kwargs)
        # sample episode-level data aug
        if self.randomize_at_episode:
            if self.random_gamma:
                self.episode_gamma = np.random.uniform(*self.random_gamma)
            if self.color_jitter is not None:
                self.episode_brightness = np.random.uniform(1.-self.color_jitter[0], 1.+self.color_jitter[0])
                self.episode_contrast = np.random.uniform(1.-self.color_jitter[1], 1.+self.color_jitter[1])
                self.episode_saturation = np.random.uniform(1.-self.color_jitter[2], 1.+self.color_jitter[2])
                self.episode_hue = np.random.uniform(-self.color_jitter[3], +self.color_jitter[3])
        observation = self.pp_observation(observation)
        return observation

    def step(self, action):
        observation, reward, done, info = super().step(action)
        observation = self.pp_observation(observation)
        return observation, reward, done, info
    
    def pp_observation(self, observation):
        (i1, j1, i2, j2) = self.roi
        if isinstance(observation, dict):
            out = dict()
            for k, v in observation.items():
                obs = v[0] if isinstance(v, list) else v
                pp_obs = Image.fromarray(obs[:,:,::-1]) # PIL follows RGB
                if self.random_gamma:
                    pp_obs = self._adjust_gamma(pp_obs)
                if self.color_jitter is not None:
                    pp_obs = self._adjust_brightness(pp_obs)
                    pp_obs = self._adjust_contrast(pp_obs)
                    pp_obs = self._adjust_saturation(pp_obs)
                    pp_obs = self._adjust_hue(pp_obs)
                pp_obs = np.array(pp_obs)[:,:,::-1]

                pp_obs = cv2.resize(pp_obs[i1:i2, j1:j2], None, fx=self.fx, fy=self.fy)

                if self.imagenet_normalize:
                    pp_obs = self._imagenet_normalize(pp_obs)
                if self.standardize:
                    pp_obs = self._standardize(pp_obs)
                out[k] = [pp_obs] + v[1:] if isinstance(v, list) else pp_obs
    
                cropped_obs = cv2.resize(obs[i1:i2, j1:j2], None, fx=self.fx, fy=self.fy)
                if hasattr(self, 'observation_for_render'):
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

    def _adjust_hue(self, x):
        if self.randomize_at_episode:
            factor = self.episode_hue
        else:
            low = -self.color_jitter[3] # NOTE: hue factor range from -0.5 to 0.5
            high = self.color_jitter[3]
            factor = np.random.uniform(low, high)
        return torchvision.transforms.functional.adjust_hue(x, factor)

    def _adjust_saturation(self, x):
        if self.randomize_at_episode:
            factor = self.episode_saturation
        else:
            low = 1. - self.color_jitter[2]
            high = 1. + self.color_jitter[2]
            factor = np.random.uniform(low, high)
        return torchvision.transforms.functional.adjust_saturation(x, factor)

    def _adjust_contrast(self, x):
        if self.randomize_at_episode:
            factor = self.episode_contrast
        else:
            low = 1. - self.color_jitter[1]
            high = 1. + self.color_jitter[1]
            factor = np.random.uniform(low, high)
        return torchvision.transforms.functional.adjust_contrast(x, factor)

    def _adjust_brightness(self, x):
        if self.randomize_at_episode:
            factor = self.episode_brightness
        else:
            low = 1. - self.color_jitter[0]
            high = 1. + self.color_jitter[0]
            factor = np.random.uniform(low, high)
        return torchvision.transforms.functional.adjust_brightness(x, factor)

    def _adjust_gamma(self, x):
        if self.randomize_at_episode:
            factor = self.episode_gamma
        else:
            factor = np.random.uniform(*self.random_gamma)
        return torchvision.transforms.functional.adjust_gamma(x, factor)

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
    def __init__(self, env, center_coeff=0.01, jitter_coeff=0.0, 
                 inherit_reward=False, curvature_deque_len=3):
        super(BasicManeuverReward, self).__init__(env)
        self.center_coeff = center_coeff
        self.jitter_coeff = jitter_coeff
        self.inherit_reward = inherit_reward
        assert curvature_deque_len >= 3, 'Otherwise cannot compute second derivative'
        self.curvature_deque = deque(maxlen=curvature_deque_len)

    def reset(self, **kwargs):
        self.curvature_deque.clear()
        self.curvature_deque.append(0.)
        self.curvature_deque.append(0.)
        return super().reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = super().step(action)
        for agent_id in reward.keys():
            agent = self.world.agents[self.agent_ids.index(agent_id)]
            # compute center reward
            free_width = agent.trace.road_width - agent.width
            center_rew = 1. - (agent.relative_state.x / free_width) ** 2
            # compute jitter reward
            self.curvature_deque.append(agent.curvature)
            curvature = np.array(self.curvature_deque) / (self.curvature_bound[1] - self.curvature_bound[0])
            dcurvature = curvature[1:] - curvature[:-1]
            ddcurvature = dcurvature[1:] - dcurvature[:-1]
            jitter = np.abs(ddcurvature).mean()
            info[agent_id]['jitter'] = jitter
            jitter_rew = -jitter
            # assign reward
            if self.inherit_reward:
                reward[agent_id] += self.center_coeff * center_rew + self.jitter_coeff * jitter_rew
            else:
                reward[agent_id] = self.center_coeff * center_rew + self.jitter_coeff * jitter_rew
        return observation, reward, done, info


class SingleAgent(gym.Wrapper):
    def __init__(self, env, single_agent_id=None):
        super(SingleAgent, self).__init__(env)
        self.single_agent_id = self.ref_agent_id if single_agent_id is None else single_agent_id

    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        return self._wrap_single_agent(observation)

    def step(self, action):
        action = {self.single_agent_id: action}
        observation, reward, done, info = map(self._wrap_single_agent, super().step(action))
        return observation, reward, done, info

    def _wrap_single_agent(self, data):
        return data[self.single_agent_id]