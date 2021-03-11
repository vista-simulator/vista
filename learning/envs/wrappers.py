import gym
import numpy as np
import cv2
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class MultiAgentMonitor(gym.wrappers.Monitor, MultiAgentEnv):
    """ Only for expedient use. """
    def __init__(self, env, directory, video_callable=None, force=False, resume=False,
                 write_upon_reset=False, uid=None, mode=None):
        super(MultiAgentMonitor, self).__init__(env, directory, video_callable, 
            force, resume, write_upon_reset, uid, mode)

    def step(self, action):
        self._before_step(action)
        observation, reward, done, info = self.env.step(action)
        self._after_step(observation['agent_0'], reward['agent_0'], done['__all__'], info['agent_0'])

        return observation, reward, done, info


class PreprocessObservation(gym.ObservationWrapper, MultiAgentEnv):
    def __init__(self, env, fx=1.0, fy=1.0):
        super(PreprocessObservation, self).__init__(env)
        self.fx, self.fy = fx, fy
        self.roi = env.world.agents[0].sensors[0].camera.get_roi() # NOTE: use sensor config from the first agent
        (i1, j1, i2, j2) = self.roi
        new_h, new_w = int((i2 - i1) * self.fy), int((j2 - j1) * self.fx)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(new_h, new_w, 3),
            dtype=np.uint8)
    
    def observation(self, observation):
        (i1, j1, i2, j2) = self.roi
        if isinstance(observation, dict):
            out = dict()
            for k, v in observation.items():
                out[k] = cv2.resize(v[i1:i2, j1:j2], None, fx=self.fx, fy=self.fy)
        elif isinstance(observation, list):
            out = []
            for v in observation:
                out.append(cv2.resize(v[i1:i2, j1:j2], None, fx=self.fx, fy=self.fy))
        else:
            out = cv2.resize(observation[i1:i2, j1:j2], None, fx=self.fx, fy=self.fy)
        return out
