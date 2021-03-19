import os
from collections import deque
import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from shapely.geometry import LineString
from descartes import PolygonPatch
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
        for i, agent_id in enumerate(self.agents_with_sensor.keys()):
            this_gs = self.gs[:4, 4*i:4*(i+1)]
            self.ax_obs[agent_id] = self.fig.add_subplot(this_gs)
            self.artists['im:{}'.format(agent_id)] = self.ax_obs[agent_id].imshow(
                self.fit_img_to_ax(self.ax_obs[agent_id], \
                    np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)))
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
        for agent_id, obs in self.observation_for_render.items():
            i = self.agent_ids.index(agent_id)
            if self.crash_to_others[i]:
                text = 'Crash Into Another Car'
            elif self.world.agents[i].isCrashed:
                text = 'Out-of-lane / Exceed Max Rotation'
            else:
                text = 'Running'
            self.ax_obs[agent_id].set_title(text, color='white', size=20, weight='bold')
            self.artists['im:{}'.format(agent_id)].set_data(self.fit_img_to_ax(
                self.ax_obs[agent_id], obs[:,:,::-1]))

        # add speed and steering wheel
        for i, agent_id in enumerate(self.agents_with_sensor.keys()):
            agent = self.world.agents[i]

            current_timestamp = self.get_timestamp_readonly(agent, current=True)
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

    def get_timestamp_readonly(self, agent, index=0, current=False):
        index = agent.current_frame_index if current else index
        index = min(len(agent.trace.syncedLabeledTimestamps[
                agent.current_segment_index]) - 1, index)
        return agent.trace.syncedLabeledTimestamps[
            agent.current_segment_index][index]

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
        env = _MultiAgentMonitor(env)
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
