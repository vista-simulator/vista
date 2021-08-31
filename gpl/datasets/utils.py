from typing import List
import numpy as np
import torchvision.transforms.functional as TF

from vista.utils import transform
from vista.entities.sensors.Camera import Camera
from vista.entities.sensors.Lidar import Lidar
from vista.entities.sensors.EventCamera import EventCamera
from vista.entities.agents.Dynamics import tireangle2curvature
from vista.core.Display import events2frame


def transform_lidar(pcd: np.ndarray, sensor: Lidar, train: bool):
    xyz = pcd.xyz / 100.
    intensity = np.log(pcd.intensity)
    intensity = intensity - intensity.mean()
    data = np.concatenate((xyz, intensity[:, np.newaxis]), axis=1)
    data = data.astype(np.float32)
    data = TF.to_tensor(data)
    return data


def transform_rgb(img: np.ndarray, sensor: Camera, train: bool):
    (i1, j1, i2, j2) = sensor.camera_param.get_roi()
    img = img[i1:i2, j1:j2].copy() # need copy here probably since img is not contiguous
    img = TF.to_tensor(img)
    if train: # perform color jitter
        gamma_range = [0.5, 1.5]
        brightness_range = [0.5, 1.5]
        contrast_range = [0.3, 1.7]
        saturation_range = [0.5, 1.5]

        img = TF.adjust_gamma(img, np.random.uniform(*gamma_range))
        img = TF.adjust_brightness(img, np.random.uniform(*brightness_range))
        img = TF.adjust_contrast(img, np.random.uniform(*contrast_range))
        img = TF.adjust_saturation(img, np.random.uniform(*saturation_range))
    img = standardize(img)
    return img


def transform_events(events: List[np.ndarray], sensor: EventCamera, train: bool):
    cam_h = sensor.camera_param.get_height()
    cam_w = sensor.camera_param.get_width()
    frame = events2frame(events, cam_h, cam_w)
    frame = TF.to_tensor(frame)
    frame = standardize(frame)
    return frame


def standardize(x):
    # follow https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization
    mean, stddev = x.mean(), x.std()
    adjusted_stddev = max(stddev, 1.0/np.sqrt(np.prod(x.shape)))
    return (x - mean) / adjusted_stddev


def pure_pursuit(agent, optimal_control_config):
    lookahead_dist = optimal_control_config['lookahead_dist']
    dt = optimal_control_config['dt']
    Kp = optimal_control_config['Kp']
    speed = agent.human_speed

    road = agent.road
    ego_pose = agent.ego_dynamics.numpy()[:3]
    road_in_ego = np.array([ # TODO: vectorize this: slow if road buffer size too large
        transform.compute_relative_latlongyaw(_v, ego_pose)
        for _v in road
    ])
    road_in_ego = road_in_ego[road_in_ego[:,1] > 0] # drop road in the back

    dist = np.linalg.norm(road_in_ego[:,:2], axis=1)
    tgt_idx = np.argmin(np.abs(dist - lookahead_dist))
    dx, dy, dyaw = road_in_ego[tgt_idx]

    lat_shift = -agent.relative_state.x
    dx += lat_shift * np.cos(dyaw)
    dy += lat_shift * np.sin(dyaw)

    arc_len = speed * dt
    curvature = (Kp * np.arctan2(-dx, dy) * dt) / arc_len
    curvature_bound = [
        tireangle2curvature(_v, agent.wheel_base)
        for _v in agent.ego_dynamics.steering_bound]
    curvature = np.clip(curvature, *curvature_bound)

    return curvature, speed
