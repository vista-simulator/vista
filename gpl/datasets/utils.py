from typing import List
import numpy as np
import torchvision.transforms.functional as TF

from vista.entities.sensors.Camera import Camera
from vista.entities.sensors.Lidar import Lidar
from vista.entities.sensors.lidar_utils import Pointcloud
from vista.entities.sensors.EventCamera import EventCamera
from vista.core.Display import events2frame


def transform_lidar(pcd: Pointcloud, sensor: Lidar, train: bool):
    pcd = pcd[pcd.dist < 40.]
    xyz = pcd.xyz / 100.
    intensity = np.log(pcd.intensity)
    intensity = intensity - intensity.mean()
    data = np.concatenate((xyz, intensity[:, np.newaxis]), axis=1)
    data = data.astype(np.float32)
    data = TF.to_tensor(data)
    return data


def transform_rgb(img: np.ndarray, sensor: Camera, train: bool):
    (i1, j1, i2, j2) = sensor.camera_param.get_roi()
    # need copy here probably since img is not contiguous
    img = img[i1:i2, j1:j2].copy()
    img = TF.to_tensor(img)
    if train:  # perform color jitter
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


def transform_events(events: List[np.ndarray], sensor: EventCamera,
                     train: bool):
    cam_h = sensor.camera_param.get_height()
    cam_w = sensor.camera_param.get_width()
    frame = events2frame(events, cam_h, cam_w)
    frame = TF.to_tensor(frame)
    frame = standardize(frame)
    return frame


def standardize(x):
    # follow https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization
    mean, stddev = x.mean(), x.std()
    adjusted_stddev = max(stddev, 1.0 / np.sqrt(np.prod(x.shape)))
    return (x - mean) / adjusted_stddev
