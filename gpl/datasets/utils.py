from typing import List

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
from vista.core.Display import events2frame
from vista.entities.sensors.Camera import Camera
from vista.entities.sensors.EventCamera import EventCamera
from vista.entities.sensors.Lidar import Lidar
from vista.entities.sensors.lidar_utils import Pointcloud


def transform_lidar(pcd: Pointcloud,
                    sensor: Lidar,
                    train: bool,
                    label: float = None):
    # pcd = pcd[pcd.dist < 40.]

    xyz = pcd.xyz
    intensity = pcd.intensity
    if train:
        random_scale = [0.95, 1.05]
        random_flip = 0.5

        xyz = xyz * np.random.uniform(*random_scale)
        if label is not None and np.random.uniform() < random_flip:
            xyz[:, 1] *= -1.0
            label *= -1.0

    intensity = np.log(1 + intensity)
    intensity = intensity - intensity.mean()

    lidar = np.concatenate((xyz, intensity[:, np.newaxis]), axis=1)
    lidar = lidar.astype(np.float32)

    coords, feats = lidar[:, :3], lidar
    coords = np.round(coords / 0.1)
    coords -= coords.min(0, keepdims=1)
    coords, indices = sparse_quantize(coords, return_index=True)
    feats = feats[indices]
    tensor = SparseTensor(torch.from_numpy(feats), torch.from_numpy(coords))

    return tensor if label is None else (tensor, label)


def transform_rgb(img: np.ndarray,
                  sensor: Camera,
                  train: bool,
                  label: float = None):
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
    return img if label is None else (img, label)


def transform_events(events: List[np.ndarray],
                     sensor: EventCamera,
                     train: bool,
                     label: float = None):
    cam_h = sensor.camera_param.get_height()
    cam_w = sensor.camera_param.get_width()
    frame = events2frame(events, cam_h, cam_w)
    frame = TF.to_tensor(frame)
    frame = standardize(frame)
    return frame if label is None else (frame, label)


def standardize(x):
    # follow https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization
    mean, stddev = x.mean(), x.std()
    adjusted_stddev = max(stddev, 1.0 / np.sqrt(np.prod(x.shape)))
    return (x - mean) / adjusted_stddev
