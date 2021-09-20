from typing import List, Optional

from collections import deque
import numpy as np
import torch
import torchvision.transforms.functional as TF
try:
    from torchsparse import SparseTensor
    from torchsparse.utils.quantize import sparse_quantize
except:
    pass
from vista.core.Display import events2frame
from vista.entities.sensors.Camera import Camera
from vista.entities.sensors.EventCamera import EventCamera
from vista.entities.sensors.Lidar import Lidar
from vista.entities.sensors.lidar_utils import Pointcloud


class RejectionSampler:
    def __init__(self, buffer_size=int(4e5)):
        super(RejectionSampler, self).__init__()
        self.samples = deque(maxlen=buffer_size)

    def add_to_history(self, value: float):
        self.samples.append(value)

    def get_sampling_probability(self,
                                 value: float,
                                 smoothing_factor: Optional[float] = 0.01,
                                 min_p: Optional[float] = 0.25,
                                 n_bins: Optional[float] = 30):

        # Find which latent bin every data sample falls in
        if len(self.samples) == 0:
            return 1.
        density, bins = np.histogram(self.samples, density=True, bins=n_bins)
        bins[0] = -float('inf')
        bins[-1] = float('inf')

        # smooth the density function
        smooth_density = density / density.sum()
        smooth_density = smooth_density + (smoothing_factor + 1e-10)
        smooth_density /= smooth_density.sum()

        # invert the density function and normalize
        p = 1.0 / smooth_density
        p = (p - p.min()) / (p.max() - p.min())
        p = np.clip(p, p + min_p, 1)

        # Find which bin the new value sample belongs to and return that prob
        bin_idx = np.digitize(value, bins)
        prob = p[bin_idx - 1]
        return prob


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
                     label: float = None,
                     use_roi: bool = False):
    cam_h = sensor.camera_param.get_height()
    cam_w = sensor.camera_param.get_width()
    frame = events2frame(events, cam_h, cam_w)
    if use_roi:
        (i1, j1, i2, j2) = sensor.camera_param.get_roi()
        # need copy here probably since img is not contiguous
        frame = frame[i1:i2, j1:j2].copy()
    frame = TF.to_tensor(frame)
    frame = standardize(frame)
    return frame if label is None else (frame, label)


def standardize(x):
    # follow https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization
    mean, stddev = x.mean(), x.std()
    adjusted_stddev = max(stddev, 1.0 / np.sqrt(np.prod(x.shape)))
    return (x - mean) / adjusted_stddev
