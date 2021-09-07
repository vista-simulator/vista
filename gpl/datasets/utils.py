from typing import List

from collections import defaultdict
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


class RejectionSampler:
    def __init__(self, resolution: float = 0.1):
        super(RejectionSampler, self).__init__()
        self.resolution = resolution
        self.counts = defaultdict()

    def add_to_history(self, value):
        _bin = np.floor(value / self.resolution) * self.resolution
        if _bin not in self.counts.keys():
            self.counts[_bin] = 0
        self.counts[_bin] += 1
    
    def get_sampling_probability(self, value, alpha=0.1):
        all_bins = self.counts.keys()
        if len(all_bins) == 0:
            return 1.
        minbin, maxbin = min(all_bins), max(all_bins)
        num_bins = int((maxbin - minbin) / self.resolution) + 1

        bin_edges = np.linspace(minbin, maxbin, num_bins)
        bin_edges = np.insert(bin_edges, 0, -float('inf'))
        bin_edges = np.append(bin_edges, float('inf'))

        count_array = np.zeros_like(bin_edges)
        for i in range(1, len(bin_edges) - 1): # keep inf/-inf zero count
            # handle floating point keys
            key = [k for k in self.counts.keys() if np.allclose(k, bin_edges[i])]
            if len(key) != 1: # skip bins w/o counts
                continue
            count_at_bin = self.counts[key[0]]
            count_array[i] = count_at_bin

        total_area = count_array.sum() * self.resolution
        density = count_array * self.resolution / total_area

        smoothed_density = density + alpha
        smoothed_density = smoothed_density / smoothed_density.sum()

        prob = 1 / (smoothed_density + 1e-8)
        prob = prob / prob.sum()

        # we use np.floor for adding bin
        bin_idx = np.searchsorted(bin_edges, value, side='right') - 1

        sampling_probability = prob[bin_idx]

        return sampling_probability



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
