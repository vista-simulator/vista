from typing import List, Optional

from collections import deque
import numpy as np
import torch
import torchvision.transforms.functional as TF
from vista.entities.sensors.Camera import Camera


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


def standardize(x):
    # follow https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization
    mean, stddev = x.mean(), x.std()
    adjusted_stddev = max(stddev, 1.0 / np.sqrt(np.prod(x.shape)))
    return (x - mean) / adjusted_stddev
