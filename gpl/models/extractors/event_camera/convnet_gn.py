import math
import torch
from torch import nn

__all__ = ['Net']


class Net(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        group_size = 16
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2, 2),
            nn.GroupNorm(math.ceil(24/float(group_size)), 24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 36, 5, 2, 2),
            nn.GroupNorm(math.ceil(36/float(group_size)), 36),
            nn.ReLU(inplace=True),
            nn.Conv2d(36, 48, 3, 2, 1),
            nn.GroupNorm(math.ceil(48/float(group_size)), 48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, 3, 1, 1),
            nn.GroupNorm(math.ceil(64/float(group_size)), 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.GroupNorm(math.ceil(64/float(group_size)), 64),
            nn.ReLU(inplace=True)
        )

    @property
    def feature_size(self) -> int:
        return 2560

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        z = z.mean(2)
        z = z.flatten(start_dim=1, end_dim=2)
        return z
