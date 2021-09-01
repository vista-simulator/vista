import torch
from torch import nn

__all__ = ['Net']


class Net(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, (1, 4), 1),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 1, 1),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 1, 1),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 1, 1),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, 1, 1),
            # nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True))

    @property
    def feature_size(self) -> int:
        return 1024

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # z = torch.unsqueeze(x, 1) # (B, 1, N, 3)
        x = self.features(x)
        x = torch.mean(x, dim=2)
        x = x.flatten(start_dim=1)
        return x
