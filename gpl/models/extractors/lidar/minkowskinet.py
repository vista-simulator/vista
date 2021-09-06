import torch
from torch import nn
from torchsparse import SparseTensor
from torchsparse import nn as spnn
from torchsparse.nn import functional as F

__all__ = ['MinkowskiNet']


class ResBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            spnn.Conv3d(in_channels,
                        out_channels,
                        kernel_size,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(out_channels, track_running_stats=False),
            spnn.ReLU(True),
            spnn.Conv3d(out_channels,
                        out_channels,
                        kernel_size,
                        dilation=dilation),
            spnn.BatchNorm(out_channels, track_running_stats=False),
        )

        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                spnn.Conv3d(in_channels, out_channels, 1, stride=stride),
                spnn.BatchNorm(out_channels, track_running_stats=False),
            )
        else:
            self.downsample = nn.Sequential()

    def forward(self, x: SparseTensor) -> SparseTensor:
        return F.relu(self.features(x) + self.downsample(x))


class Net(nn.Module):

    def __init__(self, in_channels: int = 4) -> None:
        super().__init__()
        layers = []
        for out_channels in [16]:
            layers.append(
                nn.Sequential(
                    spnn.Conv3d(in_channels, out_channels, 3),
                    spnn.BatchNorm(out_channels, track_running_stats=False),
                    spnn.ReLU(True),
                ))
            in_channels = out_channels
        for out_channels in [16, 32, 64]:
            layers.extend([
                nn.Sequential(
                    spnn.Conv3d(in_channels, in_channels, 2, stride=2),
                    spnn.BatchNorm(in_channels, track_running_stats=False),
                    spnn.ReLU(True),
                ),
                ResBlock(in_channels, out_channels, 3),
            ])
            in_channels = out_channels
        self.features = nn.Sequential(*layers)

    @property
    def feature_size(self) -> int:
        return 64

    def forward(self, x: SparseTensor) -> torch.Tensor:
        x = self.features(x)
        x = F.global_avg_pool(x)
        return x
