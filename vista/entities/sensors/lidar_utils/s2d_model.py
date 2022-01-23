import torch
from torch import nn
import numpy as np


class LidarModel(torch.nn.Module):
    def __init__(self, layers, filters):
        super().__init__()

        self.unet = UNet(in_channels=3,
                         out_channels=2,
                         layers=layers,
                         filters=filters)

        scale = np.array([50., 255., 1.], dtype=np.float32).reshape(1, 3, 1, 1)

        # Save constants into the state dict of the model (for loading back)
        self.register_buffer('scale', torch.from_numpy(scale))
        self.register_buffer('layers', torch.tensor(layers))
        self.register_buffer('filters', torch.tensor(filters))

    def __call__(self, x):
        scale = self.scale
        x = self.unet(x / scale)
        x = scale[:, :2] * torch.exp(x - torch.log(scale[:, :2]) / 2.)
        return x


class UNet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 layers=2,
                 filters=32,
                 kernel=3,
                 pad=1):
        super().__init__()

        # Convolutional
        self.conv_down = [
            self.contract_block(in_channels, filters, kernel, pad)
        ]
        filt = filters
        for i in range(layers):
            block = self.contract_block(filt, 2 * filt, kernel, pad)
            self.conv_down.append(block)
            filt *= 2
        self.conv_down = nn.ModuleList(self.conv_down)

        # Deconvolutional
        self.conv_up = [self.expand_block(filt, filt // 2, kernel, pad)]
        filt //= 2
        for i in range(layers - 1):
            block = self.expand_block(filt * 2, filt // 2, kernel, pad)
            self.conv_up.append(block)
            filt //= 2
        head = self.expand_block(filt * 2, out_channels, kernel, pad)
        self.conv_up.append(head)
        self.conv_up = nn.ModuleList(self.conv_up)

        # Final layer with no activation
        self.final = torch.nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def __call__(self, x):
        # Downsampling
        outs = []
        for layer in self.conv_down:
            x = layer(x)
            outs.append(x)
        # conv1 = self.conv1(x)
        # conv2 = self.conv2(conv1)
        # conv3 = self.conv3(conv2)

        # Upsampling
        x = self.conv_up[0](x)
        # upconv3 = self.upconv3(conv3)

        for layer, skip in zip(self.conv_up[1:], outs[:-1][::-1]):
            clip = torch.tensor(x.shape)[-2:] - torch.tensor(skip.shape)[-2:]
            x = x[:, :, clip[0]:, clip[1]:].clone() # make shape of x match skip
            x = layer(torch.cat([x, skip], 1))
        # upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        # upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))
        x = self.final(x)

        return x
        # return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding),
            torch.nn.BatchNorm2d(out_channels), torch.nn.LeakyReLU(),
            torch.nn.Conv2d(out_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding),
            torch.nn.BatchNorm2d(out_channels), torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(
            torch.nn.Conv2d(in_channels,
                            out_channels,
                            kernel_size,
                            stride=1,
                            padding=padding),
            torch.nn.BatchNorm2d(out_channels), torch.nn.LeakyReLU(),
            torch.nn.Conv2d(out_channels,
                            out_channels,
                            kernel_size,
                            stride=1,
                            padding=padding),
            torch.nn.BatchNorm2d(out_channels), torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     output_padding=1))
        return expand
