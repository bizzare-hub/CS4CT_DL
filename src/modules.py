from typing import Dict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_radon import RadonFanbeam

from src.cs_utils import filter_sinogram


class ConvNormAct(nn.Module):

    _kernel2pad: Dict = {1: 0, 3: 1, 9: 4}
    _norm_mapping: Dict = {'batchnorm': nn.BatchNorm2d}
    _act_mapping: Dict = {
        'relu': nn.ReLU,
        'sigmoid': nn.Sigmoid
    }

    def __init__(
        self, in_channels: int, out_channels: int, 
        kernel_size: int = 3, norm: str = 'batchnorm', act: str = 'relu'
    ):
        super().__init__()

        pad = self._kernel2pad[kernel_size]

        layer = []

        layer.append(nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=pad))
        if norm in self._norm_mapping:
            layer.append(self._norm_mapping[norm](out_channels))
        if act:
            layer.append(self._act_mapping[act]())
        
        self.layer = nn.Sequential(*layer)
    
    def forward(self, x):
        return self.layer(x)


class ResidualLayer(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()

        self.layer1 = ConvNormAct(n_channels, n_channels)
        self.layer2 = ConvNormAct(n_channels, n_channels)
    
    def forward(self, x):
        residual = x

        x = self.layer1(x)
        x = self.layer2(x)

        return residual + x


class SubPixelConv(nn.Module):
    def __init__(self, n_channels: int, scaling_factor: int):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(n_channels, n_channels * (scaling_factor ** 2), 
                      kernel_size=3, padding=1),
            nn.PixelShuffle(scaling_factor),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


class DecoderLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.forward_layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            ConvNormAct(in_channels, out_channels),
            # ConvNormAct(in_channels, out_channels)
        )
        self.shortcut_layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        residual = self.shortcut_layer(x)
        x = residual + self.forward_layer(x)

        return x


class FBPModule(nn.Module):
    """Wrapper around Filtered back-projection algorithm.
    """
    def __init__(
        self, resolution: int, n_projections: int, 
        source_distance: float, det_distance: float = -1, 
        det_spacing: float = -1, clip_to_circle: bool = False
    ) -> None:
        super().__init__()

        angles = np.linspace(0, 2 * np.pi, n_projections, endpoint=False)
        self.fbp = RadonFanbeam(
            resolution, angles, source_distance, det_distance, 
            det_spacing=det_spacing, clip_to_circle=clip_to_circle)

    @torch.no_grad()
    def encode(self, image, subsample_factor=1):
        mask = torch.zeros_like(image)
        mask[:, :, ::subsample_factor] = 1

        sinogram = self.fbp.forward(image)
        sparse_sinogram = sinogram * mask

        return sinogram, sparse_sinogram

    def decode(self, sinogram):
        sinogram = filter_sinogram(sinogram)
        sinogram = self.fbp.backprojection(sinogram)

        return sinogram
