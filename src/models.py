from typing import Dict

import math
import timm

import torch
import torch.nn as nn

from src.modules import (
    ConvNormAct, ResidualLayer, SubPixelConv, 
    DecoderLayer, FBPModule
)


class UNetSEResNext50(nn.Module):

    model_name: str = 'seresnext50_32x4d'

    def __init__(self):
        super().__init__()

        encoder = timm.create_model(self.model_name)
        
        self.encoder0 = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            encoder.conv1,
            encoder.bn1,
            encoder.act1
        )
        self.encoder1 = nn.Sequential(
            encoder.maxpool,
            encoder.layer1
        )
        self.encoder2 = encoder.layer2
        self.encoder3 = encoder.layer3
        self.encoder4 = encoder.layer4

        self.bottleneck = nn.Conv2d(2048, 512, kernel_size=1)

        self.decoder4 = DecoderLayer(512 + 2048, 64)
        self.decoder3 = DecoderLayer(64 + 1024, 64)
        self.decoder2 = DecoderLayer(64 + 512, 64)
        self.decoder1 = DecoderLayer(64 + 256, 64)
        self.decoder0 = DecoderLayer(64, 64)

        for idx in range(1, 5):
            self.add_module(
                f'upsample{idx}', 
                nn.Upsample(scale_factor=2 ** idx, mode='bilinear')
            )

        self.final_layer = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )
    
    def forward(self, x):
        x0 = self.encoder0(x)  # (64,h/2,w/2)
        x1 = self.encoder1(x0)  # (256,h/4,w/4)
        x2 = self.encoder2(x1)  # (512,h/8,w/8)
        x3 = self.encoder3(x2)  # (1024,h/16,w/16)
        x4 = self.encoder4(x3)  # (2048,h/32,w/32)

        y5 = self.bottleneck(x4)  # (512,h/32,w/32)

        y4 = self.decoder4(torch.cat([y5, x4], dim=1))  # (64,h/16,w/16)
        y3 = self.decoder3(torch.cat([y4, x3], dim=1))  # (64,h/8,w/8)
        y2 = self.decoder2(torch.cat([y3, x2], dim=1))  # (64,h/4,w/4)
        y1 = self.decoder1(torch.cat([y2, x1], dim=1))  # (64,h/2,w/2)
        y0 = self.decoder0(y1)  # (64,h,w)

        y4 = self.upsample4(y4)  # (64,h,w)
        y3 = self.upsample3(y3)  # (64,h,w)
        y2 = self.upsample2(y2)  # (64,h,w)
        y1 = self.upsample1(y1)  # (64,h,w)

        output = torch.cat([y0, y1, y2, y3, y4], dim=1)
        output = self.final_layer(output)

        return output


class SRResNet(nn.Module):
    def __init__(self, n_blocks: int = 16, scaling_factor: int = 4):
        super().__init__()

        n_upscale_layers = int(math.log2(scaling_factor))

        self.transition1 = ConvNormAct(1, 64, kernel_size=9, norm='')
        self.encoder = nn.Sequential(
            *[ResidualLayer(64) for _ in range(n_blocks)])
        self.transition2 = ConvNormAct(64, 64, act='')
        self.decoder = nn.Sequential(
            *[SubPixelConv(64, scaling_factor=2) 
            for _ in range(n_upscale_layers)])
        self.transition3 = ConvNormAct(64, 1, kernel_size=9, norm='')
    
    def forward(self, x):
        x = self.transition1(x)

        residual = x
        x = self.encoder(x)
        x = self.transition2(x)
        x = residual + x
        x = self.decoder(x)
        x = self.transition3(x)

        return x


class RecModel(nn.Module):
    def __init__(
        self, 
        model_cls: nn.Module,
        proj_kwargs: Dict = None,
        model_kwargs: Dict = None, 
        subsample_factor: int = 4
    ):
        """Reconstruction model wrapper that does projection to sinograms in-place.
        
        Note: Use `model` during inference.
        """
        super().__init__()

        proj_kwargs = proj_kwargs if proj_kwargs else {}
        model_kwargs = model_kwargs if model_kwargs else {}

        self.projection = FBPModule(**proj_kwargs)
        self.model = model_cls(**model_kwargs)

        self.subsample_factor = subsample_factor
    
    def forward(self, images):
        sinograms, sparse_sinograms \
            = self.projection.encode(images, self.subsample_factor)
        rec_sinograms = self.model(sparse_sinograms)
        images = self.projection.decode(rec_sinograms)
        images = self.normalize(images)

        return images, sinograms, rec_sinograms

    def normalize(self, images):
        i_min, _ = torch.min(images.flatten(1), dim=1)
        i_max, _ = torch.max(images.flatten(1), dim=1)

        i_min = i_min[:, None, None, None]
        i_max = i_max[:, None, None, None]

        images = (images - i_min) / (i_max - i_min + 1e-8)

        return images
