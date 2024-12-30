import torch
import torch.nn as nn
import numpy as np
from typing import Optional, TypeVar
import math
import torch.nn.functional as F
from timm.models.vision_transformer import Attention, Mlp
import einops
from my_LDM.vae.layers import AttentionLayer, ResNetBlock, Downsample, zero_module


class BiCNN_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # --------------------------- input --------------------------- #
        self.conv1 = nn.Conv2d(38, 256, kernel_size=3, stride=1, padding=1)
        self.blocks_1 = nn.Sequential(
            # -> [150,8×50/2=200]
            ResNetBlock(256, 256),
            ResNetBlock(256, 256),
            Downsample(input_channels=256, resample_with_conv=True),
            # -> [75,100]
            ResNetBlock(256, 256),
            ResNetBlock(256, 512),
            Downsample(input_channels=512, resample_with_conv=True),
            # -> [37,50]
            ResNetBlock(512, 512),
            ResNetBlock(512, 768),
            Downsample(input_channels=768, resample_with_conv=True),
            # -> [18,25]
            ResNetBlock(768, 768),
            ResNetBlock(768, 1024),
            Downsample(input_channels=1024, resample_with_conv=True),
            # -> [9,12]
            ResNetBlock(1024, 1024),
            ResNetBlock(1024, 1024),
            AttentionLayer(input_channels=1024),
            ResNetBlock(1024, 1024),
            ResNetBlock(1024, 1024),
            AttentionLayer(input_channels=1024),
        )

    def forward(self, x):
        h = self.conv1(x)
        h = self.blocks_1(h)
        return h


class BiCNN_Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            # -> [9,12]
            ResNetBlock(1024, 768),
            ResNetBlock(768, 768),
            ResNetBlock(768, 768),
            nn.Upsample(scale_factor=2, mode="nearest"),
            # -> [18,24]
            nn.ConstantPad2d((0, 1, 0, 0), 0),
            # -> [18,25]
            ResNetBlock(768, 512),
            ResNetBlock(512, 512),
            ResNetBlock(512, 512),
            nn.Upsample(scale_factor=2, mode="nearest"),
            # -> [36,50]
            nn.ConstantPad2d((0, 0, 0, 1), 0),
            # -> [37,50]
            ResNetBlock(512,  512),
            ResNetBlock(512, 512),
            ResNetBlock(512, 512),
            nn.Upsample(scale_factor=2, mode="nearest"),
            # -> [74,100]
            nn.ConstantPad2d((0, 0, 0, 1), 0),
            # -> [75,100]
            ResNetBlock(512, 256),
            ResNetBlock(256, 256),
            ResNetBlock(256, 256),
            nn.Upsample(scale_factor=2, mode="nearest"),
            # -> [150,200]
            ResNetBlock(256, 256),
            ResNetBlock(256, 256),
            nn.Conv2d(256, 36, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        h = self.blocks(x)
        return h


class BiCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BiCNN_Encoder()
        self.decoder = BiCNN_Decoder()

    def forward(self, x):
        # res connect里，不要最后floris的部分
        h = self.encoder(x)
        h = self.decoder(h)
        return h
