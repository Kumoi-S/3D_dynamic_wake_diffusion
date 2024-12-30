import torch
import torch.nn as nn
import numpy as np
from typing import Optional, TypeVar
import math
import torch.nn.functional as F
from timm.models.vision_transformer import Attention, Mlp
import einops
from my_LDM.vae.layers import AttentionLayer, ResNetBlock, Downsample, zero_module
from my_LDM.ldm.modules.unet_modules import ResBlock
from my_LDM.ldm.modules.dit_modules import TimestepEmbedder


class ModuleWrapper(nn.Sequential):
    """包裹模块，这样Unet里forwar直接同时输入x,emb,context即可，无需区分模块类型
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, t):
        for layer in self:
            if isinstance(layer, ModuleWrapper):
                # 如果是嵌套的 UnetModuleWrapper，确保传入 emb 和 context
                # 按照正常的编写，这个if不会触发，但是以防万一干脆写上了
                print("\033[91mUnetModuleWrapper嵌套了UnetModuleWrapper，将解包继续执行，但这是不应该的\033[0m")
                x = layer(x, t)
            elif isinstance(layer, ResBlock):
                # 如果是接受timestep embd的ResBlock
                x = layer(x, t)
            elif isinstance(layer, AttentionLayer):
                # 如果是接受context的AttentionLayer
                x = layer(x)
            else:
                # 如果是杂七杂八的，比如dowmsample, conv等
                x = layer(x)
        return x


time_embed_dim = 128 * 2


class BiCNN_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # --------------------------- input --------------------------- #
        self.conv1 = nn.Conv2d(38, 256, kernel_size=3, stride=1, padding=1)
        self.blocks_1 = ModuleWrapper(
            # -> [150,8×50/2=200]
            ResBlock(channels=256, emb_channels=time_embed_dim, out_channels=256),
            ResBlock(channels=256, emb_channels=time_embed_dim, out_channels=256),
            Downsample(input_channels=256, resample_with_conv=True),
            # -> [75,100]
            ResBlock(channels=256, emb_channels=time_embed_dim, out_channels=256),
            ResBlock(channels=256, emb_channels=time_embed_dim, out_channels=512),
            Downsample(input_channels=512, resample_with_conv=True),
            # -> [37,50]
            ResBlock(channels=512, emb_channels=time_embed_dim, out_channels=512),
            ResBlock(channels=512, emb_channels=time_embed_dim, out_channels=768),
            Downsample(input_channels=768, resample_with_conv=True),
            # -> [18,25]
            ResBlock(channels=768, emb_channels=time_embed_dim, out_channels=768),
            ResBlock(channels=768, emb_channels=time_embed_dim, out_channels=1024),
            Downsample(input_channels=1024, resample_with_conv=True),
            # -> [9,12]
        )
        self.blocks_2 = ModuleWrapper(
            ResBlock(channels=1024, emb_channels=time_embed_dim, out_channels=1024),
            ResBlock(channels=1024, emb_channels=time_embed_dim, out_channels=1024),
            AttentionLayer(input_channels=1024),
            ResBlock(channels=1024, emb_channels=time_embed_dim, out_channels=1024),
            ResBlock(channels=1024, emb_channels=time_embed_dim, out_channels=1024),
            AttentionLayer(input_channels=1024),
        )

    def forward(self, x, t):
        x = self.conv1(x)
        x = self.blocks_1(x, t)
        x = self.blocks_2(x, t)
        return x


class BiCNN_Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = ModuleWrapper(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            # -> [9,12]
            ResBlock(channels=1024, emb_channels=time_embed_dim, out_channels=768),
            ResBlock(channels=768, emb_channels=time_embed_dim, out_channels=768),
            ResBlock(channels=768, emb_channels=time_embed_dim, out_channels=768),
            nn.Upsample(scale_factor=2, mode="nearest"),
            # -> [18,24]
            nn.ConstantPad2d((0, 1, 0, 0), 0),
            # -> [18,25]
            ResBlock(channels=768, emb_channels=time_embed_dim, out_channels=512),
            ResBlock(channels=512, emb_channels=time_embed_dim, out_channels=512),
            ResBlock(channels=512, emb_channels=time_embed_dim, out_channels=512),
            nn.Upsample(scale_factor=2, mode="nearest"),
            # -> [36,50]
            nn.ConstantPad2d((0, 0, 0, 1), 0),
            # -> [37,50]
            ResBlock(channels=512, emb_channels=time_embed_dim, out_channels=512),
            ResBlock(channels=512, emb_channels=time_embed_dim, out_channels=512),
            ResBlock(channels=512, emb_channels=time_embed_dim, out_channels=512),
            nn.Upsample(scale_factor=2, mode="nearest"),
            # -> [74,100]
            nn.ConstantPad2d((0, 0, 0, 1), 0),
            # -> [75,100]
            ResBlock(channels=512, emb_channels=time_embed_dim, out_channels=256),
            ResBlock(channels=256, emb_channels=time_embed_dim, out_channels=256),
            ResBlock(channels=256, emb_channels=time_embed_dim, out_channels=256),
            nn.Upsample(scale_factor=2, mode="nearest"),
            # -> [150,200]
            ResBlock(channels=256, emb_channels=time_embed_dim, out_channels=256),
            ResBlock(channels=256, emb_channels=time_embed_dim, out_channels=256),
            nn.Conv2d(256, 38, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, t):
        h = self.blocks(x, t)
        return h


class BiCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embed = TimestepEmbedder(hidden_size=time_embed_dim)
        self.encoder = BiCNN_Encoder()
        self.decoder = BiCNN_Decoder()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # t = self.time_embed(t)
        t = t.unsqueeze(1).repeat(1, time_embed_dim)
        h = self.encoder(x, t)
        h = self.decoder(h, t)
        return h
