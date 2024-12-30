# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Helpful layers and functions for UNet and Autoencoder construction."""

from typing import Optional, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import xformers  # type: ignore
except:
    pass

_T = TypeVar("_T", bound=nn.Module)


def zero_module(module: _T) -> _T:
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


class ResNetBlock(nn.Module):
    """Basic ResNet block.

    Args:
        input_channels (int): Number of input channels.
        output_channels (int): Number of output channels.
        use_conv_shortcut (bool): Whether to use a conv on the shortcut. Default: `False`.
        dropout (float): Dropout probability. Defaults to 0.0.
        zero_init_last (bool): Whether to initialize the last conv layer to zero. Default: `False`.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: Optional[int] = None,
        use_conv_shortcut: bool = False,
        dropout_probability: float = 0.0,
        zero_init_last: bool = False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels if output_channels is not None else input_channels
        self.use_conv_shortcut = use_conv_shortcut
        self.dropout_probability = dropout_probability
        self.zero_init_last = zero_init_last

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=self.input_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(self.input_channels, self.output_channels, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="linear")
        # Output layer is immediately after a silu. Need to account for that in init.
        self.conv1.weight.data *= 1.6761
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=self.output_channels, eps=1e-6, affine=True)
        self.dropout = nn.Dropout2d(p=self.dropout_probability)
        self.conv2 = nn.Conv2d(self.output_channels, self.output_channels, kernel_size=3, stride=1, padding=1)

        # Optionally use a conv on the shortcut, but only if the input and output channels are different
        if self.input_channels != self.output_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    self.input_channels, self.output_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.conv_shortcut = nn.Conv2d(
                    self.input_channels, self.output_channels, kernel_size=1, stride=1, padding=0
                )
            nn.init.kaiming_normal_(self.conv_shortcut.weight, nonlinearity="linear")
        else:
            self.conv_shortcut = nn.Identity()

        # Init the final conv layer parameters to zero if desired. Otherwise, kaiming uniform
        if self.zero_init_last:
            self.residual_scale = 1.0
            self.conv2 = zero_module(self.conv2)
        else:
            self.residual_scale = 0.70711
            nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="linear")
            # Output layer is immediately after a silu. Need to account for that in init.
            self.conv2.weight.data *= 1.6761 * self.residual_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through the residual block."""
        shortcut = self.residual_scale * self.conv_shortcut(x)
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + shortcut





class AttentionLayer(nn.Module):
    """Basic single headed attention layer for use on tensors with HW dimensions.

    Args:
        input_channels (int): Number of input channels.
        dropout (float): Dropout probability. Defaults to 0.0.
    """

    def __init__(self, input_channels: int, dropout_probability: float = 0.0):
        super().__init__()
        self.input_channels = input_channels
        self.dropout_probability = dropout_probability
        # Normalization layer. Here we're using groupnorm to be consistent with the original implementation.
        self.norm = nn.GroupNorm(num_groups=32, num_channels=self.input_channels, eps=1e-6, affine=True)
        # Conv layer to transform the input into q, k, and v
        self.qkv_conv = nn.Conv2d(self.input_channels, 3 * self.input_channels, kernel_size=1, stride=1, padding=0)
        # Init the qkv conv weights
        nn.init.kaiming_normal_(self.qkv_conv.weight, nonlinearity="linear")
        # Conv layer to project to the output.
        self.proj_conv = nn.Conv2d(self.input_channels, self.input_channels, kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal_(self.proj_conv.weight, nonlinearity="linear")

    def _reshape_for_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape the input tensor for attention."""
        # x is (B, C, H, W), need it to be (B, H*W, C) for attention
        x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1]).contiguous()
        return x

    def _reshape_from_attention(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Reshape the input tensor from attention."""
        # x is (B, H*W, C), need it to be (B, C, H, W) for conv
        x = x.reshape(x.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through the attention layer."""
        # Need to remember H, W to get back to it
        H, W = x.shape[2:]
        h = self.norm(x)
        # Get q, k, and v
        qkv = self.qkv_conv(h)
        qkv = self._reshape_for_attention(qkv)
        q, k, v = torch.split(qkv, self.input_channels, dim=2)
        # Use torch's built in attention function
        h = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_probability)
        # Reshape back into an image style tensor
        h = self._reshape_from_attention(h, H, W)
        # Project to the output
        h = self.proj_conv(h)
        return x + h


class Downsample(nn.Module):
    """Downsampling layer that downsamples by a factor of 2.

    Args:
        input_channels (int): Number of input channels.
        resample_with_conv (bool): Whether to use a conv for downsampling.
    """

    def __init__(self, input_channels: int, resample_with_conv: bool):
        super().__init__()
        self.input_channels = input_channels
        self.resample_with_conv = resample_with_conv
        if self.resample_with_conv:
            self.conv = nn.Conv2d(self.input_channels, self.input_channels, kernel_size=3, stride=2, padding=0)
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.resample_with_conv:
            # Need to do asymmetric padding to ensure the correct pixels are used in the downsampling conv
            # and ensure the correct output size is generated for even sizes.
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class Upsample(nn.Module):
    """Upsampling layer that upsamples by a factor of 2.

    Args:
        input_channels (int): Number of input channels.
        resample_with_conv (bool): Whether to use a conv for upsampling.
    """

    def __init__(self, input_channels: int, resample_with_conv: bool):
        super().__init__()
        self.input_channels = input_channels
        self.resample_with_conv = resample_with_conv
        if self.resample_with_conv:
            self.conv = nn.Conv2d(self.input_channels, self.input_channels, kernel_size=3, stride=1, padding=1)
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest", antialias=False)
        if self.resample_with_conv:
            x = self.conv(x)
        return x
