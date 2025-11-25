# -*- encoding: utf-8 -*-
"""
@type: module

@brief: CNN Enhancement for MODIS data.

@author: guanshikang

Created on Fri Nov 21 14:47:13 2025, HONG KONG
"""
import torch
import torch.nn as nn

class SpectralAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 reduction_ratio=4.0
                ):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.attn = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, _, _, _ = x.shape
        y = self.avg_pool(x).view(B, C)
        attn_weights = self.attn(y).view(B, C, 1, 1, 1)

        return x * attn_weights.expand_as(x)


class CNNFeatureEnhancement(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=None
                ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.GELU()
        )
        self.spectral_attn = SpectralAttention(out_channels, reduction_ratio=2.0)
        self.spatial_attn = nn.Sequential(
            nn.Conv3d(out_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        x = self.double_conv(x)
        x = self.spectral_attn(x)
        sa = self.spatial_attn(x)
        x = x * sa
        output = x + residual
        return output
