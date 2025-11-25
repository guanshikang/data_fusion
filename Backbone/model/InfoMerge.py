# -*- encoding: utf-8 -*-
"""
@type: module

@brief: MODIS Information Merge.

@author: guanshikang

Created on Fri Nov 21 14:53:05 2025, HONG KONG
"""
import torch
import torch.nn as nn

class SpatialDownScale(nn.Module):
    """U-Net DownSample Module"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv1 = nn.Sequential(
            nn.Conv3d(in_channels,
                      out_channels // 2,
                      kernel_size=(1, 3, 3),
                      padding=(0, 1, 1)),
            nn.BatchNorm3d(out_channels // 2),
            nn.GELU(),
            nn.Conv3d(out_channels // 2,
                      out_channels // 2,
                      kernel_size=(1, 3, 3),
                      padding=(0, 1, 1)),
            nn.BatchNorm3d(out_channels // 2),
            nn.GELU()
        )
        self.maxpool = nn.MaxPool3d((1, 2, 2))
        self.double_conv2 = nn.Sequential(
            nn.Conv3d(out_channels // 2,
                      out_channels,
                      kernel_size=(1, 3, 3),
                      padding=(0, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels,
                      out_channels,
                      kernel_size=(1, 3, 3),
                      padding=(0, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        x = self.double_conv1(x)
        x = self.maxpool(x)
        x = self.double_conv2(x)
        return x.permute(0, 2, 3, 4, 1)


class SpatialUpScale(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pre_conv = nn.Sequential(
            nn.Conv3d(in_channels,
                      out_channels,
                      kernel_size=(1, 3, 3),
                      padding=(0, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.GELU()
        )
        self.up_scale = nn.Upsample(scale_factor=(1, 2, 2),
                                    mode='trilinear',
                                    align_corners=False)
        self.post_conv = nn.Sequential(
            nn.Conv3d(out_channels,
                      out_channels,
                      kernel_size=(1, 3, 3),
                      padding=(0, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.up_scale(x)
        x = self.post_conv(x)

        return x


class InformationMerge(nn.Module):
    def __init__(self,
                 in_channels=(2, 9),
                 embed_dim=96
                ):
        super().__init__()
        self.up_scale = SpatialUpScale(in_channels[1], embed_dim)
        self.res_conv = nn.Conv3d(in_channels[0], embed_dim, 1) if in_channels[0] != embed_dim else nn.Identity()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels[0],
                      embed_dim,
                      kernel_size=(1, 3, 3),
                      padding=(0, 1, 1)),
            nn.BatchNorm3d(embed_dim),
            nn.GELU(),
            nn.Conv3d(embed_dim,
                      embed_dim,
                      kernel_size=(1, 3, 3),
                      padding=(0, 1, 1)),
            nn.BatchNorm3d(embed_dim),
            nn.GELU()
        )
        self.final_conv = nn.Conv3d(embed_dim * 2, embed_dim, kernel_size=1)

    def forward(self, x1, x2):
        x2 = self.up_scale(x2)
        x1_res = self.res_conv(x1)
        x1 = self.double_conv(x1)
        x1 = x1 + x1_res
        x = torch.cat([x1, x2], dim=1)
        x = self.final_conv(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        B, _, _, _, C = x.shape
        x = x.view(B, -1, C)

        return x
