# -*- encoding: utf-8 -*-
"""
@brief: UNet for surface reflectance

@author: guanshikang

@type: script

Created on Sun May 11 17:54:25 2025, HONG KONG
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    (conv -> [BN] -> ReLU) * 2
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownScale(nn.Module):
    """
    下采样.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpScale(nn.Module):
    """
    上采样.
    """
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        # if bilinear, use normal convolutions to reduce the number of channels
        if bilinear:
            self.upscale = nn.Upsample(scale_factor=2,
                                       mode='bilinear',
                                       align_corners=True)
        else:
            self.upscale = nn.Sequential(
                nn.ConvTranspose2d(in_channels,
                                   in_channels // 2,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(in_channels // 2),
                nn.ReLU(inplace=True)
            )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Args:
            x1 (tensor): Features in the RIGHT section.
            x2 (tensor): Features in the LEFT section.
        """
        x1 = self.upscale(x1)

        diffH = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffW = torch.tensor([x2.size()[3] - x1.size()[3]])

        # # OPTION 1: Padding 张量，下采样的特征形状不变
        # x1 = F.pad(x1, [diffH // 2, diffH - diffH // 2,
        #                 diffW // 2, diffW - diffW // 2])

        # OPTION 2: Crop 张量
        x2 = x2[:, :, diffH // 2:x2.size()[2] - diffH // 2,
                diffW // 2:x2.size()[3] - diffW // 2]
        x = torch.cat([x1, x2], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, 64)
        self.down_scale1 = DownScale(64, 128)
        self.down_scale2 = DownScale(128, 256)
        self.down_scale3 = DownScale(256, 512)
        self.down_scale4 = DownScale(512, 1024)
        self.up_scale1 = UpScale(1024, 512)
        self.up_scale2 = UpScale(512, 256)
        self.up_scale3 = UpScale(256, 128)
        self.up_scale4 = UpScale(128, 64)
        self.out_conv = OutConv(64, 7)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down_scale1(x1)
        x3 = self.down_scale2(x2)
        x4 = self.down_scale3(x3)
        x5 = self.down_scale4(x4)
        x = self.up_scale1(x5, x4)
        x = self.up_scale2(x, x3)
        x = self.up_scale3(x, x2)
        x = self.up_scale4(x, x1)
        logits = self.out_conv(x)

        return logits
