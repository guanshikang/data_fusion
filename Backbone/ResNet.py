# -*- encoding: utf-8 -*-
"""
@brief: ResNet for surface reflectance

@author: guanshikang

@type: script

Created on Wed May 07 13:39:37 2024, HONG KONG
"""
import torch.nn as nn
import torchvision


class ResNet(nn.Module):
    """
    定义ResNet模型.
    """
    def __init__(self, in_channels, out_channels=7):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=False)
        self.dropout = nn.Dropout(p=0.1)
        self.resnet.conv1 = nn.Conv2d(
            in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.trans_conv1 = nn.Sequential(
            # nn.ConvTranspose2d(2048, 1024, 3, 2, 1, 1),
            nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True),
            nn.Conv2d(2048, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.GELU()
        )
        self.trans_conv2 = nn.Sequential(
            # nn.ConvTranspose2d(1024, 512, 3, 2, 1, 1),
            nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True),
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.GELU()
        )
        self.trans_conv3 = nn.Sequential(
            # nn.ConvTranspose2d(512, 256, 3, 2, 1, 1),
            nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.GELU()
        )
        self.trans_conv4 = nn.Sequential(
            # nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True),
            nn.Conv2d(256, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        self.trans_conv5 = nn.Sequential(
            # nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        self.pred = nn.Sequential(
            nn.Conv2d(64, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_d2 = self.resnet.conv1(x)
        x_d2 = self.resnet.bn1(x_d2)
        x_d2 = self.resnet.relu(x_d2)
        x_d4 = self.resnet.maxpool(x_d2)
        x_d4 = self.resnet.layer1(x_d4)
        x_d8 = self.resnet.layer2(x_d4)
        x_d16 = self.resnet.layer3(x_d8)
        x_d32 = self.resnet.layer4(x_d16)
        x_up_d16 = self.trans_conv1(x_d32)
        x_up_d16 = x_up_d16 + x_d16
        x_up_d8 = self.trans_conv2(x_up_d16)
        x_up_d8 = x_up_d8 + x_d8
        x_up_d4 = self.trans_conv3(x_up_d8)
        x_up_d4 = x_up_d4 + x_d4
        x_up_d2 = self.trans_conv4(x_up_d4)
        x_up_d2 += x_d2
        x_up_d1 = self.trans_conv5(x_up_d2)
        logits = self.pred(x_up_d1)
        return logits
