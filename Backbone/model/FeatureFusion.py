# -*- encoding: utf-8 -*-
"""
@type: module

@brief: Feature Fusion for specific stage.

@author: guanshikang

Created on Fri Nov 21 14:55:52 2025, HONG KONG
"""
import torch
import torch.nn as nn


class FeatureFusion(nn.Module):
    def __init__(self,
                 out_dim
                ):
        super().__init__()
        self.downsample_convs = nn.ModuleList([
            # Stage 1: 36 -> 32
            nn.Sequential(
                nn.AdaptiveAvgPool3d((None, 32, 32)),
                nn.Conv3d(in_channels=out_dim,
                          out_channels=out_dim,
                          kernel_size=(1, 3, 3),
                          padding=(0, 1, 1))
            ),
            # Stage 2: 18 -> 16
            nn.Sequential(
                nn.AdaptiveAvgPool3d((None, 16, 16)),
                nn.Conv3d(in_channels=out_dim,
                          out_channels=out_dim,
                          kernel_size=(1, 3, 3),
                          padding=(0, 1, 1))
            ),
            # Stage 3: 9 -> 8
            nn.Sequential(
                nn.AdaptiveAvgPool3d((None, 8, 8)),
                nn.Conv3d(in_channels=out_dim,
                          out_channels=out_dim,
                          kernel_size=(1, 1, 1))
            ),
            # Stage 4: 9 -> 8
            nn.Sequential(
                nn.AdaptiveAvgPool3d((None, 8, 8)),
                nn.Conv3d(in_channels=out_dim,
                          out_channels=out_dim,
                          kernel_size=(1, 1, 1))
            )
        ])

        self.linear_fuse = nn.Linear(2 * out_dim, out_dim)

    def forward(self, main_feat, main_resolution, aux_feat, aux_resolution, fusion_stage):
        B, _, C = main_feat.shape
        x_tblk, x_hblk, x_wblk = main_resolution
        x = main_feat.view(B, x_tblk, x_hblk, x_wblk, -1).permute(0, 4, 1, 2, 3)
        x1_tblk, x1_hblk, x1_wblk = aux_resolution
        x1 = aux_feat.view(B, x1_tblk, x1_hblk, x1_wblk, -1).permute(0, 4, 1, 2, 3)

        x1 = self.downsample_convs[fusion_stage](x1)
        x = torch.cat([x, x1], axis=1)
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(B, -1, C * 2)
        x = self.linear_fuse(x)

        return x
