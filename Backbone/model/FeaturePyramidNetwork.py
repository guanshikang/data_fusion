# -*- encoding: utf-8 -*-
"""
@type: module

@brief: FPN as decoder.

@author: guanshikang

Created on Fri Nov 21 14:50:02 2025, HONG KONG
"""
import torch
import torch.nn as nn
import torch.functional as F

class PPM(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 reduction_ratio=4.0,
                 pool_size=[(1, 1, 1), (2, 2, 2), (3, 3, 3), (6, 6, 6)]
                ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.pool_sizes = pool_size
        pyramid_out_channels = out_channels // reduction_ratio
        self.pyramid_convs = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool3d(output_size=size),
                nn.Conv3d(in_channels, pyramid_out_channels, kernel_size=1),
                nn.BatchNorm3d(pyramid_out_channels),
                nn.GELU()
            ) for size in pool_size
        ])

        self.fuse_conv = nn.Sequential(
            nn.Conv3d(in_channels + pyramid_out_channels * len(pool_size), out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        T, H, W = x.shape[2:]
        features = [x]
        for conv in self.pyramid_convs:
            pooled = conv(x)
            upsampled = F.interpolate(pooled, size=(T, H, W), mode='trilinear', align_corners=True)
            features.append(upsampled)

        concated = torch.cat(features, dim=1)
        fused = self.fuse_conv(concated)

        return fused

class FPN(nn.Module):
    def __init__(self,
                 embed_dim=768,
                 out_chans=6,
                 skip_dims=[96, 96, 192, 384, 768]
                ):
        super().__init__()
        self.pyramid_dim = 96  # Fixed channel dimension for FPN features
        num_levels = len(skip_dims)

        # Lateral convolutions for the skip connections (deep to shallow)
        self.lateral_convs = nn.ModuleList()
        for dim in skip_dims[::-1]:
            self.lateral_convs.append(nn.Conv3d(dim, self.pyramid_dim, kernel_size=1))

        # Separate lateral for the final (bottom-most) encoder output x
        self.lateral_final = nn.Conv3d(embed_dim, self.pyramid_dim, kernel_size=1)

        # Smooth convolutions after each fusion
        self.smooth_convs = nn.ModuleList()
        for _ in range(5):
            self.smooth_convs.append(
                nn.Sequential(
                    nn.Conv3d(self.pyramid_dim, self.pyramid_dim, kernel_size=3, padding=1),
                    nn.BatchNorm3d(self.pyramid_dim),
                    nn.GELU()
                )
            )

        # Integrated Pyramid Pooling Module
        self.ppm = PPM(self.pyramid_dim * num_levels, self.pyramid_dim)
        # Final convolution to output channels
        self.final_conv = nn.Sequential(
            nn.Conv3d(self.pyramid_dim, out_chans, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x, thw_blocks, encoder_features):
        B, _, _ = x.shape
        t_blk, h_blk, w_blk = thw_blocks

        # Reshape and permute the bottom-most feature
        x = x.view(B, t_blk, h_blk, w_blk, -1).permute(0, 4, 1, 2, 3)  # (B, embed_dim, t_blk, h_blk, w_blk)

        # Fuse the bottom-most x with the deepest skip feature
        # using lateral convs and addition
        deepest_skip = encoder_features[0].permute(0, 4, 1, 2, 3)
        p = self.lateral_final(x) + self.lateral_convs[0](deepest_skip)
        p = self.smooth_convs[0](p)

        # Collect all features in the top_down pathway
        pyramid_features = [p]

        # Top-down FPN pathway with upsampling and lateral additions
        for i in range(1, 5):
            # Upsample the previous pyramid level
            p = F.interpolate(p, scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
            # Add lateral connection from current skip
            skip_feat = encoder_features[i].permute(0, 4, 1, 2, 3)
            lateral = self.lateral_convs[i](skip_feat)
            p = p + lateral
            # Apply smoothing conv
            p = self.smooth_convs[i](p)

            pyramid_features.append(p)

        # Apply PPM to the FPN features
        target_size = pyramid_features[-1].shape[2:]
        upsampled_features = []
        for pf in pyramid_features:
            upsampled = F.interpolate(pf, size=target_size, mode='trilinear', align_corners=True)
            upsampled_features.append(upsampled)
        multi_level_feat = torch.cat(upsampled_features, dim=1)
        p = self.ppm(multi_level_feat)
        # Final spatial upsample to original image size
        p = F.interpolate(p, scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)

        # Apply final conv and average over the temporal dimension
        x = self.final_conv(p)
        x = x.mean(dim=2)

        return x
