# -*- encoding: utf-8 -*-
"""
@brief: SpatioTemporal ViT (Single time step)

@author: guanshikang

@type: script

Created on Mon Mar 17 14:22:42 2025, HONG KONG
"""
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class SpatialViT(nn.Module):
    def __init__(self, in_channels=7, img_size=256, patch_size=16, d_model=512, depth=6, heads=8):
        super().__init__()
        self.path_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        self.to_patch_embed = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, d_model))
        encoder_layers = TransformerEncoderLayer(
            d_model,
            heads,
            d_model * 4,
            dropout=0.1
        )
        self.transformer = TransformerEncoder(encoder_layers, depth)

    def forward(self, x):
        # x: (B * T, C, H, W)
        x = self.to_patch_embed(x)  # (B * T, D, H_patch, W_patch)
        x = rearrange(x, "b d h w -> b (h w) d")
        x += self.pos_embed
        x = self.transformer(x)  # (B, num_patches, dim)

        return x


class TemporalTransformer(nn.Module):
    def __init__(self, d_model=512, depth=4, heads=8, max_len=100):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, max_len + 1, d_model))
        encoder_layers = TransformerEncoderLayer(
            d_model,
            heads,
            d_model * 4,
            dropout=0.1
        )
        self.transformer = TransformerEncoder(encoder_layers, depth)

    def forward(self, x):
        # x: (B * num_patches, T, d_model)
        Bn, T, D = x.shape
        cls_tokens = self.cls_token.expand(Bn, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        pos_embed = self.pos_embed[:, :T+1, :].expand(Bn, -1, -1)
        x += pos_embed
        x = x.permute(1, 0, 2)  # (T, B * num_patches, d_model)
        x = self.transformer(x)
        # x = x.permute(1, 0, 2)  # (B * num_patches, T, d_model)

        return x[0]


class SkipConnection(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, 3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.GELU()
        )
        self.skip = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear'),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.main(x) + self.skip(x)


class Decoder(nn.Module):
    def __init__(self, d_model=512, out_channels=7):
        super().__init__()
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(d_model, 256, 4, 2, 1),  # 16 x 16 -> 32 x 32
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 32 x 32 -> 64 x 64
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 64 x 64 -> 128 x 128
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 128 x 128 -> 256 x 256
        #     nn.ReLU(),
        #     nn.Conv2d(32, out_channels, 3, padding=1)
        # )
        self.decoder = nn.Sequential(
            nn.Conv2d(d_model, 256 * 4, 3, padding=1),
            nn.PixelShuffle(2),  # 16 x 16 -> 32 x 32
            nn.GELU(),
            SkipConnection(256, 128, scale_factor=2),  # 32 x 32 -> 64 x 64
            SkipConnection(128, 64, scale_factor=2),  # 64 x 64 -> 128 x 128
            nn.Conv2d(64, 32 * 4, 3, padding=1),
            nn.PixelShuffle(2),  # 128 x 128 -> 256 x 256
            nn.GELU(),
            nn.Conv2d(32, out_channels, 1)
        )

    def forward(self, x):
        # x: (B, H_patch, W_patch, d_model)
        x = self.decoder(x)  # (B, d_model, H_patch, W_patch)

        return x  # (B, 7, 256, 256)


class ViTTimeSeriesModel(nn.Module):
    def __init__(self, img_size=256, patch_size=16):
        super().__init__()
        self.spatial_vit = SpatialViT(img_size=img_size, patch_size=patch_size)
        self.temporal_transformer = TemporalTransformer()
        self.decoder = Decoder()
        self.patch_size = patch_size

    def forward(self, x):
        # x: (B, C=7, T=46, H, W)
        B, C, T, H, W = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        patch_H, patch_W = H // self.patch_size, W // self.patch_size
        num_patches = patch_H * patch_W

        # 提取每个时间步的空间特征
        # spatial_features = []
        # for t in range(T):
        #     frame = x[:, :, t, :, :]  # (B, C, H, W)
        #     features = self.spatial_vit(frame)  # (B, num_patches, d_model)
        #     spatial_features.append(features)
        # spatial_features = torch.stack(spatial_features, dim=2)  # (B, num_patches, T, d_model)

        # # 时间序列处理
        # x = rearrange(spatial_features, "b n t d -> (b n) t d")  # (B * num_patches, T, d_model)
        # x = self.temporal_transformer(x)  # (B * num_patches, T, d_model)
        # x = x.mean(dim=1)  # 使用平均作为时间步的输出
        # x = rearrange(x, "(b n) d -> b n d", b=B, n=num_patches)  # (B, num_patches, d_model)

        # 时间步统一处理
        spatial_features = self.spatial_vit(x)  # (B * T, num_patches, D)
        spatial_features = rearrange(
            spatial_features, "(b t) n d -> (b n) t d",
            b=B, t=T
        )

        temporal_features = self.temporal_transformer(spatial_features)  # (B * num_patches, D)

        # 重组特征图并解码
        x = rearrange(temporal_features, "(b n) d -> b (n d)", b=B, n=num_patches)
        x = rearrange(x, "b (h w d) -> b h w d", h=patch_H, w=patch_W, d=512)
        x = x.permute(0, 3, 1, 2)
        x = self.decoder(x)  # (B, C, H, W)

        return x
