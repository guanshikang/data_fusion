# -*- encoding: utf-8 -*-
"""
@brief: Hierarchical Swin Transformer for Surface Reflectance.

@author: guanshikang

@type: script

Created on Thu Oct 9 14:48:36 2025, HONG KONG

Second Version of FPN and PPM Decoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt
from functools import reduce
from operator import mul
from torch.utils.checkpoint import checkpoint


class PatchEmbed3D(nn.Module):
    def __init__(self, img_size=256, time_steps=12, patch_size=4, t_patch=2,
                 in_chans=7, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.t_patch = t_patch
        self.num_patches = (time_steps // t_patch) * (img_size // patch_size) ** 2
        self.proj = nn.Conv3d(in_chans, embed_dim,
                              kernel_size=(t_patch, patch_size, patch_size),
                              stride=(t_patch, patch_size, patch_size))

    def forward(self, x):
        B, _, T, _, _ = x.shape
        if T % self.t_patch != 0:
            pad_t = self.t_patch - (T % self.t_patch)
            x = F.pad(x, (0, 0, 0, 0, 0, pad_t))
            T = T + pad_t
            self.num_patches = (T // self.t_patch) * (self.img_size //
                                                      self.patch_size) ** 2

        x = self.proj(x)
        B, D, t_blk, h_blk, w_blk = x.shape
        x = x.permute(0, 2, 3, 4, 1)

        return x, (t_blk, h_blk, w_blk)


class PatchMerging3D(nn.Module):
    """Patch Merging Layer for 3D data (reduce spatial resolution, increase
    channels)"""
    def __init__(self, dim, norm_layer=nn.LayerNorm, temporal_merge=False, isreduction=False):
        super().__init__()
        self.dim = dim
        self.temporal_merge = temporal_merge
        if temporal_merge:
            # Merge in temporal dimension as well (2 x 2 x 2)
            out_dim = 4 * dim
            if isreduction:
                out_dim = 2 * dim
            self.reduction = nn.Linear(8 * dim, out_dim, bias=False)
            self.norm = norm_layer(8 * dim)
        else:
            # Only merge spatial dimensions (1 x 2 x 2)
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, T, H, W, C
        """
        B, T, H, W, C = x.shape
        pad_input = (H % 2 == 1) or (W % 2 == 1) or \
            (self.temporal_merge and T % 2 == 1)
        if pad_input:
            pad_t = 2 - (T % 2) if self.temporal_merge and T % 2 != 0 else 0
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2, 0, pad_t))
            T += pad_t
            H += H % 2
            W += W % 2
        if self.temporal_merge:
            x000 = x[:, 0::2, 0::2, 0::2, :]
            x001 = x[:, 0::2, 0::2, 1::2, :]
            x010 = x[:, 0::2, 1::2, 0::2, :]
            x011 = x[:, 0::2, 1::2, 1::2, :]
            x100 = x[:, 1::2, 0::2, 0::2, :]
            x101 = x[:, 1::2, 0::2, 1::2, :]
            x110 = x[:, 1::2, 1::2, 0::2, :]
            x111 = x[:, 1::2, 1::2, 1::2, :]
            x = torch.cat([x000, x001, x010, x011, x100, x101, x110, x111], -1)  # B T // 2, H // 2, W // 2, 8 * C
            t_blk = T // 2
        else:
            x0 = x[:, :, 0::2, 0::2, :]
            x1 = x[:, :, 0::2, 1::2, :]
            x2 = x[:, :, 1::2, 0::2, :]
            x3 = x[:, :, 1::2, 1::2, :]
            x = torch.cat([x0, x1, x2, x3], -1)  # B T H // 2 W // 2 4 * C
            t_blk = T // 1

        h_blk = H // 2
        w_blk = W // 2

        x = self.norm(x)
        x = self.reduction(x)

        return x, t_blk, h_blk, w_blk


class WindowAttention3D(nn.Module):
    """Window based multi-head self attention module for 3D data"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size= window_size  # (T, H, W)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Define Relative Position Bias Table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) *
                        (2 * window_size[1] - 1) *
                        (2 * window_size[2] - 1),
                        num_heads))

        relative_position_index = self._get_relative_index(window_size)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, num_heads, N, head_dim

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B_, num_heads, N, N

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                N, N, -1
            ).permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + \
                mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def _get_relative_index(self, window_size):
        coords_t = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(
            torch.meshgrid(
                [coords_t,
                coords_h,
                coords_w],
                indexing='ij')
        )  # 3, T, H, W
        coords_flatten = torch.flatten(coords, 1)  # 3, T * H * W
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        # Shift to start from 0
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * \
            (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # T * H * W, T * H

        return relative_position_index


class SwinTransformerBlock3D(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=(1, 7, 7),
                 shift_size=(0, 0, 0), mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # Ensure Shift Size is less than Window Size
        if any(s > w for s, w in zip(shift_size, window_size)):
            raise ValueError("Shift size should be less than window size.")
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_features=mlp_hidden_dim, drop=drop)

        attn_mask = self._create_attention_mask(
            shift_size,
            window_size,
            input_resolution
        )
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        T, H, W = self.input_resolution
        B, L, C = x.shape
        assert L == T * H * W, "Input feature has wrong size."

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, T, H, W, C)
        pad_u = pad_l = pad_t0 = 0
        window_size = self.window_size
        pad_t1 = (window_size[0] - T % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_u, pad_b, pad_t0, pad_t1))
        _, Tp, Hp, Wp, _ = x.shape

        # Cyclic Shift for Shifted Window Attention
        if any(i > 0 for i in self.shift_size):
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0],
                                              -self.shift_size[1],
                                              -self.shift_size[2]),
                                   dims=(1, 2, 3))
            attn_mask = self.attn_mask
        else:
            shifted_x = x
            attn_mask = None

        # Partition Windows
        # nW * B, window_size[0] * window_size[1] * window_size[2], C
        x_windows = window_partition(shifted_x, window_size)
        # nW * B, window_t * window_h * window_w, C
        x_windows = x_windows.view(-1, reduce(mul, window_size), C)
        # W-MSA/SW-MSA
        # nW * B, window_t * window_h * window_w, C
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C, )))
        # B, T, H, W, C
        shifted_x = window_reverse(attn_windows, window_size, Tp, Hp, Wp)

        # Reverse Cyclic Shift
        if any(i > 0 for i in self.shift_size):
            x = torch.roll(shifted_x, shifts=(self.shift_size[0],
                                              self.shift_size[1],
                                              self.shift_size[2]),
                           dims=(1, 2, 3))
        else:
            x = shifted_x
        if pad_t1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :T, :H, :W, :].contiguous()

        x = x.view(B, T * H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

    def _create_attention_mask(self, shift_size, window_size, input_resolution):
        if any(i > 0 for i in shift_size):
            # Calculate attention mask for SW-MSA
            T, H, W = input_resolution
            pad_u = pad_l = pad_t0 = 0
            pad_t1 = (window_size[0] - T % window_size[0]) % window_size[0]
            pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
            pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
            img_mask = torch.zeros((1, T, H, W, 1))
            img_mask = F.pad(img_mask,
                             (0, 0, pad_l, pad_r, pad_u, pad_b, pad_t0, pad_t1))
            t_slices = (slice(0, -window_size[0]),
                        slice(-window_size[0], -shift_size[0]),
                        slice(-shift_size[0], None))
            h_slices = (slice(0, -window_size[1]),
                        slice(-window_size[1], -shift_size[1]),
                        slice(-shift_size[1], None))
            w_slices = (slice(0, -window_size[2]),
                        slice(-window_size[2], -shift_size[2]),
                        slice(-shift_size[2], None))
            cnt = 0
            for t in t_slices:
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, t, h, w, :] = cnt
                        cnt += 1

            mask_windows = window_partition(img_mask, window_size)
            mask_windows = mask_windows.view(-1, reduce(mul, window_size))
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)). \
                masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        return attn_mask


def window_partition(x, window_size):
    """
    Args:
        x: B, T, H, W, C
        window_size (tuple[int]): window size (t, h, w)

    Returns:
        windows: (B * num_windows, window_size[0], window_size[1], window_size[2], C)
    """
    B, T, H, W, C = x.shape
    x = x.view(B,
               T // window_size[0], window_size[0],
               H // window_size[1], window_size[1],
               W // window_size[2], window_size[2],
               C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(
        -1, reduce(mul, window_size), C
    )

    return windows


def window_reverse(windows, window_size, T, H, W):
    """
    Args:
        windows: (B * num_windows, window_size[0], window_size[1], window_size[2], C)
        window_size (tuple[int]): window size.
        T (int): Temporal length.
        H (int): Height of image.
        W (int): Width of image.
    """
    B = int(windows.shape[0] /
            (T * H * W / (window_size[0] * window_size[1] * window_size[2])))
    x = windows.view(B,
                     T // window_size[0],
                     H // window_size[1],
                     W // window_size[2],
                     window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, T, H, W, -1)

    return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class BasicLayer(nn.Module):
    """
    A Basic Swin Transformer Layer for One Stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key,
                                   value. Default: True.
        qk_scale (float | None, optional): Override default qk scale of head_dim
                                           ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0.
        attn_drop (float, optional): Attention dropout rate. Default: 0.0.
        downsample (nn.Moudle | None, optional): Downsample layer at the end of
                                                 the layer. Default: None.
    """
    def __init__(self, dim, resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., downsample=None, temporal_merge=False,
                 isreduction=False):
        super().__init__()
        self.dim = dim
        self.resolution = resolution
        self.depth = depth
        self.temporal_merge = temporal_merge

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                input_resolution=resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else (window_size[0] // 2,
                                                           window_size[1] // 2,
                                                           window_size[2] // 2),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop
            ) for i in range(depth)
        ])

        if downsample is not None:
            self.downsample = PatchMerging3D(dim=dim, norm_layer=nn.LayerNorm,
                                             temporal_merge=temporal_merge,
                                             isreduction=isreduction)
        else:
            self.downsample = None

    def forward(self, x):
        T, H, W = self.resolution
        B, L, C = x.shape
        assert (L == T * H * W), "Input feature has wrong size."

        # Process through transformer blocks
        for i in range(self.depth):
            def checkpointed_block(x):
                return self.blocks[i](x)
            x = checkpoint(checkpointed_block, x, use_reentrant=False)

        # Downsample (patch merging) if needed
        if self.downsample is not None:
            # Reshape to 3D format for merging
            x_3d = x.view(B, T, H, W, -1)
            x, t_blk, h_blk, w_blk = self.downsample(x_3d)
            x = x.view(B, -1, x.shape[-1])  # Flatten to (B, N, C)
        else:
            t_blk, h_blk, w_blk = T, H, W

        return x, t_blk, h_blk, w_blk

class SpectralAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
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
    def __init__(self, in_channels, out_channels=None):
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
        self.spectral_attn = SpectralAttention(out_channels, reduction_ratio=2)
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
        return x + residual


class ResiduleUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temporal_up=True):
        super().__init__()
        if temporal_up:
            ksp = (4, 2, 1)
        else:
            ksp = (1, 1, 0)
        self.up_conv = nn.ConvTranspose3d(
            in_channels, out_channels,
            kernel_size=(ksp[0], 4, 4),
            stride=(ksp[1], 2, 2),
            padding=(ksp[2], 1, 1)
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x, skip=None):
        x_up = self.up_conv(x)
        x_up = self.bn1(x_up)

        if skip is not None:
            if skip.shape[-2:] != x_up.shape[-2:]:
                skip = F.interpolate(
                    skip,
                    size=x_up.shape[-2:],
                    mode='bilinear',
                    align_corners=True
                )
            x_up = torch.cat([x_up, skip], dim=1)

        return self.activation(x_up)


class PPM(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio=4,
                 pool_size=[(1, 1, 1), (2, 2, 2), (3, 3, 3), (6, 6, 6)]):
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
    def __init__(self, embed_dim=768, img_size=256, t_patch=24,
                 patch_size=16, out_chans=7, skip_dims=[96, 96, 192, 384, 768]):
        super().__init__()
        self.pyramid_dim = 96  # Fixed channel dimension for FPN features
        num_levels = len(skip_dims)

        # Lateral convolutions for the skip connections (deep to shallow)
        self.lateral_convs = nn.ModuleList()
        for dim in skip_dims[::-1]:
            self.lateral_convs.append(nn.Conv3d(dim, self.pyramid_dim, kernel_size=1))

        # Separate lateral for the final (bottom-most) encoder output x
        self.lateral_final = nn.Conv3d(embed_dim, self.pyramid_dim, kernel_size=1)

        # Smooth convolutions after each fusion (with BN and GELU for better convergence)
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
        self.final_conv = nn.Conv3d(self.pyramid_dim, out_chans, kernel_size=1)

    def forward(self, x, original_shape, thw_blocks, encoder_features):
        B, _, _ = x.shape
        _, _, T, H, W = original_shape
        t_blk, h_blk, w_blk = thw_blocks

        # Reshape and permute the bottom-most feature
        x = x.view(B, t_blk, h_blk, w_blk, -1).permute(0, 4, 1, 2, 3)  # (B, embed_dim, t_blk, h_blk, w_blk)

        # Fuse the bottom-most x with the deepest skip feature using lateral convs and addition
        deepest_skip = encoder_features[0].permute(0, 4, 1, 2, 3)
        p = self.lateral_final(x) + self.lateral_convs[0](deepest_skip)
        p = self.smooth_convs[0](p)

        # Collect all features in the top_down pathway
        pyramid_features = [p]

        # Top-down FPN pathway with upsampling and lateral additions
        for i in range(1, 5):
            # Determine temporal scale (only upscale time at the specific level)
            scale_t = 2 if i == 4 else 1
            # Upsample the previous pyramid level
            p = F.interpolate(p, scale_factor=(scale_t, 2, 2), mode='trilinear', align_corners=True)
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


class SpatialDownScale(nn.Module):
    """U-Net DownSample Module"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 2, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(out_channels // 2),
            nn.GELU(),
            nn.Conv3d(out_channels // 2, out_channels // 2, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(out_channels // 2),
            nn.GELU()
        )
        self.maxpool = nn.MaxPool3d((1, 2, 2))
        self.double_conv2 = nn.Sequential(
            nn.Conv3d(out_channels // 2, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
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
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.GELU()
        )
        self.up_scale = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear',
                                    align_corners=False)
        self.post_conv = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.up_scale(x)
        x = self.post_conv(x)

        return x


class InformationMerge(nn.Module):
    def __init__(self, in_channels=(2, 10), embed_dim=96):
        super().__init__()
        self.up_scale = SpatialUpScale(in_channels[1], embed_dim)
        self.res_conv = nn.Conv3d(in_channels[0], embed_dim, 1) if in_channels[0] != embed_dim else nn.Identity()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels[0], embed_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(embed_dim),
            nn.GELU(),
            nn.Conv3d(embed_dim, embed_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
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

class FeatureFusion(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.linear_fuse = nn.Linear(2 * out_dim, out_dim)
        self.conv = nn.Conv3d(out_dim, out_dim, (1, 2, 2))

    def forward(self, main_feat, main_resolution, aux_feat, aux_resolution):
        B, _, C = main_feat.shape
        x_tblk, x_hblk, x_wblk = main_resolution
        x = main_feat.view(B, x_tblk, x_hblk, x_wblk, -1).permute(0, 4, 1, 2, 3)
        x1_tblk, x1_hblk, x1_wblk = aux_resolution
        x1 = aux_feat.view(B, x1_tblk, x1_hblk, x1_wblk, -1).permute(0, 4, 1, 2, 3)
        x1 = self.conv(x1)
        x = torch.cat([x, x1], axis=1)
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(B, -1, C * 2)
        x = self.linear_fuse(x)

        return x


class SwinTransformer(nn.Module):
    def __init__(self, main_size=256, main_steps=12, main_spatch=4,
                 main_tpatch=2, main_inchans=7, aux_size=36, aux_steps=12,
                 aux_spatch=4, aux_tpatch=2, aux_inchans=(2, 10),
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., dropout=0., out_chans=7,
                 window_sizes=[(2, 8, 8), (2, 8, 8), (2, 4, 4), (2, 4, 4)]):
        super().__init__()
        self.depth = sum(depths)
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim
        # First Downsample
        self.first_conv = SpatialDownScale(main_inchans, embed_dim)
        # Patch embedding
        self.main_patch_embed = PatchEmbed3D(main_size, main_steps, main_spatch,
                                             main_tpatch, main_inchans,
                                             embed_dim)
        self.main_num_patches = self.main_patch_embed.num_patches
        main_tblk = main_steps // main_tpatch
        main_hblk = main_size // main_spatch
        main_wblk = main_size // main_spatch
        self.main_resolution = (main_tblk, main_hblk, main_wblk)

        aux_tblk = aux_steps
        aux_hblk = aux_size
        aux_wblk = aux_size
        self.aux_resolution = (aux_tblk, aux_hblk, aux_wblk)

        # Hierarchical Swin Transformer Stages
        self.x_stages = nn.ModuleList()
        self.x1_stages = nn.ModuleList()
        self.encoder_features = []

        # Output dimensions per stage
        x_out_dims = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
        x1_out_dims = [embed_dim, embed_dim, embed_dim * 4, embed_dim * 8]
        x_temporal_merges = [False, False, False, False]  # Whether to merge temporal dimension
        x1_temporal_merges = [False, True, True, False]
        x1_reduction = [False, False, True, False]

        # Build Stages
        for i in range(len(depths)):
            layer = BasicLayer(
                dim=x_out_dims[i],
                resolution=self.main_resolution,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_sizes[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                downsample=PatchMerging3D if i < len(depths) - 1 else None,
                temporal_merge=x_temporal_merges[i]
            )
            self.x_stages.append(layer)

            layer = BasicLayer(
                dim=x1_out_dims[i],
                resolution=self.aux_resolution,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_sizes[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                downsample=PatchMerging3D if i in [1, 2] else None,
                temporal_merge=x1_temporal_merges[i],
                isreduction=x1_reduction[i]
            )
            self.x1_stages.append(layer)

            # Update resolution for next stage
            if x_temporal_merges[i]:
                main_tblk += main_tblk % 2
                main_tblk //= 2
            main_hblk += main_hblk % 2
            main_hblk //= 2
            main_wblk += main_wblk % 2
            main_wblk //= 2
            self.main_resolution = (main_tblk, main_hblk, main_wblk)

            if i in [1, 2]:
                if x1_temporal_merges[i]:
                    aux_tblk += aux_tblk % 2
                    aux_tblk //= 2
                aux_hblk += aux_hblk % 2
                aux_hblk //= 2
                aux_wblk += aux_wblk % 2
                aux_wblk //= 2
                self.aux_resolution = (aux_tblk, aux_hblk, aux_wblk)

        # CNN Enhance Module
        dims = [embed_dim * 2, embed_dim * 4, embed_dim * 8, embed_dim * 8]
        self.cnn_enhance = nn.ModuleList([
            CNNFeatureEnhancement(dims[i]) for i in range(len(depths))
        ])
        self.fusion_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dims[i], dims[i] // 4),
                nn.GELU(),
                nn.Linear(dims[i] // 4, 1),
                nn.Sigmoid()
            ) for i in range(len(depths))
        ])

        self.info_merge = InformationMerge(aux_inchans, embed_dim)

        self.end_fusion = FeatureFusion(x_out_dims[-1])
        # Image Reconstrcution
        self.decoder = FPN(x_out_dims[-1], main_size,
                           main_tpatch, main_spatch, out_chans,
                           skip_dims=[embed_dim] + x_out_dims)

    def forward(self, x, x1, x2):
        # Save 128 Features
        self.encoder_features = []
        self.encoder_features.append(self.first_conv(x))

        # Patch Embedding
        original_shape = x.shape
        x, _ =  self.main_patch_embed(x)
        B, _, _, _, C = x.shape
        self.encoder_features.append(x.clone())
        x = x.view(B, -1, C)  # Flatten to (B, N, C)

        # Process Auxiliary Branch
        x1 = self.info_merge(x1, x2)

        # Process through hierarchical stages
        for i, (x_stage, x1_stage) in enumerate(zip(self.x_stages, self.x1_stages)):
            # Apply Swin Transformer Stage
            x, x_tblk, x_hblk, x_wblk = x_stage(x)
            x1, x1_tblk, x1_hblk, x1_wblk = x1_stage(x1)
            # Apply CNN enhancement at specific stages
            if i in [0, 1, 2]:
                # Convert to 3D for CNN Processing
                x_3d = x.view(B, x_tblk, x_hblk, x_wblk, -1).permute(0, 4, 1, 2, 3)
                cnn_feature = self.cnn_enhance[i](x_3d)
                cnn_feature = cnn_feature.permute(0, 2, 3, 4, 1)

                # Gate fusion
                gate = self.fusion_gates[i](x)
                fused_feature = gate * x + (1 - gate) * cnn_feature.view(
                    B, -1, cnn_feature.shape[-1]
                )
                self.encoder_features.append(fused_feature.view(B, x_tblk, x_hblk,
                                                                x_wblk, -1))
            if i == 3:
                x = self.end_fusion(x, (x_tblk, x_hblk, x_wblk),
                                    x1, (x1_tblk, x1_hblk, x1_wblk))
        x = self.decoder(x.view(B, -1, x.shape[-1]), original_shape,
                         (x_tblk, x_hblk, x_wblk), self.encoder_features[::-1])

        return x
