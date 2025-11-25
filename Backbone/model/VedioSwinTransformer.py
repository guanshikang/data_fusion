# -*- encoding: utf-8 -*-
"""
@type: module

@brief: Video SwinTransformer.

@author: guanshikang

Created on Fri Nov 21 14:24:32 2025, HONG KONG
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
from operator import mul
from PatchMerging import PatchMerging3D


class PatchEmbed3D(nn.Module):
    def __init__(self,
                 img_size=256,
                 time_steps=2,
                 patch_size=4,
                 t_patch=2,
                 in_chans=6,
                 embed_dim=96
                 ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.t_patch = t_patch
        self.num_patches = (time_steps // t_patch) * (img_size // patch_size) ** 2
        self.proj = nn.Conv3d(in_chans,
                              embed_dim,
                              kernel_size=(t_patch, patch_size, patch_size),
                              stride=(t_patch, patch_size, patch_size))

    def forward(self, x):
        _, _, T, _, _ = x.shape
        if T % self.t_patch != 0:
            pad_t = self.t_patch - (T % self.t_patch)
            x = F.pad(x, (0, 0, 0, 0, 0, pad_t))
            T = T + pad_t
            self.num_patches = (T // self.t_patch) * \
                (self.img_size // self.patch_size) ** 2

        x = self.proj(x)
        _, _, t_blk, h_blk, w_blk = x.shape
        x = x.permute(0, 2, 3, 4, 1)

        return x, (t_blk, h_blk, w_blk)


class WindowAttention3D(nn.Module):
    """Window based multi-head self attention module for 3D data"""
    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.
                 ):
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
                        (2 * window_size[2] - 1), num_heads)
        )

        relative_position_index = self._get_relative_index(window_size)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, num_heads, N, head_dim

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B, num_heads, N, N

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1).permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + \
                mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
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
    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=(1, 7, 7),
                 shift_size=(0, 0, 0),
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.0,
                 attn_drop=0.0):
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
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop
        )

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
                     window_size[0],
                     window_size[1],
                     window_size[2],
                     -1
                     )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, T, H, W, -1)

    return x


class MLP(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 drop=0.0
                 ):
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
    def __init__(self,
                 dim,
                 resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.0,
                 attn_drop=0.0,
                 downsample=None,
                 temporal_merge=False
                 ):
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
            self.downsample = PatchMerging3D(dim=dim,
                                             in_length=resolution[0],
                                             norm_layer=nn.LayerNorm,
                                             temporal_merge=temporal_merge)
        else:
            self.downsample = None

    def forward(self, x, xt_idx):
        T, H, W = self.resolution
        B, L, C = x.shape
        assert (L == T * H * W), "Input feature has wrong size."

        # Process through transformer blocks
        for i in range(self.depth):
            x = self.blocks[i](x)

        # Downsample (patch merging) if needed
        if self.downsample is not None:
            # Reshape to 3D format for merging
            x_3d = x.view(B, T, H, W, -1)
            x, t_blk, h_blk, w_blk = self.downsample(x_3d, xt_idx)
            x = x.view(B, -1, x.shape[-1])  # Flatten to (B, N, C)
        else:
            t_blk, h_blk, w_blk = T, H, W

        return x, t_blk, h_blk, w_blk
