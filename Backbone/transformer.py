# -*- encoding: utf-8 -*-
"""
@brief: plain transformer.

@author: guanshikang

@type: script

Created on Tue Mar 04 18:04:19 2025, HONG KONG
"""
import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor
from torchsummary import summary
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchvision.models import VisionTransformer
from torchvision.transforms import Compose, Resize, ToTensor


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emd_size: int = 768, img_size: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv3d(in_channels, emd_size, kernel_size=patch_size, stride=patch_size),
            Rearrange("b c t (h) (w) -> b (h w) c t"),
            Reduce("b p c t -> b p c", "mean")
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emd_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emd_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, "() n e -> b n e", b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 512, num_heads: int = 8, dropout: float = 0.):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # 使用单个矩阵一次性计算keys, queries, values
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        # self.keys = nn.Linear(emb_size, emb_size)
        # self.queries = nn.Linear(emb_size, emb_size)
        # self.values = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        # keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        # values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.masked_fill(~mask, fill_value)
        scaling = self.emb_size ** (1 / 2)
        attention = F.softmax(energy, dim=-1) / scaling
        attention = self.dropout(attention)
        out = torch.einsum("bhal, bhlv -> bhav", attention, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)

        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size,
                    expansion=forward_expansion,
                    drop_p=forward_drop_p
                ),
                nn.Dropout(drop_p)
            ))
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(
            *[TransformerEncoderBlock(**kwargs) for _ in range(depth)]
        )


class RegressionHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, out_channels: int = 3):
        super().__init__(
            Reduce("b n e-> b e", reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, emb_size),
            Rearrange("b (c h w) -> b c h w", c=emb_size, h=1, w=1),
            nn.ConvTranspose2d(emb_size, int(emb_size / 2), 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(int(emb_size / 2), int(emb_size / 4), 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(int(emb_size / 4), int(emb_size / 8), 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(int(emb_size / 8), int(emb_size / 16), 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(int(emb_size / 16), int(emb_size / 32), 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(int(emb_size / 32), int(emb_size / 64), 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(int(emb_size / 64), int(emb_size / 128), 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(int(emb_size / 128), int(emb_size / 256), 4, 2, 1),
            nn.ReLU()
        )


class ViT(nn.Sequential):
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 emb_size: int = 768,
                 img_size: int = 224,
                 depth: int = 12,
                 **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            RegressionHead(emb_size, in_channels)
        )
