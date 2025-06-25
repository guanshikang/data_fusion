# -*- encoding: utf-8 -*-
"""
@brief: spatiotemporal ViT (Have questions with PatchEmbed).

@author: guanshikang

@type: script

Created on Sat Mar 15 14:13:40 2025, HONG KONG
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed3D(nn.Module):
    def __init__(self, in_channels=7, t_patch=4, patch_size=16,
                 d_model=512, img_size=256):
        super().__init__()
        self.t_patch = t_patch
        self.patch_size = patch_size
        self.d_model = d_model
        self.img_size = img_size

        self.time_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=d_model // 2,
            kernel_size=(t_patch, 1, 1),
            stride=(t_patch, 1, 1),
        )
        self.space_conv = nn.Conv3d(
            in_channels=d_model // 2,
            out_channels=d_model,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
        )
        self.temporal_pos = nn.Parameter(torch.randn(1, 64, d_model))
        self.spatial_pos = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2, d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, C, T, H, W = x.shape

        # Step 1: 时间padding
        if T % self.t_patch != 0:
            pad_t = self.t_patch - (T % self.t_patch)
            x = F.pad(x, (0, 0, 0, 0, 0, pad_t))  # (B, C, T + pad_t)
            T = T + pad_t

        x = self.time_conv(x)
        x = self.space_conv(x)  # (B, d_model, t_blocks, H_blocks, W_blocks)
        B, D, t_blk, h_blk, w_blk = x.shape
        x = x.permute(0, 2, 3, 4, 1)  # (B, t_blocks, H_blocks, W_blocks, d_model)
        temporal_embed = self.temporal_pos[:, :t_blk].unsqueeze(2).unsqueeze(2)
        spatial_embed = self.spatial_pos.view(1, 1, h_blk, w_blk, D)
        x = x + temporal_embed + spatial_embed
        x = x.reshape(B, -1, D)  # (B, num_patches, d_model)

        return self.norm(x), (t_blk, h_blk, w_blk)

    def _get_pos_embed(self, t_blk, h_blk, w_blk, B, device):
        # 时间位置编码 (t_blk, D)
        time_ids = torch.arange(t_blk, device=device)
        time_embed = self.time_embed(time_ids)  # (t_blk, D)

        # 空间位置编码 (h_blk * w_blk, D)
        space_ids = torch.arange(h_blk * w_blk, device=device)
        space_embed = self.space_embed(space_ids)  # (h_blk * w_blk, D)

        # 合并时空编码 (t_blk, h_blk * w_blk, D)
        combined_embed = time_embed.unsqueeze(1) + space_embed.unsqueeze(0)
        combined_embed = combined_embed.view(-1, self.d_model)  # (t_blk * h_blk * w_blk, D)

        # 添加CLS位置编码 (1, D)
        cls_embed = self.cls_pos  # (1, D)
        full_embed = torch.cat([cls_embed, combined_embed], dim=0)  # (num_patches + 1, D)

        # 扩展至批次维度 (B, num_patches + 1, D)
        return full_embed.unsqueeze(0).expand(B, -1, -1)


class DOYEncoder(nn.Module):
    def __init__(self, d_model=128, t_patch=4):
        super().__init__()
        self.t_patch = t_patch
        self.periodic_enc = nn.Sequential(
            nn.Linear(2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 2)
        )
        self.attn = nn.MultiheadAttention(d_model // 2, 4, dropout=0.1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, doy):
        B, T = doy.shape
        if T % self.t_patch != 0:
            pad_t = self.t_patch - (T % self.t_patch)
            doy = F.pad(doy, (0, pad_t), value=1)  # (B, C, T + pad_t)
            T = T + pad_t
        doy_rad = 2 * math.pi * (doy.float() - 1) / 365.25
        periodic = torch.stack(
            [torch.sin(doy_rad),
             torch.cos(doy_rad),
             torch.sin(2 * doy_rad),
             torch.cos(2 * doy_rad)],
            dim=-1
        )
        periodic = self.periodic_enc(periodic)
        periodic = periodic.permute(1, 0, 2)
        attn, _ = self.attn(periodic, periodic, periodic)
        attn = attn.permute(1, 0, 2)

        return self.norm(torch.cat([attn, periodic], dim=-1))


class SpatioTemporalTransformer(nn.Module):
    def __init__(self, d_model=768, nhead=8, num_layers=6):
        super().__init__()
        self.blocks = nn.ModuleList([
            EfficientTransformerLayer(d_model, nhead) for _ in range(num_layers)
        ])
        self.gammas = nn.Parameter(torch.ones(num_layers))

    def forward(self, x, mask=None):
        residual = x
        for i, blk in enumerate(self.blocks):
            x = blk(x, mask)
            x = self.gammas[i] * x + residual
            residual = x

        return x


class EfficientTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.local_attn = nn.MultiheadAttention(d_model, nhead // 2, dropout=0.1, batch_first=True)
        self.global_attn = nn.MultiheadAttention(d_model, nhead // 2, dropout=0.1, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            # nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        local_x = x.permute(1, 0, 2)
        local_attn, _ = self.local_attn(
            local_x, local_x, local_x,
            attn_mask=self._create_local_mask(local_x)
        )
        local_attn = local_attn.permute(1, 0, 2)

        global_attn, _ = self.global_attn(
            x, x, x,
            attn_mask=mask.repeat_interleave(
                self.global_attn.num_heads, dim=0
            ) if mask is not None else None
        )

        x = self.norm1(x + local_attn + global_attn)
        x = self.norm2(x + self.ffn(x))

        return x

    def _create_local_mask(self, x, window=16):
        seq_len = x.size(1)
        mask = torch.ones(seq_len, seq_len)
        for i in range(seq_len):
            start = max(0, i - window // 2)
            end = min(seq_len, i + window // 2)
            mask[i, start:end] = 0

        return mask.bool().to(x.device)


class SpatioTemporalDecoder(nn.Module):
    def __init__(self, t_patch=4, patch_size=16, d_model=768, out_channels=7):
        super().__init__()
        self.t_patch = t_patch
        self.patch_size = patch_size

        # 特征重组
        self.reassemble = nn.Sequential(
            nn.Linear(d_model, d_model * t_patch // 2),
            nn.GELU()
        )

        self.up_sampling = nn.Sequential(
            # 时空分离上采样
            # Block 1: 1024 -> 512, 12 x 16 x 16 -> 12 x 32 x 32
            # nn.ConvTranspose3d(
            #     1536,
            #     768,
            #     kernel_size=(1, 3, 3),
            #     stride=(1, 2, 2),
            #     padding=(0, 1, 1),
            #     output_padding=(0, 1, 1)
            # ),
            nn.Upsample(scale_factor=(1, 2, 2),
                        mode='trilinear',
                        align_corners=True),
            nn.Conv3d(1024, 512, kernel_size=(1, 3, 3), padding=1),
            nn.BatchNorm3d(512),
            nn.GELU(),

            # Block 2: 512 -> 256, 12 x 32 x 32 -> 12 x 64 x 64
            # nn.ConvTranspose3d(
            #     768,
            #     384,
            #     kernel_size=(1, 3, 3),
            #     stride=(1, 2, 2),
            #     padding=(0, 1, 1),
            #     output_padding=(0, 1, 1)
            # ),
            nn.Upsample(scale_factor=(1, 2, 2),
                        mode='trilinear',
                        align_corners=True),
            nn.Conv3d(512, 256, kernel_size=(1, 3, 3), padding=1),
            nn.BatchNorm3d(256),
            nn.GELU(),

            # Block 3: 256 -> 128, 12 x 64 x 64 -> 24 x 128 x 128
            # nn.ConvTranspose3d(
            #     384,
            #     192,
            #     kernel_size=3,
            #     stride=2,
            #     padding=1,
            #     output_padding=1
            # ),
            nn.Upsample(scale_factor=(2, 2, 2),
                        mode='trilinear',
                        align_corners=True),
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.GELU(),

            # Block 4: 128 -> 64, 24 x 128 x 128 -> 48 x 256 x 256
            # nn.ConvTranspose3d(
            #     192,
            #     96,
            #     kernel_size=3,
            #     stride=2,
            #     padding=1,
            #     output_padding=1
            # ),
            nn.Upsample(scale_factor=(2, 2, 2),
                        mode='trilinear',
                        align_corners=True),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.GELU(),

            # Block 5: 64 -> 7, restore channel dimension.
            nn.Conv3d(d_model // 8, out_channels, kernel_size=1),
        )

    def forward(self, x, original_shape):
        x = self.reassemble(x)
        B, N, D = x.shape
        _, _, T, H, W = original_shape

        # 恢复时空块结构
        t_blk = (T + self.t_patch - 1) // self.t_patch
        h_blk = H // self.patch_size
        w_blk = W // self.patch_size
        x = x.view(B, t_blk, h_blk, w_blk, -1)
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, t_blk, h_blk, w_blk)

        # 3D上采样
        x = self.up_sampling(x)

        # 时间维度处理
        x = x[:, :, :T]  # 裁剪到原始时间长度
        x = x.mean(dim=2)  # 时间维度平均
        return x


class SpatioTemporalViT(nn.Module):
    def __init__(self, t_patch=4, patch_size=16, d_model=512):
        super().__init__()
        self.t_patch = t_patch
        self.patch_size = patch_size
        self.d_model = d_model
        self.patch_embed = PatchEmbed3D(
            t_patch=t_patch,
            patch_size=patch_size,
            d_model=d_model
        )
        self.fusion = nn.Linear(d_model + d_model // 2, d_model),
        self.doy_encoder = DOYEncoder(d_model // 2, 4)
        self.doy_proj = nn.Linear(d_model // 2, d_model)
        self.transformer = SpatioTemporalTransformer(d_model=d_model)
        self.decoder = SpatioTemporalDecoder(t_patch, patch_size, d_model, 7)
        self.drop = nn.Dropout(0.1)

        self.mask_proj = nn.Sequential(
            nn.Conv3d(7, 1, kernel_size=1),
            nn.MaxPool3d((t_patch, patch_size, patch_size)),
            nn.Sigmoid()
        )

    def forward(self, x, mask=None):
        # Step 0: 生成patch级别的掩码
        # patch_mask = self.mask_proj(mask)
        # patch_mask = (patch_mask < 0.5).squeeze(1)
        # patch_mask = patch_mask.view(x.size(0), -1)

        # Step 1: 分块嵌入
        original_shape = x.shape
        x, (t_blk, h_blk, w_blk) = self.patch_embed(x)  # (B, num_patches + 1, d_model)

        # Step 2: DOY编码
        # 编码按时间分块聚合
        # doy_embed = self.doy_encoder(doy)
        # doy_embed = F.interpolate(
        #     doy_embed.permute(0, 2, 1),
        #     size=x.size(1),
        #     mode='nearest'
        # ).permute(0, 2, 1)
        # x = self.fusion(torch.cat([x, doy_embed]), dim=-1)
        # x = self.drop(x)
        # doy_embed = self.doy_proj(doy_embed)  # (B, T, d_model)
        # doy_embed = doy_embed.reshape(
        #     original_shape[0],
        #     t_blk,
        #     self.t_patch,
        #     self.d_model
        # )
        # doy_embed = doy_embed.mean(dim=1)  # (B, t_blk, d_model)

        # 将DOY编码广播到时间分块
        # doy_embed = doy_embed.unsqueeze(2).unsqueeze(2)  # (B, t_blk, 1, 1, d_model)
        # doy_embed = doy_embed.expand(-1, -1, h_blk, w_blk, -1)  # (B, t_blk, h_blk, w_blk, d_model)
        # doy_embed = doy_embed.reshape(original_shape[0], -1, self.d_model)
        # x[:, 0] += doy_embed

        # Step 3: Transformer
        # batch_size = x.size(0)
        # seq_len = x.size(1)
        # attn_mask = torch.zeros(batch_size, seq_len, seq_len, device=x.device)
        # for b in range(batch_size):
        #     invalid_indices = torch.where(patch_mask[b])[0]
        #     attn_mask[b, :, invalid_indices] = -1e9
        #     attn_mask[b, invalid_indices, :] = -1e9
        x = self.transformer(x)  # (B, num_patches + 1, d_model)

        # Step 3: 解码到图像空间
        x = self.decoder(x, original_shape)

        return x  # (B, C, H, W)
