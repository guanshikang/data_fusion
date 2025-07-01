# -*- encoding: utf-8 -*-
"""
@brief: ViT with Skip connection module.

@author: guanshikang

@type: script

Created on Sun May 18 15:23:11 2025, HONG KONG
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed3D(nn.Module):
    def __init__(self, img_size=256, time_steps=12, patch_size=16,
                 t_patch=2, in_chans=7, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.t_patch = t_patch
        self.num_patches = (time_steps // t_patch) * (img_size //
                                                      patch_size) ** 2
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
        x = x.view(B, -1, D)

        return x, (t_blk, h_blk, w_blk)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
            qkv = self.qkv(x).reshape(
                B, N, 3, self.num_heads, C // self.num_heads
            ).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            qkv = x.reshape(
                B, N, self.num_heads, C // self.num_heads
            ).permute(0, 2, 1, 3)
            q, k, v = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
            x = self.proj(x)
            x = self.proj_drop(x)

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


class TransformerBlock(nn.Module):
    def __init__(self, dim=768, num_heads=8, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_features=int(dim * mlp_ratio), drop=drop)

        # depthwise convolution
        self.dw_conv = nn.Conv3d(
            dim, dim,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            groups=dim
        )
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, h_blk=None, w_blk=None, cnn_branch=False):
        # original attention + MLP
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        # Depthwise Convolution Pathway
        if cnn_branch:
            B, N, C = x.shape
            x_reshaped = x.view(B, -1, h_blk, w_blk, C).permute(0, 4, 1, 2, 3)
            x_spatial = self.dw_conv(x_reshaped).squeeze(2)
            x_spatial = x_spatial.permute(0, 2, 3, 4, 1).view(B, N, C)
            x = x + self.norm3(x_spatial)

        return x


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
        self.channel_attn = nn.Sequential(
            nn.AdaptiveMaxPool3d(1),
            nn.Conv3d(out_channels, out_channels // 8, 1),
            nn.GELU(),
            nn.Conv3d(out_channels // 8, out_channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv3d(out_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        x = self.double_conv(x)
        ca = self.channel_attn(x)
        x = x * ca
        sa = self.spatial_attn(x)
        x = x * sa

        return x + residual


class ResiduleUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_conv = nn.ConvTranspose3d(
            in_channels, out_channels,
            kernel_size=(1, 4, 4),
            stride=(1, 2, 2),
            padding=(0, 1, 1)
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.double_conv = nn.Sequential(
            nn.Conv3d(out_channels, out_channels,
                      kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels,
                      kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(out_channels)
        )
        self.shortcut = nn.Identity()
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

        identity = self.shortcut(x_up)
        x_out = self.double_conv(x_up)
        x_out += identity

        return self.activation(x_out)


class Decoder(nn.Module):
    def __init__(self, embed_dim=768, img_size=256, t_patch=24,
                 patch_size=16, out_chans=7, skip_dims=[768, 768, 768, 768]):
        super().__init__()
        self.t_patch = t_patch
        self.patch_size = patch_size
        self.reassemble = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.GELU()
        )
        self.skip_features = [1024, 512, 256, 128]  # ! Original just has 3 features
        self.skip_fuse = nn.ModuleList([
            # nn.Sequential(  # * Original Feature Fusion
            #     nn.Conv3d(skip_dim, skip_target, kernel_size=1),
            #     nn.BatchNorm3d(skip_target),
            #     nn.GELU()
            # ) for skip_dim, skip_target in zip(skip_dims, self.skip_features)
            nn.Sequential(    # * Fine Feature Fusion
                nn.Conv3d(skip_dim, skip_dim,
                          kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                nn.BatchNorm3d(skip_dim),
                nn.GELU(),
                # nn.AdaptiveMaxPool3d(1),
                nn.Conv3d(skip_dim, skip_dim // 8, kernel_size=1),
                nn.GELU(),
                nn.Conv3d(skip_dim // 8, skip_dim, kernel_size=1),
                nn.Sigmoid(),
                nn.Conv3d(skip_dim, skip_target, kernel_size=1),
                nn.BatchNorm3d(skip_target),
                nn.GELU()
            ) for skip_dim, skip_target in zip(skip_dims, self.skip_features)
        ])

        # Feature Alignment
        self.align_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(skip_target, skip_target, kernel_size=1),
                nn.BatchNorm3d(skip_target),
                nn.GELU()
            ) for skip_target in self.skip_features
        ])

        self.residule_up = nn.ModuleList([
            ResiduleUpBlock(in_ch, out_ch)
            for in_ch, out_ch in zip(
                [1024 + self.skip_features[0],
                 512 + self.skip_features[1],
                 256 + self.skip_features[2],
                 128 + self.skip_features[3]],  # ! Original - 3
                [512, 256, 128, 64])
        ])
        self.feature_fusion_attn = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(skip_target, skip_target, kernel_size=1),
                nn.GELU(),
                nn.Conv3d(skip_target, skip_target, kernel_size=1),
                nn.Sigmoid()
            ) for skip_target in self.skip_features
        ])
        self.cross_scale_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(feat_dim, feat_dim // 2, kernel_size=1),
                nn.GELU(),
                nn.Conv3d(feat_dim // 2, feat_dim, kernel_size=1)
            ) for feat_dim in skip_dims
        ])

        self.up_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose3d(
                    in_channels=1024 + self.skip_features[0],
                    out_channels=512,
                    kernel_size=(1, 4, 4),
                    stride=(1, 2, 2),
                    padding=(0, 1, 1)
                ),
                nn.BatchNorm3d(512),
                nn.GELU()
            ),
            nn.Sequential(
                nn.ConvTranspose3d(
                    in_channels=512 + self.skip_features[1],
                    out_channels=256,
                    kernel_size=(1, 4, 4),
                    stride=(1, 2, 2),
                    padding=(0, 1, 1)
                ),
                nn.BatchNorm3d(256),
                nn.GELU()
            ),
            nn.Sequential(
                nn.ConvTranspose3d(
                    in_channels=256 + self.skip_features[2],
                    out_channels=128,
                    kernel_size=(1, 4, 4),
                    stride=(1, 2, 2),
                    padding=(0, 1, 1)
                ),
                nn.BatchNorm3d(128),
                nn.GELU()
            ),
            nn.Sequential(  # ! Original - 3
                nn.ConvTranspose3d(
                    in_channels=128 + self.skip_features[3],
                    out_channels=64,
                    kernel_size=(1, 4, 4),
                    stride=(1, 2, 2),
                    padding=(0, 1, 1)
                ),
                nn.BatchNorm3d(64),
                nn.GELU()
            )
        ])

        self.final_up = nn.Sequential(
            nn.ConvTranspose3d(
                64, 64,
                kernel_size=(1, 4, 4),
                stride=(1, 2, 2),
                padding=(0, 1, 1)
            ),
            nn.BatchNorm3d(64),
            nn.GELU()
        )

        self.conv = nn.Conv3d(64, out_chans,
                              kernel_size=(3, 1, 1),
                              stride=(1, 1, 1),
                              padding=(0, 0, 0))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, original_shape, thw_blocks, encoder_features):
        # encoder_features.reverse()
        B, _, _ = x.shape
        _, _, T, H, W = original_shape

        # 恢复时空块结构
        t_blk, h_blk, w_blk = thw_blocks
        x = self.reassemble(x)
        x = x.view(B, t_blk, h_blk, w_blk, -1)
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, t_blk, h_blk, w_blk)

        # 3D上采样
        for i, (up_block, skip_feat) in enumerate(zip(self.residule_up,
                                                      encoder_features)):
            skip_feat = skip_feat.reshape(B, t_blk, h_blk, w_blk, -1) \
                .permute(0, 4, 1, 2, 3)

            if i > 0:  # * CNN fused features
                prev_feat = encoder_features[i - 1].reshape(
                    B, t_blk, h_blk, w_blk, -1).permute(0, 4, 1, 2, 3)
                prev_feat = self.cross_scale_fusion[i](prev_feat)
                skip_feat = skip_feat + prev_feat

                del prev_feat

            _, _, _, current_h, current_w = x.shape
            skip_feat = F.interpolate(skip_feat,
                                      size=(t_blk, current_h, current_w),
                                      mode='trilinear')
            skip_feat = self.skip_fuse[i](skip_feat)
            skip_feat = skip_feat * self.feature_fusion_attn[i](skip_feat)  # * CNN gate control

            x = torch.cat([x, skip_feat], dim=1)
            x = up_block(x, skip=None)

        # 时间维度处理
        # x = self.final_up(x)
        x = self.conv(x)
        x = x.mean(dim=2)
        x = x[:, :, :H, :W]

        return self.sigmoid(x)


class SpatialDownScale(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.GELU()
        )
        self.maxpool = nn.MaxPool3d((1, 2, 2))

    def forward(self, x):
        x = self.double_conv(x)
        x = self.maxpool(x)

        return x


class InformationMerge(nn.Module):
    def __init__(self, img_size=18, time_steps=136, patch_size=16, t_patch=2,
                 in_channels=(2, 10), embed_dim=768):
        super().__init__()
        self.double_conv = SpatialDownScale(in_channels[0], 4 * in_channels[0])
        self.in_channels = 4 * in_channels[0] + in_channels[1]
        self.patch_embed = PatchEmbed3D(img_size, time_steps, patch_size,
                                        t_patch, self.in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

    def forward(self, x1, x2):
        x1 = self.double_conv(x1)
        x = torch.cat([x1, x2], dim=1)
        x, _ = self.patch_embed(x)
        x = x + self.pos_embed

        return x


class CrossAttentionFusion(nn.Module):
    """
    Cross Attention Fusion for Landsat and MODIS.
    # * Results did not perform as expected.
    """
    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4., drop_rate=0.):
        super().__init__()
        self.gamma1 = nn.Parameter(1e-4 * torch.ones(embed_dim))
        self.gamma2 = nn.Parameter(1e-4 * torch.ones(embed_dim))
        self.norm_main = nn.LayerNorm(embed_dim)
        self.norm_aux = nn.LayerNorm(embed_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=drop_rate, batch_first=True
        )
        self.attn_dropout = nn.Dropout(drop_rate)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(drop_rate)
        )
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, main_feature, aux_feature):
        norm_main = self.norm_main(main_feature)
        norm_aux = self.norm_aux(aux_feature)
        attn_output, _ = self.cross_attn(
            query=norm_main,
            key=norm_aux,
            value=norm_aux
        )
        attn_output = main_feature + self.gamma1 * self.attn_dropout(attn_output)
        gate = self.gate(torch.cat([attn_output, main_feature], dim=-1))
        gate_output = gate * attn_output + (1 - gate) * main_feature
        output = gate_output + self.gamma2 * self.ffn(self.norm_main(gate_output))

        return self.dropout(output)


class FeatureFusion(nn.Module):
    def __init__(self, main_dim, aux_dim, out_dim):
        super().__init__()
        self.main_fc = nn.Linear(main_dim, out_dim)
        self.aux_fc = nn.Linear(aux_dim, out_dim)

        self.gate = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.Sigmoid()
        )

    def forward(self, main_feat, aux_feat):
        main_proj = self.main_fc(main_feat)
        _, L, _ = main_proj.shape
        aux_proj = self.aux_fc(aux_feat)
        _, l, _ = aux_proj.shape
        if l > L:
            aux_proj = aux_proj[:, :L, :]  # 截断到主干相同长度
        else:
            pad_l = L - l
            aux_proj = F.pad(aux_proj, (0, 0, 0, pad_l))

        combined = torch.cat([main_proj, aux_proj], dim=-1)
        gate = self.gate(combined)
        fused = gate * main_proj + (1 - gate) * aux_proj

        return fused


class ViT_Skip(nn.Module):
    def __init__(self, main_size=256, main_steps=12, main_spatch=16,
                 main_tpatch=2, main_inchans=7, aux_size=18, aux_steps=136,
                 aux_spatch=16, aux_tpatch=2, aux_inchans=(2, 10),
                 embed_dim=768, depth=12, num_heads=12, depth_wise=[3, 6, 9],
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., dropout=0., out_chans=7):
        super().__init__()
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim
        self.depth_wise = depth_wise
        self.patch_embed = PatchEmbed3D(main_size, main_steps, main_spatch,
                                        main_tpatch, main_inchans, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        # Positional Embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.main_blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate
            ) for _ in range(depth)
        ])

        self.aux_blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate
            ) for _ in range(depth)
        ])

        self.encoder_features = []

        # CNN Enhance Module
        self.cnn_blocks = nn.ModuleList()
        self.fusion_gates = nn.ModuleList()
        for i in range(depth):
            if i in depth_wise:
                self.cnn_blocks.append(CNNFeatureEnhancement(embed_dim))
                self.fusion_gates.append(
                    nn.Sequential(
                        nn.Linear(embed_dim, embed_dim // 4),
                        nn.GELU(),
                        nn.Linear(embed_dim // 4, 1),
                        nn.Sigmoid()
                    )
                )
            else:
                self.cnn_blocks.append(nn.Identity())

        # Auxiliary Information Flow
        self.info_merge = InformationMerge(
            aux_size, aux_steps, aux_spatch, aux_tpatch, aux_inchans, embed_dim
        )

        # Information Flow Fusion Module (plus directly)
        self.feature_fusion = FeatureFusion(
            main_dim=embed_dim,
            aux_dim=embed_dim,
            out_dim=embed_dim
        )

        # # Cross Attention Fusion Module
        # # * Results did not perform as expected.
        # self.cross_attn_fusion = nn.ModuleList([
        #     CrossAttentionFusion(
        #         embed_dim, num_heads, mlp_ratio, drop_rate
        #         ) for _ in range(3)
        # ])

        # image reconstruction
        self.decoder = Decoder(embed_dim, main_size, main_tpatch, main_spatch,
                               out_chans, skip_dims=[embed_dim] * 4)

    def forward(self, x, x1, x2):
        self.encoder_features = []
        original_shape = x.shape
        x, thw_blocks = self.patch_embed(x)
        t_blk, h_blk, w_blk = thw_blocks
        B, _, D = x.shape
        self.encoder_features.append(x.clone())  # store initial features

        # Resize the positional embeddings in case
        # they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            other_pos_embed = pos_embed.unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(h_blk, w_blk),
                                          mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        x1 = self.info_merge(x1, x2)
        del x2
        for i, (main_blk, aux_blk, cnn_blk) in enumerate(zip(
            self.main_blocks, self.aux_blocks, self.cnn_blocks
        )):
            x = main_blk(x, h_blk, w_blk, cnn_branch=True)
            x1 = aux_blk(x1)
            if i in self.depth_wise:  # Capture features at specific depths
                # self.encoder_features.append(x)  # * Original features
                x = self.feature_fusion(x, x1)
                get_idx = self.depth_wise.index(i)  # * CNN fused features
                x_3d = x.view(B, t_blk, h_blk, w_blk, -1).permute(0, 4, 1, 2, 3)
                cnn_feature = cnn_blk(x_3d)
                cnn_feature = cnn_feature.permute(0, 2, 3, 4, 1).view(B, -1, D)

                gate = self.fusion_gates[get_idx](x)
                fused_feature = gate * x + (1 - gate) * cnn_feature

                self.encoder_features.append(fused_feature)
                x = fused_feature

                del cnn_feature, fused_feature

        x = self.decoder(x, original_shape, thw_blocks, self.encoder_features)

        return x
