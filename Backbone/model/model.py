# -*- encoding: utf-8 -*-
"""
@type: module

@brief: 

@author: guanshikang

Created on Fri Nov 21 14:58:35 2025, HONG KONG
"""
import torch.nn as nn
from FeaturePyramidNetwork import FPN
from FeatureFusion import FeatureFusion
from CNNEnhancement import CNNFeatureEnhancement
from InfoMerge import SpatialDownScale, InformationMerge
from VedioSwinTransformer import PatchEmbed3D, PatchMerging3D, BasicLayer


class SwinTransformer(nn.Module):
    def __init__(self,
                 main_size=256,
                 main_steps=2,
                 main_spatch=4,
                 main_tpatch=2,
                 main_inchans=6,
                 aux_size=36,
                 aux_steps=3,
                 aux_inchans=(2, 9),
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 mlp_ratio=4.0,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 dropout=0.0,
                 out_chans=6,
                 window_sizes=[(2, 8, 8), (2, 8, 8), (2, 4, 4), (2, 4, 4)]
                ):
        super().__init__()
        self.depth = sum(depths)
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim
        # First Downsample
        self.first_conv = SpatialDownScale(main_inchans, embed_dim)
        # Patch embedding
        self.main_patch_embed = PatchEmbed3D(img_size=main_size,
                                             time_steps=main_steps,
                                             patch_size=main_spatch,
                                             t_patch=main_tpatch,
                                             in_chans=main_inchans,
                                             embed_dim=embed_dim)
        main_tblk = main_steps // main_tpatch
        main_hblk = main_size // main_spatch
        main_wblk = main_size // main_spatch
        self.main_resolution = (main_tblk, main_hblk, main_wblk)

        self.aux_resolution = (aux_steps, aux_size, aux_size)
        aux_tblk, aux_hblk, aux_wblk = self.aux_resolution

        # Hierarchical Swin Transformer Stages
        self.x_stages = nn.ModuleList()
        self.x1_stages = nn.ModuleList()
        self.encoder_features = []

        # Output dimensions per stage
        x_out_dims = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
        x1_out_dims = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
        x_temporal_merges = [False, False, False, False]  # Whether to merge temporal dimension
        x1_temporal_merges = [True, False, False, False]

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
                downsample=PatchMerging3D if i in [0, 1, 2] else None,
                temporal_merge=x1_temporal_merges[i],
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

            if x1_temporal_merges[i]:
                aux_tblk += aux_tblk % 2
                aux_tblk //= 2
            axu_hblk += aux_hblk % 2
            aux_hblk //= 2
            aux_wblk += aux_wblk % 2
            aux_wblk //= 2
            self.aux_resolution = (aux_tblk, aux_hblk, aux_wblk)

        # CNN Enhance Module
        dims = [embed_dim * 2, embed_dim * 4, embed_dim * 8, embed_dim * 8]
        self.cnn_enhance = nn.ModuleList([
            CNNFeatureEnhancement(dims[i]) for i in range(len(depths))
        ])

        self.info_merge = InformationMerge(aux_inchans, embed_dim)

        self.feature_fusion = nn.ModuleList([
            FeatureFusion(dims[i]) for i in range(len(depths))
        ])
        # Image Reconstrcution
        self.decoder = FPN(x_out_dims[-1], main_size,
                           main_tpatch, main_spatch, out_chans,
                           skip_dims=[embed_dim] + x_out_dims)

    def forward(self, x, x1, x2, xt_idx, x1t_idx):
        # Save 128 Features
        self.encoder_features = []
        self.encoder_features.append(self.first_conv(x))

        # Patch Embedding
        x, _ = self.main_patch_embed(x)
        B, _, _, _, C = x.shape
        self.encoder_features.append(x.clone())
        x = x.view(B, -1, C)  # Flatten to (B, N, C)

        # Process Auxiliary Branch
        x1 = self.info_merge(x1, x2)

        # Process through hierarchical stages
        for stage, (x_stage, x1_stage) in enumerate(zip(self.x_stages, self.x1_stages)):
            # Apply Swin Transformer Stage
            x, x_tblk, x_hblk, x_wblk = x_stage(x, xt_idx)
            x1, x1_tblk, x1_hblk, x1_wblk = x1_stage(x1, x1t_idx)

            # Apply CNN enhancement at specific stages
            if stage in [0, 1, 2]:
                # Convert to 3D for CNN Processing
                B, _, C = x1.shape
                x1_3d = x1.view(B, x1_tblk, x1_hblk, x1_wblk, -1).permute(0, 4, 1, 2, 3)
                cnn_feature = self.cnn_enhance[stage](x1_3d)
                cnn_feature = cnn_feature.permute(0, 2, 3, 4, 1)
                fused_feature = self.feature_fusion[stage](
                    x, (x_tblk, x_hblk, x_wblk),
                    cnn_feature.view(B, -1, C), (x1_tblk, x1_hblk, x1_wblk),
                    stage
                )
                self.encoder_features.append(fused_feature.view(B, x_tblk, x_hblk,
                                                                x_wblk, -1))
            if stage == 3:
                fused_feature = self.feature_fusion[stage](
                    x, (x_tblk, x_hblk, x_wblk),
                    x1, (x1_tblk, x1_hblk, x1_wblk),
                    stage
                )

        # Reconstruction Image
        x = self.decoder(fused_feature.view(B, -1, x.shape[-1]),
                         (x_tblk, x_hblk, x_wblk),
                         self.encoder_features[::-1])
        return x