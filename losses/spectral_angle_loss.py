# -*- encoding: utf-8 -*-
"""
@type: module

@brief: spectral angle loss for hyperspectral image fusion.

@author: guanshikang

Created on Mon Nov 17 11:32:12 2025, HONG KONG
"""
import torch

class SpectralAngleLoss:
    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def __call__(self, logits, label):
        return self.forward(logits, label)

    def forward(self, logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 4 or label.dim() != 4:
            raise ValueError(f"Input tensors must be matched, got logits: {logits.dim()}, label: {logits.dim()}")
        B, C, H, W = logits.shape
        logits_flat = logits.view(B, C, -1).permute(0, 2, 1)
        label_flat = label.view(B, C, -1).permute(0, 2, 1)

        # calculate dot product
        dot_product = torch.sum(logits_flat * label_flat, dim=2)  # [B, H*W]

        # calculate module length
        logits_norm = torch.norm(logits_flat, dim=2, p=2)  # [B, H*W]
        label_norm = torch.norm(label_flat, dim=2, p=2)    # [B, H*W]

        # calculate cosine similarity
        cos_similarity = dot_product / (logits_norm * label_norm + self.eps)
        cos_similarity = torch.clamp(cos_similarity, -1.0 + self.eps, 1.0 - self.eps)

        # calculate spectral angle（arc）
        spectral_angle = torch.acos(cos_similarity)  # [B, H*W]
        spectral_angle_loss = torch.mean(spectral_angle)

        return spectral_angle_loss

