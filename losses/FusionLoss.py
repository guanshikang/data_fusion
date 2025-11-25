# -*- encoding: utf-8 -*-
"""
@type: module

@brief: comprehensive fusion loss.

@author: guanshikang

Created on Fri Nov 07 17:36:01 2025, HONG KONG
"""
import sys
sys.path.append(sys.path[0] + "/losses")
import torch

from smooth_loss import SmoothLoss
from metrics import SSIM_band
from edge_consistent_loss import EdgeConsistentLoss
from physical_guided_loss import PhysicalGuidedLoss
from spectral_angle_loss import SpectralAngleLoss
from spectral_feature_consistent_loss import SpectralFeatureConsistentLoss

class FusionLoss:
    def __init__(self, weights=None):
        self.weights = weights or {
            'reconstruction': 1.0,  # basic reconstrcution loss, most important.
            'ssim_loss': 1.0,  # SSIM loss.
            'edge_loss': 0.05,  # spatial structure to retain edge information.
            'spectral_loss': 0.2,   # spectral consistent.
            'smooth_loss': 0.05,  # smooth noise for non-edge areas.
            'spectral_angle': 0.05,  # spectral angle.
        }
        self.reconstruction = torch.nn.MSELoss()
        self.edge_loss = EdgeConsistentLoss(belta=0.0)
        self.spectral_loss = SpectralFeatureConsistentLoss(belta=0.0)
        self.physical_loss = PhysicalGuidedLoss()
        self.spectral_angle_loss = SpectralAngleLoss()

    def __call__(self, logits, label, modis_A=None, modis_Q=None):
        return self.forward(logits, label, modis_A, modis_Q)

    def forward(self, logits, label, modis_A=None, modis_Q=None):
        total_loss = 0.0

        # 1. pixel-level reconstruction loss.
        reconstruction = self.reconstruction(logits, label)
        total_loss += self.weights['reconstruction'] * reconstruction

        # 2. ssim loss.
        all_band_ssim = SSIM_band(logits, label)
        ssim_loss = 1 - all_band_ssim.mean()
        total_loss += self.weights['ssim_loss'] * ssim_loss

        # 3. edge-consistent
        if self.weights['edge_loss'] > 0:
            edge_loss = self.edge_loss(logits, label, modis_Q)
            total_loss += self.weights['edge_loss'] * edge_loss

        # 4. spectral feature consistent
        if self.weights['spectral_loss'] > 0:
            spectral_loss = self.spectral_loss(logits, label, modis_A)
            total_loss += self.weights['spectral_loss'] * spectral_loss

        # 5. smooth loss
        if self.weights['smooth_loss'] > 0:
            smooth_loss = SmoothLoss(logits)
            total_loss += self.weights['smooth_loss'] * smooth_loss

        # 6. spectral angle loss
        if self.weights['spectral_angle'] > 0:
            spectral_angle_loss = self.spectral_angle_loss(logits, label)
            total_loss += self.weights['spectral_angle'] * spectral_angle_loss

        return total_loss, all_band_ssim
