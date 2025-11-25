# -*- encoding: utf-8 -*-
"""
@type: module

@brief: spectral feature consistent loss.

@author: guanshikang

Created on Thu Nov 06 21:26:39 2025, HONG KONG
"""
import torch
import torch.nn.functional as F

class SpectralFeatureConsistentLoss:
    def __init__(self,
                 belta=0.2  # control weight of modis spectral features.
                 ):
        self.belta = belta

    def __call__(self, logits, label, modis):
        return self.forward(logits, label, modis)

    def forward(self,
                logits: torch.Tensor,  # predicted image
                label: torch.Tensor,  # ground truth.
                modis: torch.Tensor=None  # modis image, MOD09A1 should be give priority.
                ):
        logits_features = torch.cat([
            logits.mean(dim=(2, 3)),
            logits.std(dim=(2, 3))
        ], dim=1)
        label_features = torch.cat([
            label.mean(dim=(2, 3)),
            label.std(dim=(2, 3))
        ], dim=1)
        modis_features = torch.zeros_like(logits_features)
        if modis is not None:
            modis_features = torch.cat([
                modis.mean(dim=(2, 3)),
                modis.std(dim=(2, 3))
            ], dim=1)
        spectral_fidelity = F.mse_loss(logits_features, label_features)
        spectral_consistency = F.huber_loss(logits_features,
                                            modis_features,
                                            delta=0.1)  # should not be far away from MODIS

        return (1 - self.belta) * spectral_fidelity + self.belta * spectral_consistency