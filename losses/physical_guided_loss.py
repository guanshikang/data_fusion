# -*- encoding: utf-8 -*-
"""
@type: module

@brief: physical guided loss.

@author: guanshikang

Created on Fri Nov 07 15:07:39 2025, HONG KONG
"""
import torch
import torch.nn.functional as F

class PhysicalGuidedLoss():
    def __init__(self,
                 belta=0.0  # control weight of NDVI contribution.
                 ):
        self.belta = belta

    def __call__(self, logits):
        return self.forward(logits)

    def forward(self, logits):
        upper = F.relu(logits - 1.0)
        lower = F.relu(-logits)
        range_loss = upper.mean() + lower.mean()

        return range_loss
