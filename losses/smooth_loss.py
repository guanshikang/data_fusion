# -*- encoding: utf-8 -*-
"""
@type: module

@brief: smooth loss to suppress noise.

@author: guanshikang

Created on Thu Nov 06 23:06:14 2025, HONG KONG
"""
import torch

def SmoothLoss(logits):
    grad_x = logits[:, :, 1:, :] - logits[:, :, :-1, :]
    grad_y = logits[:, :, :, 1:] - logits[:, :, :, :-1]
    weight_x = torch.exp(-torch.abs(grad_x.mean(dim=1, keepdim=True)) * 10)
    weight_y = torch.exp(-torch.abs(grad_y.mean(dim=1, keepdim=True)) * 10)

    smooth_loss_x = torch.mean(weight_x * grad_x ** 2)
    smooth_loss_y = torch.mean(weight_y * grad_y ** 2)
    return smooth_loss_x + smooth_loss_y
