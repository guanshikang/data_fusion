# -*- encoding: utf-8 -*-
"""
@type: module

@brief: edge consistentcy loss.

@author: guanshikang

Created on Thu Nov 06 20:35:15 2025, HONG KONG
"""
import torch
import torch.nn.functional as F

class EdgeConsistentLoss:
    def __init__(self, belta: float = 0.0):
        self.belta = belta
        self.sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

    def __call__(self, logits, label, modis=None):
        return self.forward(logits, label, modis)

    def compute_edges(self, x):
        device = x.device

        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)
        x_mean = x.mean(dim=1, keepdim=True)
        edges_x = F.conv2d(x_mean, sobel_x, padding=1)
        edges_y = F.conv2d(x_mean, sobel_y, padding=1)
        edges = torch.abs(edges_x) + torch.abs(edges_y)

        return edges

    def forward(self, logits, label, modis=None):
        logits_edges = self.compute_edges(logits)
        label_edges = self.compute_edges(label)

        edge_loss = F.mse_loss(logits_edges, label_edges)

        modis_loss = 0.0
        if modis is not None and self.belta > 0:
            modis_edges = self.compute_edges(modis)
            if modis_edges.shape[-2:] != logits.shape[-2:]:
                modis_edges = F.interpolate(modis_edges, size=logits.shape[-2:], mode='bilinear')
            modis_loss = self.belta * F.mse_loss(logits_edges, modis_edges)

        return edge_loss + modis_loss