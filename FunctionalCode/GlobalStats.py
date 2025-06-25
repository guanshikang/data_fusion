# -*- encoding: utf-8 -*-
"""
@brief: Compute global stats.

@author: guanshikang

@type: script

Created on Thu Mar 20 17:58:57 2025, HONG KONG
"""
import numpy as np
from tqdm import tqdm


class ComputeStats:
    def __init__(self) -> None:
        pass

    def compute_global_stats(self, dataset, flag, channel_num=7) -> np.ndarray:
        pixel_count = np.zeros(channel_num, dtype=np.int64)
        channels_sum = np.zeros(channel_num, dtype=np.float32)
        channel_sq_sum = np.zeros(channel_num, dtype=np.float32)

        # temp_idx = [np.inf, -np.inf]
        for data in dataset:
            img = data[flag]
            img = np.squeeze(img, axis=0)
            img = np.where(img < -1, np.nan, img)
            valid_mask = ~np.isnan(img)
            valid_count = np.sum(valid_mask, axis=(1, 2, 3))
            img = np.nan_to_num(img, nan=-1.0)

            channels_sum += np.sum(img, axis=(1, 2, 3))
            channel_sq_sum += np.sum(img ** 2, axis=(1, 2, 3))
            pixel_count += valid_count

            # if flag == "landsat":
            #     idx = data['modis_idx']
            #     temp_idx[0] = idx[0] if idx[0] < temp_idx[0] else temp_idx[0]
            #     temp_idx[1] = idx[1] if idx[1] > temp_idx[1] else temp_idx[1]

        epsilon = 1e-8
        mean = channels_sum / np.maximum(pixel_count, 1)
        std = np.sqrt(
            (channel_sq_sum / np.maximum(pixel_count, 1))
            - mean ** 2 + epsilon
        )

        mean = mean.astype(np.float32)
        std = std.astype(np.float32)
        np.savez(f"./stats_{flag}.npz", mean=mean, std=std)

        return mean, std
