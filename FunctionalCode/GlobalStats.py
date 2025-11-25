# -*- encoding: utf-8 -*-
"""
@brief: Compute global stats.

@author: guanshikang

@type: class

Created on Thu Mar 20 17:58:57 2025, HONG KONG
"""
import os
import numpy as np

class ComputeStats:
    def __init__(self) -> None:
        pass

    def compute_global_stats(self, dataset, flag, mode='std', category=None, save_dir=None, channel_num=7):
        if mode == "std":
            pixel_count = np.zeros(channel_num, dtype=np.int64)
            channels_sum = np.zeros(channel_num, dtype=np.float32)
            channel_sq_sum = np.zeros(channel_num, dtype=np.float32)

            for data in dataset:
                img = data[flag].numpy()
                if flag == "landsat":
                    valid_mask = data['valid_mask'].numpy()
                    img = np.where(valid_mask, img, np.nan)
                else:
                    img = np.where(img < -1.0, np.nan, img)
                    valid_mask = ~np.isnan(img)
                valid_count = np.sum(valid_mask, axis=(0, 2, 3, 4))
                channels_sum += np.nansum(img, axis=(0, 2, 3, 4))
                channel_sq_sum += np.nansum(img ** 2, axis=(0, 2, 3, 4))
                pixel_count += valid_count

            epsilon = 1e-8
            mean = channels_sum / np.maximum(pixel_count, 1)
            std = np.sqrt(
                (channel_sq_sum / np.maximum(pixel_count, 1))
                - mean ** 2 + epsilon
            )

            mean = mean.astype(np.float32)
            std = std.astype(np.float32)
            save_dir = os.getcwd() if save_dir is None else save_dir
            np.savez(os.path.join(save_dir, "stats", f"stats_{flag}_{category}.npz"),
                     mean=mean, std=std)

            return mean, std

        if mode == "min-max":
            max_value = np.full((channel_num,), -np.inf)
            min_value = np.full((channel_num,), np.inf)

            for data in dataset:
                img = data[flag].numpy()
                img = np.squeeze(img, axis=(0))
                temp_max = np.max(img, axis=(1, 2, 3))
                temp_min = np.min(img, axis=(1, 2, 3))
                max_value = np.maximum(max_value, temp_max)
                min_value = np.minimum(min_value, temp_min)

            np.savez(os.path.join(save_dir, "stats", f"stats_label{category}.npz"),
                    max_value=max_value,
                    min_value=min_value)

            return max_value, min_value
