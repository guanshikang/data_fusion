# -*- encoding: utf-8 -*-
"""
@brief: Compute global stats.

@author: guanshikang

@type: class

Created on Thu Mar 20 17:58:57 2025, HONG KONG
"""
import os
import numpy as np
from tqdm import tqdm


class ComputeStats:
    def __init__(self) -> None:
        pass

    def compute_global_stats(self, dataset, flag, mode='std', category=None, save_dir=None, channel_num=7):
        if mode == "std":
            pixel_count = np.zeros(channel_num, dtype=np.int64)
            channels_sum = np.zeros(channel_num, dtype=np.float32)
            channel_sq_sum = np.zeros(channel_num, dtype=np.float32)

            # temp_idx = [np.inf, -np.inf]
            label_max = np.full((channel_num,), -np.inf)
            label_min = np.full((channel_num,), np.inf)
            for data in dataset:
                img = data[flag].numpy()
                img = np.squeeze(img, axis=0)
                img = np.where(img < -1.0, np.nan, img)
                valid_mask = ~np.isnan(img)
                valid_count = np.sum(valid_mask, axis=(1, 2, 3))
                img = np.nan_to_num(img, nan=-1.0)

                channels_sum += np.sum(img, axis=(1, 2, 3))
                channel_sq_sum += np.sum(img ** 2, axis=(1, 2, 3))
                pixel_count += valid_count

                if flag == "landsat":
                    label = data['label'].numpy()
                    label = np.squeeze(label, axis=0)
                    temp_max = np.max(label, axis=(1, 2, 3))
                    temp_min = np.min(label, axis=(1, 2, 3))
                    label_max = np.maximum(label_max, temp_max)
                    label_min = np.minimum(label_min, temp_min)

            epsilon = 1e-8
            mean = channels_sum / np.maximum(pixel_count, 1)
            std = np.sqrt(
                (channel_sq_sum / np.maximum(pixel_count, 1))
                - mean ** 2 + epsilon
            )

            mean = mean.astype(np.float32)
            std = std.astype(np.float32)
            save_dir = os.getcwd() if save_dir is None else save_dir
            np.savez(os.path.join(save_dir, f"stats_{flag}{category}.npz"),
                     mean=mean, std=std)
            np.savez(os.path.join(save_dir, f"stats_label{category}.npz"),
                     label_min=label_min, label_max=label_max)

            # if flag == "landsat":
            #     idx = data['modis_idx']
            #     temp_idx[0] = idx[0] if idx[0] < temp_idx[0] else temp_idx[0]
            #     temp_idx[1] = idx[1] if idx[1] > temp_idx[1] else temp_idx[1]

            return mean, std, label_max, label_min

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

            np.savez(os.path.join(save_dir, f"stats_label{category}.npz"),
                    max_value=max_value,
                    min_value=min_value)

            return max_value, min_value
