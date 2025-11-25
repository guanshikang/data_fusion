# -*- encoding: utf-8 -*-
"""
@type: module

@brief: Merge Temporal to reduce time series.

@author: guanshikang

Created on Sun Nov 23 21:25:25 2025, HONG KONG
"""
import torch
import torch.nn as nn
import torch.functional as F

class PatchMerging3D(nn.Module):
    """Patch Merging Layer for 3D data (reduce spatial resolution, increase channels)"""
    def __init__(self,
                 dim,
                 in_length,  # input time series length
                 norm_layer=nn.LayerNorm,
                 temporal_merge=False,
                 ):
        super().__init__()
        self.dim = dim
        self.in_length = in_length
        self.temporal_merge = temporal_merge
        self.group_merge = None
        if temporal_merge:
            out_length = (in_length + in_length % 2) // 2
            self.group_merge = TemporalGroupMerging(dim, in_length, out_length)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, xt_idx):
        """
        x: B, T, H, W, C
        """
        _, T, H, W, _ = x.shape
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2, 0, 0))
            H += H % 2
            W += W % 2
        t_blk = T
        if self.group_merge:
            x, t_blk = self.group_merge(x, xt_idx)
        x0 = x[:, :, 0::2, 0::2, :]
        x1 = x[:, :, 0::2, 1::2, :]
        x2 = x[:, :, 1::2, 0::2, :]
        x3 = x[:, :, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)  # B T H // 2 W // 2 4 * C
        h_blk = H // 2
        w_blk = W // 2

        x = self.norm(x)
        x = self.reduction(x)

        return x, t_blk, h_blk, w_blk


class TemporalGroupMerging(nn.Module):
    def __init__(self, dim, in_length, out_length, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.in_length = in_length
        # group stride equals to out_length (in_length is odd number)
        # or out_length + 1 (in_length is even number)
        self.stride = out_length if in_length % 2 == 1 else out_length + 1
        self.linear = nn.Linear(dim * out_length, dim)
        self.norm = norm_layer(dim * out_length)

    def forward(self, x, relative_index):
        # generate time order to handle boundary date. Training stage won't change time index.
        time_order = self._generate_time_order(list(range(x.size(2))), relative_index)
        x = x[:, :, time_order, :, :]
        group = []
        for i in range(self.stride):
            x0 = x[:, :, i:i + self.stride, :, :]
            sub_order = torch.tensor(time_order[i:i + self.stride])
            _, indices = torch.sort(sub_order)
            group.append(x0[:, :, indices, :, :])

        x = torch.cat(group, -1)
        x = self.norm(x)
        x = self.linear(x)

        return x, x.size(2)

    def _generate_time_order(self, time_order, relative_index):
        if relative_index:
            if len(relative_index) == 1:
                return self._single_move(time_order, relative_index)
            elif len(relative_index) == 2:
                return self._double_move(time_order, relative_index)
            else:
                raise ValueError("target date should not exceed two.")
        else:
            return time_order

    def _single_move(self, time_order, current_index):
        n = len(time_order)
        mid_index = n // 2  # middle position index

        if current_index == mid_index:
            return time_order.copy()

        result = time_order.copy()

        # steps and orientation for movement
        steps = abs(current_index - mid_index)
        direction = -1 if current_index > mid_index else 1  # -1 for left，1 for right

        for step in range(steps):
            left_available = current_index
            right_available = n - 1 - current_index
            # current position and distance with edge determine group size
            if direction == -1:
                group_size = 2 * right_available + 1
                # start index and end index for group
                group_start = current_index - (group_size // 2)
                group_end = current_index + (group_size // 2)

                if group_size > 1:
                    if group_size - 1 >= 0:
                        left_elem = result[group_start - 1]
                        result[group_start - 1:group_end + 1] = result[group_start:group_end + 1] + [left_elem]
                else:  # group_size == 1
                    if current_index - 1 >= 0:
                        result[current_index], result[current_index - 1] = result[current_index - 1], result[current_index]

            else:
                group_size = 2 * left_available + 1
                # start index and end index for group
                group_start = current_index - (group_size // 2)
                group_end = current_index + (group_size // 2)

                if group_size > 1:
                    if group_start + 1 < n:
                        right_elem = result[group_end + 1]
                        result[group_start:group_end + 2] = [right_elem] + result[group_start:group_end + 1]
                else:  # group_size == 1
                    if current_index + 1 < n:
                        result[current_index], result[current_index + 1] = result[current_index + 1], result[current_index]

        return result

    def _double_move(self, time_order, current_index):
        idx1, idx2 = current_index
        first_value, second_value = time_order[idx1], time_order[idx2]

        # Create virtual array, merge two target date to one element
        virtual_arr = []
        virtual_idx = -1
        virtual_value = "VIRTUAL"

        for i, value in enumerate(time_order):
            if i == idx1:
                virtual_arr.append(virtual_idx)
                virtual_idx = len(virtual_arr) - 1
            elif i == idx2:
                # skip the second element cause it has been merged to virtual element.
                continue
            else:
                virtual_arr.append(value)

        virtual_arr = self._single_move(virtual_arr, virtual_idx)

        # replace virtual element to original elements
        result = []
        for val in virtual_arr:
            if val == virtual_value:
                result.append(first_value)
                result.append(second_value)
            else:
                result.append(val)

        return result
