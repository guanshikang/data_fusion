# -*- encoding: utf-8 -*-
"""
@type: module

@brief: calculate statistical indicators with torch

@author: guanshikang

Created on Sat Oct 18 21:48:45 2025, HONG KONG
"""
import torch
import torch.nn.functional as F
import numpy as np

def PSNR(logits: torch.Tensor,  # predict image
         label: torch.Tensor,  # original image
         max_val: float=1.0,  # max value of data
         ) -> float:
    """PSNR. 10 * log10(MAX^2 / MSE)."""
    mse = torch.mean((logits - label) ** 2)
    psnr = 10 * torch.log10(max_val ** 2 / (mse + 1e-8))

    return psnr.item()

def gaussian_kernel(size=11, sigma=1.5, device='cuda'):
    coords = torch.arange(size, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    return g.outer(g).unsqueeze(0).unsqueeze(0)

def SSIM_band(logits: torch.Tensor,
              label: torch.Tensor,
              window_size: int=11,
              data_range: float=1.0
              ) -> torch.Tensor:
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    # Create Guassian Kernel
    window = gaussian_kernel(window_size, 1.5, logits.device)
    window = window.repeat(logits.shape[1], 1, 1, 1)

    mu1 = F.conv2d(logits, window, padding=window_size // 2, groups=logits.shape[1])
    mu2 = F.conv2d(label, window, padding=window_size // 2, groups=label.shape[1])

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(logits * logits, window, padding=window_size // 2, groups=logits.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(label * label, window, padding=window_size // 2, groups=label.shape[1]) - mu2_sq
    sigma12 = F.conv2d(logits * label, window, padding=window_size // 2, groups=logits.shape[1]) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean(dim=[2, 3])

def SAM(logits: torch.Tensor,
        label: torch.Tensor
        ) -> float:
    """Spectral Angle Mapper"""
    logits_flat = logits.reshape(logits.shape[0], logits.shape[1], -1)  # [B, C, H * W]
    label_flat = label.reshape(label.shape[0], label.shape[1], -1)

    # calculate dot product and norm
    dot_product = torch.sum(logits_flat * label_flat, dim=1)  # [B, H * W]
    logits_norm = torch.norm(logits_flat, dim=1)
    label_norm = torch.norm(label_flat, dim=1)

    # calculate cosine similarity and convert to angle (radians)
    cos_sim = dot_product / (logits_norm * label_norm + 1e-8)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)  # avoid errors
    sam_rad = torch.acos(cos_sim)

    # convert to angle (degrees) and average
    sam_deg = sam_rad * 180.0 / torch.pi

    return torch.mean(sam_deg).item()

def ERGAS(logits: torch.Tensor,
          label: torch.Tensor,
          ratio: float=1.0  # Ratio of the spatial resolutions (e.g., high-res/low-res).
          ) -> float:
    """Relative Global Error in Synthesis"""
    # RMSE and mean per band
    rmse_per_band = torch.sqrt(torch.mean((logits - label) ** 2, dim=[0, 2, 3]))
    mean_per_band = torch.mean(label, dim=[0, 2, 3])
    mean_per_band = torch.where(mean_per_band == 0, torch.ones_like(mean_per_band), mean_per_band)  # avoid divide zero

    # calculate relative error square per band
    relative_error_sq = (rmse_per_band / mean_per_band) ** 2

    # calculate ergas
    ergas = 100.0 * ratio * torch.sqrt(torch.mean(relative_error_sq))

    return ergas.item()

def batch_evaluation(logits: torch.Tensor,
                     label: torch.Tensor,
                     eval_dict: dict,
                     all_band_ssim: torch.Tensor,
                     compute_band: bool=False
                     ) -> dict:
    _, num_bands, _, _ = logits.shape
    # Compute statistical indicators per batch
    eval_dict['sum_y'] += torch.sum(label).item()
    eval_dict['sum_y2'] += torch.sum(label ** 2).item()
    eval_dict['sum_pred'] += torch.sum(logits).item()
    eval_dict['sum_pred2'] += torch.sum(logits ** 2).item()
    eval_dict['sum_y_pred'] += torch.sum(label * logits).item()
    eval_dict['n_pixels'] += label.numel()
    with torch.no_grad():
        batch_psnr = PSNR(logits, label)
        batch_ssim = torch.mean(all_band_ssim).item()
        sam = SAM(logits, label)
        ergas = ERGAS(logits, label)
        eval_dict['psnr_list'].append(batch_psnr)
        eval_dict['ssim_list'].append(batch_ssim)
        eval_dict['sam_list'].append(sam)
        eval_dict['ergas_list'].append(ergas)

        if compute_band:
            if "band_stats" not in eval_dict:
                eval_dict['band_stats'] = [{
                    'sum_y': 0., 'sum_y2': 0., 'sum_pred': 0.,
                    'sum_pred2': 0., 'sum_y_pred': 0., 'n_pixels': 0,
                    'psnr_list': [], 'ssim_list': []
                } for _ in range(num_bands)]

            for band in range(num_bands):
                band_logits = logits[:, band:band + 1]
                band_label = label[:, band:band + 1]
                eval_dict['band_stats'][band]['sum_y'] += torch.sum(band_label).item()
                eval_dict['band_stats'][band]['sum_y2'] += torch.sum(band_label ** 2).item()
                eval_dict['band_stats'][band]['sum_pred'] += torch.sum(band_logits).item()
                eval_dict['band_stats'][band]['sum_pred2'] += torch.sum(band_logits ** 2).item()
                eval_dict['band_stats'][band]['sum_y_pred'] += torch.sum(band_label * band_logits).item()
                eval_dict['band_stats'][band]['n_pixels'] += band_label.numel()

                band_psnr = PSNR(band_logits, band_label)
                band_ssim = torch.mean(all_band_ssim[:, band]).item()

                eval_dict['band_stats'][band]['psnr_list'].append(band_psnr)
                eval_dict['band_stats'][band]['ssim_list'].append(band_ssim)

    return eval_dict

def epoch_evaluation(eval_dict: dict) -> tuple[float, float, float, float]:
    ss_tot = eval_dict['sum_y2'] - (eval_dict['sum_y'] ** 2 / eval_dict['n_pixels'])
    ss_res = eval_dict['sum_y2'] - 2 * eval_dict['sum_y_pred'] + eval_dict['sum_pred2']
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    rmse = np.sqrt(ss_res / eval_dict['n_pixels'])
    psnr = np.mean(eval_dict['psnr_list'])
    ssim = np.mean(eval_dict['ssim_list'])
    sam = np.mean(eval_dict['sam_list'])
    ergas = np.mean(eval_dict['ergas_list'])

    return r2, rmse, psnr, ssim, sam, ergas

def band_evaluation(band_stats: list, band_names=None):
    if band_names is None:
        band_names = [f"band_{i + 1}" for i in range(len(band_stats))]

    results = {}
    for band, stats in enumerate(band_stats):
        ss_tot = stats['sum_y2'] - (stats['sum_y'] ** 2 / stats['n_pixels'])
        ss_res = stats['sum_y2'] - 2 * stats['sum_y_pred'] + stats['sum_pred2']
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        rmse = np.sqrt(ss_res / stats['n_pixels'])
        psnr = np.mean(stats['psnr_list']) if stats['psnr_list'] else 0.0
        ssim = np.mean(stats['ssim_list']) if stats['ssim_list'] else 0.0

        band_name = band_names[band]

        results[band_name] = {
            'r2': r2,
            'rmse': rmse,
            'psnr': psnr,
            'ssim': ssim
        }
    return results
