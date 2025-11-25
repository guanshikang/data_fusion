# -*- encoding: utf-8 -*-
"""
@type: module

@brief: calculate validation statistical indicators.

@author: guanshikang

Created on Wed Oct 15 21:57:51 2025, HONG KONG
"""
import os
import numpy as np
import pandas as pd
from osgeo import gdal
from sklearn.metrics import r2_score as R2
from sklearn.metrics import root_mean_squared_error as RMSE
from sklearn.metrics import mean_absolute_error as MAE
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM


class DataEvaluation:
    def __init__(self):
        pass

    @staticmethod
    def SAM(reference_spectrum, target_spectrum):
        """
        Calculate the spectral angle between two spectra.

        Parameters:
        - reference_spectrum: numpy array, the reference spectrum (e.g., endmember).
        - target_spectrum: numpy array, the target spectrum to compare.

        Returns:
        - angle: float, spectral angle in radians.
        """
        # Ensure inputs are numpy arrays
        reference_spectrum = np.array(reference_spectrum)
        target_spectrum = np.array(target_spectrum)

        # Compute dot product and norms
        dot_product = np.dot(reference_spectrum, target_spectrum)
        norm_ref = np.linalg.norm(reference_spectrum)
        norm_target = np.linalg.norm(target_spectrum)

        # Calculate the spectral angle
        cos_theta = dot_product / (norm_ref * norm_target)
        cos_theta = np.clip(cos_theta, -1, 1)  # Avoid numerical issues
        angle = np.arccos(cos_theta)

        return angle

    @staticmethod
    def ERGAS(reference, fused, resolution_ratio):
        """
        Calculate the ERGAS metric.

        Parameters:
            reference (numpy.ndarray): Reference image (e.g., high-resolution ground truth).
            fused (numpy.ndarray): Fused or processed image.
            resolution_ratio (float): Ratio of the spatial resolutions (e.g., high-res/low-res).

        Returns:
            float: ERGAS value.
        """
        if reference.shape != fused.shape:
            raise ValueError("Reference and fused images must have the same dimensions.")

        # Flatten the images along the spectral bands
        bands = reference.shape[1]
        ergas_sum = 0

        for band in range(bands):
            ref_band = reference[:, band, :, :]
            fused_band = fused[:, band, :, :]
            # Mean of the reference band
            mean_ref = np.mean(ref_band)
            # Root Mean Square Error (RMSE) for the band
            rmse = np.sqrt(np.mean((ref_band - fused_band) ** 2))
            # Accumulate the ERGAS numerator
            ergas_sum += (rmse / mean_ref) ** 2

        # Final ERGAS calculation
        ergas = resolution_ratio * np.sqrt(ergas_sum / bands)
        return ergas

    def cal_npz(self, file_dir, file_name):
        """
        calculate statistical indicators for per band with nc format.

        Args:
            file_dir (str): e.g. "/fossfs/skguan/output/data_fusion/val_files"
            file_path (str): e.g. "val_result_fold0_swin(free_cloud).npz"
        """
        file_path = os.path.join(file_dir, file_name)

        data = np.load(file_path)
        label_data = data['label']
        if label_data.ndim == 1:
            label_data = label_data.reshape(-1, 6, 256, 256)
        pred_data = data['pred']
        if pred_data.ndim == 1:
            pred_data = pred_data.reshape(-1, 6, 256, 256)

        self._calculate(label_data, pred_data)

    def cal_tif(self, file_dir, ref_csv):
        """
        calculate statistical indicators for per band with tif format.

        Args:
            file_dir (str): e.g. "/fossfs/skguan/output/data_fusion/benchmark/ESTARFM"
            ref_csv (str): e.g. "/fossfs/skguan/data_fusion/dataset_23lr_c0n1.csv"

            ref_csv is used to aviod same output_name (more than 1 sites) cauising
            mismatch between predicted data and label.
        """
        df = pd.read_csv(ref_csv, usecols=["train", "val"])

        label_data = []
        pred_data = []
        files = os.listdir(file_dir)
        for file in files:
            temp_file = os.path.join(file_dir, file)
            ref_tfile = df['train'].dropna()[df['train'].dropna().str.contains(file)]
            ref_vfile = df['val'].dropna()[df['val'].dropna().str.contains(file)]
            if len(ref_tfile) + len(ref_vfile) == 1:
                temp_data = gdal.Open(temp_file).ReadAsArray()
                ref_file = ref_tfile.values or ref_vfile.values
                ref_data = gdal.Open(ref_file[0]).ReadAsArray()
                band_order = [1, 2, 3, 4, 5, 6]
                ref_data = ref_data[band_order, ...]

                pred_data.append(temp_data)
                label_data.append(ref_data)

        label_data = np.array(label_data)
        pred_data = np.array(pred_data)

        self._calculate(label_data, pred_data)

    def _calculate(self, label_data, pred_data):

        labels = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]

        sam = DataEvaluation.SAM(label_data.reshape(-1,), pred_data.reshape(-1,))
        ergas = DataEvaluation.ERGAS(label_data, pred_data, 1.0)
        print(f"SAM: {sam:.4f}, ERGAS: {ergas:.4f}")

        for i, label_name in enumerate(labels):
            label = label_data[:, i, :, :]
            pred = pred_data[:, i, :, :]
            psnr = PSNR(label, pred, data_range=1)
            ssim = SSIM(label, pred, data_range=1)
            label_list = label.reshape(-1,)
            pred_list = pred.reshape(-1,)
            r2 = R2(label_list, pred_list)
            rmse = RMSE(label_list, pred_list)
            mae = MAE(label_list, pred_list)
            print(f"{label_name}, R2: {r2:.3f}, RMSE: {rmse:.4f}, MAE: {mae:.3f}, "
                f"PSNR: {psnr:.3f}, SSIM: {ssim: .3f}")

def main():
    """Comment the other variable for one operation [file_name] | [ref_csv]"""
    file_dir = "/lustre1/g/geog_geors/skguan/data_fusion/output/data_fusion/val_files"
    file_name = "val_result_swin(tanh).npz"
    # ref_csv = "/fossfs/skguan/data_fusion/dataset_23lr_c0n1.csv"
    dp = DataEvaluation()
    try:
        file_name
        dp.cal_npz(file_dir, file_name)
    except:
        try:
            ref_csv
            dp.cal_tif(file_dir, ref_csv)
        except:
            print("Exit with no operations.")


if __name__ == "__main__":
    main()
