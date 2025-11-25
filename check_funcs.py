import os
import re
import glob
import shutil
import numpy as np
import pandas as pd
from osgeo import gdal
from sklearn.metrics import r2_score as R2
from sklearn.metrics import root_mean_squared_error as RMSE
from sklearn.metrics import mean_absolute_error as MAE
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from FunctionalCode.CommonFuncs import CommonFuncs

def cross_tile():
    files = glob.iglob("/fossfs/skguan/data_fusion/labels/*/*.tif")
    for file in files:
        data = gdal.Open(file).ReadAsArray()
        if np.any(data[5:7, :, :] == 0):
            os.remove(file)
            print(f"{file} has been reomved.")
    print("All process done.")

def empty_folder():
    file_dir = "/fossfs/skguan/data_fusion/landsat"
    sites = os.listdir(file_dir)
    for site in sites:
        files = glob.glob(os.path.join(file_dir, site, "*.tif"))
        if len(files) == 0:
            file_path = os.path.join(file_dir, site)
            shutil.rmtree(file_path)
            print(f"{file_path} has been removed.")

class DataProcess():
    def __init__(self):
        pass

    def SAM(self, reference_spectrum, target_spectrum):
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

    def ERGAS(self, reference, fused, resolution_ratio):
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
    def band_calculation(self):
        file_dir = "/fossfs/skguan/output/data_fusion/val_files"
        file_path = os.path.join(file_dir, "val_result_fold0_swin(random_dataset).npz")
        label_max = np.array([65454., 65454., 65455., 65455., 65454., 65455.])
        label_min = np.array([0., 0., 0., 0., 0., 0.])

        data = np.load(file_path)
        label_data = data['label'] * 65455.0
        pred_data = data['pred'] * 65455.0

        # label_data = np.power(label_data, 3)
        # pred_data = np.power(pred_data, 3)
        # label_data = label_data * (label_max[:, None, None] - label_min[:, None, None]) + label_min[:, None, None]
        # pred_data = pred_data * (label_max[:, None, None] - label_min[:, None, None]) + label_min[:, None, None]
        label_data = label_data * 2.75e-5 - 0.2
        pred_data = pred_data * 2.75e-5 - 0.2

        labels = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]

        sam = self.SAM(label_data.reshape(-1,), pred_data.reshape(-1,))
        ergas = self.ERGAS(label_data, pred_data, 1.0)
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

def same_patch_check():
    cf = CommonFuncs()
    file_dir = "/fossfs/skguan/data_fusion/landsat"
    site_names = os.listdir(file_dir)
    point_dict = {
        "site_id": [""] * len(site_names),
        "lon_ul": [0.0] * len(site_names),
        "lat_ul": [0.0] * len(site_names)
    }
    file_path = map(
        lambda x: os.path.join(x, os.listdir(x)[1]), map(
            lambda y: os.path.join(file_dir, y), site_names)
    )
    for i, file in enumerate(file_path):
        min_x, max_y, _, _ = cf.get_extent(file)
        site_name = file.split("/")[-2]
        point_dict['site_id'][i] = site_name
        point_dict['lon_ul'][i] = min_x
        point_dict['lat_ul'][i] = max_y
    df = pd.DataFrame(point_dict)
    df = df.groupby([df['lon_ul'], df['lat_ul']]).aggregate({'site_id': 'first'})
    df.to_csv("/fossfs/skguan/data_fusion/check.csv")

if __name__ == "__main__":
    # cross_tile()
    # empty_folder()
    dp = DataProcess()
    dp.band_calculation()
    # same_patch_check()
