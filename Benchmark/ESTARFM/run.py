# -*- encoding: utf-8 -*-
"""
@brief: Data Loader for ESTARFM

@author: guanshikang

@type: script

Created on Sun Sep 28 21:03:58 2025, HONG KONG
"""
import os
import re
import tqdm
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
from osgeo import gdal
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score as r2
from sklearn.metrics import root_mean_squared_error as rmse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from ESTARFM import ESTARFM

SEASON = {
    'MAM': ["03", "04", "05"],
    'JJA': ["06", "07", "08"],
    'SON': ["09", "10", "11"],
    'DJF': ["12", "01", "02"]
}

class DataLoader:
    """
    训练数据加载器.

    Args:
        data_utils (class): parent class.
    """
    def __init__(self, landsat_path, modis_path, label_path, files,
                 out_channels=7, image_size=256, num_pairs=3, cloud_cover=30,
                 Landsat=None):
        self.landsat_path = landsat_path
        self.modis_path = modis_path
        self.label_path = label_path
        self.out_channels = out_channels
        self.image_size = image_size
        self.num_pairs = num_pairs
        self.cloud_cover = cloud_cover
        self.files = files
        self.site_names = self.__get_sitenames()
        self.Landsat = Landsat or "Single"

    def __get_sitenames(self):
        return [x.split("/")[-2] for x in self.files]

    def __standardization(self, data, flag):
        data = np.where(data < -100., -65535, data)
        if flag == "label":
            pass
        else:
            data = (data - self.stats[flag + "_mean"][:, None, None, None]) / \
                (self.stats[flag + "_std"][:, None, None, None] + 1e-8)

        return np.nan_to_num(data, nan=0.0)

    def __load_tiff(self, file_path, flag):
        data = np.empty(shape=(8, 0, self.image_size, self.image_size))
        file_path = list(file_path)
        file_path.sort()
        for file in file_path:
            img = gdal.Open(file).ReadAsArray()
            img = img.reshape(8, 1, self.image_size, self.image_size)
            img = self.__standardization(img, flag)
            data = np.concatenate((data, img), axis=1)

        return data

    def __filter_by_season(self, date, time_dst, units, calendar, same_year=False):
        target_month = date.month
        target_season = {
            k: v for k, v in SEASON.items()
            if str(target_month).zfill(2) in v
        }
        season = [x for x in target_season.keys()][0]
        months = [x for x in target_season.values()][0]
        ori_date = nc.num2date(time_dst.values, units=units,
                                calendar=calendar)
        target_year = date.year
        if same_year:
            syear, eyear = [target_year] * 2
            if season == "DJF":
                if target_month == 12:
                    eyear += 1
                else:
                    syear -= 1
            start_date = nc.date2num(
                datetime.strptime("%d%s" % (syear, months[0]),
                                "%Y%m"),
                units, calendar
            )
            end_date = nc.date2num(
                datetime.strptime("%d%d" % (eyear, int(months[-1]) + 1),
                                  "%Y%m"),
                units, calendar
            )
            temp_date = ori_date[
                (start_date <= time_dst.values) & (time_dst.values < end_date)
            ]
            idx = np.argwhere(np.isin(ori_date, temp_date))
            return [x[0] for x in idx]

        ori_month = list(map(
            lambda x: str(x.month).zfill(2),
            ori_date
        ))
        idx = np.argwhere(np.isin(ori_month, months))

        return [x[0] for x in idx]

    def __time_index(self, target_date, landsat_dst, modis_dst, landsat_mask, site_name):
        units = "days since 1970-01-01 00:00"
        calendar = "standard"
        date = nc.date2num(target_date, units=units, calendar=calendar)
        time_landsat = landsat_dst['time']

        season_idx_landsat = self.__filter_by_season(target_date, time_landsat,
                                                     units, calendar)
        season_landsat = time_landsat[season_idx_landsat]
        season_mask = landsat_mask[season_idx_landsat]
        length = 2 * self.num_pairs  # * Here is for the num of image pairs.
        least_idx = np.argmin(np.abs(season_landsat.values - date))
        idx = []
        for i, clear_data in enumerate(season_mask):
            clear_data = clear_data.values
            clear_ratio = np.sum(clear_data == 1) * 100 / \
                (clear_data.shape[0] * clear_data.shape[1])
            if clear_ratio >= 100 - self.cloud_cover:
                idx.append(i)
        idx = np.array(idx)
        if self.Landsat == "Union":
            target_idx = np.argwhere(idx == least_idx)
            idx = np.delete(idx, target_idx)
        if len(idx) == 0:
            return [], []
        temp_idx = np.argmin(np.abs(idx - least_idx))
        temp_len = len(idx)
        min_idx, max_idx = temp_idx - length // 2, temp_idx + length // 2
        if min_idx < 0:
            min_idx = 0
            max_idx = min_idx + length
        elif max_idx > temp_len - 1:
            max_idx = temp_len
            min_idx = max_idx - length
        idx = idx[min_idx:max_idx]

        # 目标日期的MODIS图像对
        time_modis = modis_dst['time']

        season_idx_modis = self.__filter_by_season(
            target_date, time_modis, units, calendar, same_year=True
        )
        mlength = length + 1
        while len(season_idx_modis) < mlength:  # MODIS needs include one more than Landsat
            season_idx_modis.append(season_idx_modis[-1] + 1)
        least_midx = np.argmin(np.abs(time_modis.values - date))
        if least_midx not in season_idx_modis:
            if np.abs(least_midx - season_idx_modis[0]) == 1:
                season_idx_modis = np.concatenate(([least_midx], season_idx_modis))
            elif np.abs(least_midx - season_idx_modis[-1]) == 1:
                season_idx_modis = np.concatenate((season_idx_modis, [least_midx]))
            else:
                raise IndexError(f"{site_name} did not obey the season search logic.")
        tidx = np.argwhere(season_idx_modis == least_midx).squeeze()
        sidx, eidx = tidx - mlength // 2, tidx + mlength // 2
        if sidx < 0:
            sidx = 0
            eidx = sidx + mlength
        elif eidx > len(season_idx_modis) - 1:
            eidx = len(season_idx_modis)
            sidx = eidx - mlength

        midx = list(range(sidx, eidx))
        if len(midx) != mlength:
            midx = list(range(sidx, eidx + 1))
        return np.array(season_idx_landsat)[idx], np.array(season_idx_modis)[midx]

    def __getitem__(self, item):
        site_name = self.site_names[item]
        landsat_path = os.path.join(self.landsat_path, site_name)
        landsat_file = os.path.join(landsat_path, site_name + ".nc")
        landsat_dst = xr.open_dataset(landsat_file)
        if self.Landsat == "Single":
            sub_landsat = landsat_dst['data'][landsat_dst['sat'] == b"8"]
            sub_mask = landsat_dst['mask'][landsat_dst['sat'] == b"8"]
        elif self.Landsat == "Union":
            sub_landsat = landsat_dst['data']
            sub_mask = landsat_dst['mask']
        else:
            raise ValueError("Type of Landsat data should be pointed out!")

        temp_mask = np.repeat(sub_mask.values[:, None, ...], 7, axis=1)
        sub_landsat = xr.where(temp_mask == 0, 0, sub_landsat)
        label_path = self.files[item]
        date = datetime.strptime(
            re.findall("\\d{8}", label_path)[0],
            "%Y%m%d"
        )
        modis_path = [os.path.join(self.modis_path, "MOD09%s1" % x, site_name)
                      for x in ["Q", "A"]]
        modis_file = [os.path.join(x, site_name + ".nc") for x in modis_path]

        modis_dst_Q = xr.open_dataset(modis_file[0])
        landsat_idx, modis_idx = self.__time_index(date, sub_landsat,
                                                   modis_dst_Q, sub_mask, site_name)
        if len(landsat_idx) < 2 * self.num_pairs:
            print(site_name, date)
        # 提取 landsat
        landsat = sub_landsat[landsat_idx]
        landsat = landsat / 65455.0
        lon = landsat['lon']
        lat = landsat['lat']
        landsat = landsat.values.transpose(1, 0, 2, 3)
        band_order = [1, 2, 3, 4, 5, 6]
        landsat = landsat[band_order, ...]

        # # 提取 mod09_q1
        # sub_modis_Q = modis_dst_Q['data'][modis_idx]
        # modis_Q = sub_modis_Q.interp(lon=lon, lat=lat, method='slinear')
        # modis_Q = modis_Q.values.transpose(1, 0, 2, 3)
        # 提取 mod09_a1
        modis_dst_A = xr.open_dataset(modis_file[1])
        sub_modis_A = modis_dst_A['data'][modis_idx]
        modis_A = sub_modis_A.interp(lon=lon, lat=lat, method='linear', kwargs={'fill_value': 'extrapolate'})
        modis_A = modis_A.values.transpose(1, 0, 2, 3)
        modis_A = modis_A / 32768.0
        band_order = [2, 3, 0, 1, 5, 6]
        modis_A = modis_A[band_order, ...]
        # modis = np.concatenate((modis_A, modis_Q), axis=0)

        label = self.__load_tiff([label_path], "label")
        label = np.squeeze(label)
        band_order = [1, 2, 3, 4, 5, 6]
        label = label[band_order, ...]

        return {
            "landsat": landsat,
            "modis": modis_A,
            "label": label,
            "label_path": label_path
        }

    def __len__(self):
        return len(self.files)

def main():
    category = "_ESTARFM"
    config_path = "./Benchmark/ESTARFM/parameters_estarfm_fast.yaml"
    save_dir = "/fossfs/skguan/output/data_fusion"
    image_size = 256
    num_pairs = 1  # *** hyper-param: number of images before and after target label
    cloud_cover = 0
    Landsat = "Union"
    landsat_path = "/fossfs/skguan/data_fusion/landsat"
    modis_path = "/fossfs/skguan/data_fusion/modis"
    label_path = "/fossfs/skguan/data_fusion/labels"
    dataset_file = "/fossfs/skguan/data_fusion/dataset_23lr_c0n1.csv"
    df = pd.read_csv(dataset_file)
    train_files = df['train'].dropna().astype(str).to_list()
    train_files += df['val'].dropna().astype(str).to_list()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_files)):
        print(f"\nTraining Fold {fold_idx + 1} / 5")
        # 划分当前fold的数据
        fold_val_files = [train_files[i] for i in val_idx]

        valid_dataloader = DataLoader(
            landsat_path=landsat_path,
            modis_path=modis_path,
            label_path=label_path,
            files=fold_val_files,
            out_channels=6,
            num_pairs=num_pairs,
            cloud_cover=cloud_cover,
            Landsat=Landsat
        )
        label_list, pred_list = [], []
        for data in tqdm.tqdm(valid_dataloader):
            valid_landsat = data['landsat']
            valid_modis = data['modis']
            valid_label = data['label']
            label_path = data['label_path']

            valid_logits = ESTARFM(config_path, valid_landsat, valid_modis, label_path, save_dir, label_path.split("/")[-1])

            label_list.append(valid_label)
            pred_list.append(valid_logits)

            del valid_landsat, valid_modis
            del valid_label, valid_logits
        label_list = np.array(label_list)
        pred_list = np.array(pred_list)
        H, W = [image_size] * 2
        valid_psnr = psnr(label_list.reshape(-1, H, W),
                            pred_list.reshape(-1, H, W),
                            data_range=1)
        valid_ssim = ssim(label_list.reshape(-1, H, W),
                            pred_list.reshape(-1, H, W),
                            data_range=1, channel_axis=1)
        save_path = os.path.join(
            save_dir, "val_files",
            f"val_result_fold{fold_idx}{category}.npz"
        )
        np.savez(save_path, label=label_list, pred=pred_list)
        label_list = label_list.reshape(-1,)
        pred_list = pred_list.reshape(-1,)
        valid_r2 = r2(label_list, pred_list)
        valid_rmse = rmse(label_list, pred_list)

        print("valid_r2: %g, valid_rmse: %g, valid_psnr: %g, valid_ssim: %g" %
              (valid_r2, valid_rmse, valid_psnr, valid_ssim))


if __name__ == "__main__":
    main()
