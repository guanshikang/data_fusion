# -*- encoding: utf-8 -*-
"""
@brief: transformer training.

@author: guanshikang

@type: script

Created on Thu Mar 06 20:28:32 2025, HONG KONG
"""
import os
import re
import glob
import random
import argparse
import itertools
import numpy as np
import xarray as xr
import pandas as pd
import pickle as pkl
import netCDF4 as nc
import matplotlib.pyplot as plt
from osgeo import gdal
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score as r2
from sklearn.metrics import root_mean_squared_error as rmse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torchsummary import summary
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torchmetrics import R2Score, MeanSquaredError, PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torch.cuda.amp import autocast

from FunctionalCode.CommonFuncs import CommonFuncs
from FunctionalCode.GlobalStats import ComputeStats
from FunctionalCode.StatsPlotFuncs import StatsPlot
from Backbone.ResNet import ResNet
from Backbone.UNet import UNet
from Backbone.SpatioTemporalViT import SpatioTemporalViT
from Backbone.STViT import ViTTimeSeriesModel
from Backbone.ViT_SkipConnection import ViT_Skip
from Backbone.SwinTransformer_v3 import SwinTransformer


SEASON = {
    'MAM': ["03", "04", "05"],
    'JJA': ["06", "07", "08"],
    'SON': ["09", "10", "11"],
    'DJF': ["12", "01", "02"]
}


class SplitDataset:
    """
    生成训练数据集.
    """
    def __init__(self, label_dir=None, ref_dir=None):
        if label_dir and ref_dir:
            self.dataset = self.__get_dataset(label_dir, ref_dir)

    def __get_dataset(self, label_dir, ref_dir):
        sites = os.listdir(ref_dir)
        files = map(lambda x: self._filter_by_season(label_dir, x), sites)
        files = list(itertools.chain.from_iterable(files))
        file_num = len(files)
        tiles = set(map(
            lambda x: re.match("\\w+_(\\d{6})_", os.path.split(x)[-1]).group(1),
            files
        ))
        dataset = {
            "train": [],
            "val": [],
            "test": []
        }
        train_num = int(file_num * .6)
        val_num = train_num + int(file_num * .2)
        num = 0
        for tile in tiles:
            temp_files = filter(lambda x: re.match(
                    f"LC09.*_{tile}_2023.*.tif",os.path.basename(x)
            ), files)
            temp_files = list(temp_files)
            num += len(temp_files)
            train_ratio = (num - train_num) / file_num
            val_ratio = (num - val_num) / file_num
            if train_ratio < 0:
                dataset["train"].extend(temp_files)
            elif val_ratio < 0:
                dataset["val"].extend(temp_files)
            else:
                dataset["test"].extend(temp_files)

        return dataset

    def to_csv(self, file_path):
        df = pd.DataFrame(
            dict([(k, pd.Series(v)) for k, v in self.dataset.items()])
        )
        df.to_csv(file_path)

    def _filter_by_season(self, label_dir, site):

        files = []
        for i in range(3, 4):
            temp_files = glob.glob(os.path.join(
                label_dir, site, "*_202%d*_202[2-4]*.tif" % i
            ))
            if len(temp_files) > 0:
                for value in SEASON.values():
                    temp_file = list(filter(
                        lambda x: re.findall("\\d{8}", x)[0][4:6] in value,
                        temp_files
                    ))
                    if len(temp_file) > 0:
                        files.append(random.choice(temp_file))

        return files

    def random_dataset(self, current_filepath, nonoverlap_csv):
        current_dataset = pd.read_csv(current_filepath, usecols=['train', 'val', 'test'])
        current_dataset = pd.DataFrame(current_dataset.to_numpy().reshape(-1, 1, order='F'), columns=['file']).dropna()
        nonoverlap = pd.read_csv(nonoverlap_csv)
        current_dataset['site_id'] = current_dataset['file'].apply(lambda x: x.split('/')[-2])
        current_dataset = current_dataset[current_dataset['site_id'].isin(nonoverlap['site_id'])]
        current_dataset = current_dataset.sample(frac=1.0)

        current_dataset.to_csv(current_filepath)


class AlignDataset:
    """按照日期进行输入数据的padding，现已不需要."""
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.cf = CommonFuncs()

    def align_dataset(self):
        sites = os.listdir(self.input_dir)
        for site in sites:
            input_path = os.path.join(self.input_dir, site)
            os.chdir(input_path)
            files = glob.glob("LC08*_202[23]*_202*")
            files.sort()
            ref_img = files[0]
            pattern = "LC08.+\\d{6}_(\\d{8})_\\d{8}"
            s_date = re.match(pattern, ref_img).group(1)
            date_series = pd.date_range(
                start=s_date,
                end='20231231',
                freq='16D'
            )
            num_count = len(date_series)
            if len(date_series) != 46:
                num_count -= 46
                date_series = date_series.append(pd.date_range(
                    start=date_series[0],
                    periods=abs(num_count) + 2,
                    freq='9D'
                ))
            temp_dates = list(map(
                lambda x: re.match(pattern, x).group(1),
                files
            ))
            date_series = date_series.strftime("%Y%m%d")
            is_in = ~np.isin(date_series.to_numpy(), np.array(temp_dates))
            date_series = date_series[is_in]
            for date in date_series:
                output_name = re.match("\\w+_\\d{6}_", ref_img).group()
                output_name += "%s_padding.tif" % date
                data = np.full((7, 256, 256), -0.2, np.float32)
                self.cf.save_image(ref_img, output_name, data)
                print("{} in {} is Copied!".format(date, site))


class DataLoader(data_utils.Dataset):
    """
    训练数据加载器.

    Args:
        data_utils (class): parent class.
    """
    def __init__(self, landsat_path, modis_path, label_path, files,
                 out_channels=7, image_size=256, num_pairs=3, cloud_cover=30,
                 stats=None, Landsat=None, MODIS=False):
        self.landsat_path = landsat_path
        self.modis_path = modis_path
        self.label_path = label_path
        self.out_channels = out_channels
        self.image_size = image_size
        self.num_pairs = num_pairs
        self.cloud_cover = cloud_cover
        self.files = files
        self.site_names = self.__get_sitenames()
        self.stats = stats
        self.standardizable = stats is not None
        self.Landsat = Landsat or "Single"
        self.MODIS = MODIS

    def __get_sitenames(self):
        return [x.split("/")[-2] for x in self.files]

    def __standardization(self, data, flag):
        data = np.where(data < -1.0, np.nan, data)
        data = (data - self.stats[flag + "_mean"][:, None, None, None]) / \
            (self.stats[flag + "_std"][:, None, None, None] + 1e-8)

        return np.nan_to_num(data, nan=0.0)

    def __load_tiff(self, file_path):
        data = np.empty(shape=(8, 0, self.image_size, self.image_size))
        file_path = list(file_path)
        file_path.sort()
        for file in file_path:
            img = gdal.Open(file).ReadAsArray()
            img = img.reshape(8, 1, self.image_size, self.image_size)
            data = np.concatenate((data, img), axis=1)

        return data

    def __get_date_encoding(self, file_names):
        pattern = ".*\\d{6}_(\\d{8})"
        dates = list(map(lambda x: re.match(pattern, x).group(1), file_names))
        dates = list(map(lambda x: datetime.strptime(x, "%Y%m%d"), dates))
        dates.sort()
        doy = list(map(lambda x: x.timetuple().tm_yday, dates))

        return doy

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
        # OPTION 1: Limited Image Pairs
        # START
        # # Time Range
        # SUB OPTION 1: Random Search in Same Season

        season_idx_landsat = self.__filter_by_season(target_date, time_landsat,
                                                     units, calendar)
        season_landsat = time_landsat[season_idx_landsat]
        season_mask = landsat_mask[season_idx_landsat]
        # SUB OPTION 3: Random Search in the Whole Time Range (3 years)
        # PASS
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
            idx = np.delete(idx, target_idx)  # if "Union", idx will include target date.
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
        # END

        # # OPTION 2: ALL Images with Cloud Contamination In 1 Year
        # # START
        # date_ls = nc.num2date(landsat_dst.values, units=units,
        #                       calendar=calendar)
        # idx = [x.year == year for x in date_ls]
        # # END

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

        # return idx, (sidx, eidx)
        midx = list(range(sidx, eidx))
        if len(midx) != mlength:
            midx = list(range(sidx, eidx + 1))
        return np.array(season_idx_landsat)[idx], np.array(season_idx_modis)[midx]

    def __getitem__(self, item):
        site_name = self.site_names[item]
        landsat_path = os.path.join(self.landsat_path, site_name)
        # input_files = list(filter(
        #         lambda y: re.match("LC08\\w+_\\d{3}[23]\\d+.*", y),
        #         os.listdir(input_path)
        # ))  # 读取站点文件夹下的所有tif文件
        # doy = self.__get_date_encoding(input_files)
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
        # year = date.year
        modis_path = [os.path.join(self.modis_path, "MOD09%s1" % x, site_name)
                      for x in ["Q", "A"]]
        modis_file = [os.path.join(x, site_name + ".nc") for x in modis_path]

        modis_dst_Q = xr.open_dataset(modis_file[0])
        landsat_idx, modis_idx = self.__time_index(date, sub_landsat,
                                                   modis_dst_Q, sub_mask, site_name)
        if len(landsat_idx) < 2 * self.num_pairs:
            print(site_name, date)
        # 提取 landsat
        landsat = sub_landsat[landsat_idx].values.transpose(1, 0, 2, 3)  # ViT: 波段在前
        # mask = sub_mask[landsat_idx].values[None, :, :, :]
        # imgs = input_dst['data'][idx].values.reshape(-1,  # reshape成三维给CNN网络用
        #                                              self.image_size,
        #                                              self.image_size)
        landsat = landsat / 65455.0
        band_order = [1, 2, 3, 4, 5, 6]
        # band_order = [1, 2, 3, 4]
        landsat = landsat[band_order, ...]
        # landsat = np.concatenate((landsat, mask), axis=0)

        if self.MODIS:
            # 提取 mod09_q1
            sub_modis_Q = modis_dst_Q['data'][modis_idx]
            modis_Q = sub_modis_Q.values.transpose(1, 0, 2, 3)
            modis_Q = modis_Q / 32768.0
            # 提取 mod09_a1
            modis_dst_A = xr.open_dataset(modis_file[1])
            sub_modis_A = modis_dst_A['data'][modis_idx]
            modis_A = sub_modis_A.values.transpose(1, 0, 2, 3)
            modis_A = modis_A / 32768.0
            # band_order = [2, 3, 0, 1, 5, 6, 7, 8, 9]
            band_order = [2, 3, 0, 1, 5, 6, 7, 8, 9]
            modis_A = modis_A[band_order, ...]

            if self.standardizable:
                landsat = self.__standardization(landsat, "landsat")
                landsat = torch.from_numpy(landsat)
                landsat = landsat.to(torch.float32)

                modis_Q = self.__standardization(modis_Q, "modis_Q")
                modis_Q = torch.from_numpy(modis_Q)
                modis_Q = modis_Q.to(torch.float32)

                modis_A = self.__standardization(modis_A, "modis_A")
                modis_A = torch.from_numpy(modis_A)
                modis_A = modis_A.to(torch.float32)

                # label_max = self.stats['label_max']
                # label_min = self.stats['label_min']
                # label = (label - label_min[:, None, None]) \
                #     / (label_max[:, None, None] - label_min[:, None, None])
                label = self.__load_tiff([label_path])
                label = label / 65455.0
                band_order = [1, 2, 3, 4, 5, 6]
                label = label[band_order, ...]
                label = torch.from_numpy(label)
                label = torch.squeeze(label).to(torch.float32)

                return {
                    "landsat": landsat,
                    "modis_Q": modis_Q,
                    "modis_A": modis_A,
                    "label": label
                }

            return {
                "modis_Q": modis_Q,
                "modis_A": modis_A
            }

        return {
            "landsat": landsat,
            # "modis_idx": modis_idx,  # Max sequence for searching MODIS dynamically
            # "doy": torch.tensor(doy),
        }

    def __len__(self):
        return len(self.files)

def PSNRLoss(origin_img, predict_img, max_val=1.0):
    """
    PSNR损失. 10 * log10(MAX^2 / MSE).

    Args:
        origin_img (ndarray): original image.
        predict_img (ndarray): predict image.
        max_val (float, optional): max value of datatype. Defaults to 1.0.
    """
    mse = torch.mean((predict_img - origin_img) ** 2)
    psnr = 10 * torch.log10(max_val / torch.sqrt(mse))

    return psnr

def SAMLoss(pred, target, epsilon=1e-3):
    """Spectral Angle Mapper Loss"""
    dot_product = (pred * target).sum(dim=1)
    pred_norm = torch.norm(pred, dim=1) + epsilon
    target_norm = torch.norm(target, dim=1) + epsilon
    cos_theta = dot_product / (pred_norm * target_norm)
    sam = torch.mean(torch.acos(torch.clamp(cos_theta, epsilon - 1,
                                            1 - epsilon)))

    return sam

def filter_dataset(dataset):
    """
    按条件过滤数据集.

    Args:
        dataset (list): input dataset list.
    """
    dataset = np.array(dataset)
    sites = list(set(map(lambda x: x.split("/")[-2], dataset)))
    sites_order = np.array([x.split("/")[-2] for x in dataset])
    sub_dsts = [dataset[np.argwhere(sites_order == x)] for x in sites]
    new_dataset = map(lambda x: random.choice(x), sub_dsts)

    return list(itertools.chain.from_iterable(new_dataset))


def main():
    # # split dataset
    # label_dir = "/fossfs/skguan/data_fusion/labels"
    # ref_dir = "/fossfs/skguan/data_fusion/modis/MOD09A1"
    # sd = SplitDataset(label_dir, ref_dir)
    # dst_path = "/fossfs/skguan/data_fusion/dataset_Large.csv"
    # sd.to_csv(dst_path)

    # # align dataset
    # input_path = "/fossfs/skguan/data_fusion/landsat"
    # align = AlignDataset(input_path)
    # align.align_dataset()

    # training configuration
    category = "_swin(SSA_LRc0n1)"
    save_dir = "/fossfs/skguan/output/data_fusion"
    SECOND_TRAINING = True  # TODO: whether training from checkpoint
    TRAIN_MODE = True  # TODO: training mode, or prediction mode
    STATS_AHEAD = True  # TODO: whether to use statistical indicators calculated ahead
    batch_size = 4  # *** hyper-param: batch size per iteration
    num_pairs = 1  # *** hyper-param: number of images before and after target label
    cloud_cover =  0 # *** hyper-param: maximum cloud cover
    epochs = 200  # ** hyper-param: total training epoches
    warm_up = 10  # * hyper-param: warm_up epoches
    psnr_factor = 1  # ** hyper-param: psnr ratio for loss function
    lr = 2e-5  # *** hyper-parm: initialized learning rate
    accumulation_steps = 4  # *** hyper-parm: avoid cuda out of memory
    mean = {
        "landsat": np.array([
            0.18261683, 0.20459992, 0.21709642, 0.263481, 0.24679904, 0.22141413
        ]),
        "modis_Q": np.array([0.05548652, 0.08200782]),
        "modis_A": np.array([
            0.03407968, 0.04817599, 0.05523952, 0.0818292, 0.07751813, 0.05380522,
            0.12644616, 0.05240689, 0.12257224
        ]),
    } if STATS_AHEAD else None
    std = {
        "landsat": np.array([
            0.10740318, 0.10066043, 0.1064226, 0.09414231, 0.07772066, 0.07192402
        ]),
        "modis_Q": np.array([0.04422916, 0.0411989]),
        "modis_A": np.array([
            0.04024915, 0.04064206, 0.04376934, 0.04040155, 0.04256677, 0.07692242,
            0.04287472, 0.04321922, 0.26931927
        ]),
    } if STATS_AHEAD else None
    Landsat = "Union"  # two kind of landsat data use strategy. Single -> 8. Union -> 8 & 9.
    n_splits = 5
    start_epoch = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    landsat_path = "/fossfs/skguan/data_fusion/landsat"
    modis_path = "/fossfs/skguan/data_fusion/modis"
    label_path = "/fossfs/skguan/data_fusion/labels"
    dataset_file = "/fossfs/skguan/data_fusion/dataset_23l_c0n1 (1).csv"
    # sd = SplitDataset()
    # sd.random_dataset(dataset_file, "/fossfs/skguan/data_fusion/nonoverlap_sites.csv")
    df = pd.read_csv(dataset_file)
    train_files = df['train'].dropna().astype(str).to_list()
    train_files += df['val'].dropna().astype(str).to_list()
    # train_files = filter_dataset(train_files)[:5]
    test_files = df['test'].dropna().astype(str).to_list()
    # test_files = filter_dataset(test_files)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_files)):
        print(f"\nTraining Fold {fold_idx + 1} / {n_splits}")
        # 划分当前fold的数据
        fold_train_files = [train_files[i] for i in train_idx]
        fold_val_files = [train_files[i] for i in val_idx]

        if not STATS_AHEAD:
            cgs = ComputeStats()
            # 计算当前fold的landsat统计量
            stats_dataset = DataLoader(landsat_path, modis_path, label_path,
                                       fold_train_files, num_pairs=num_pairs,
                                       cloud_cover=cloud_cover, Landsat=Landsat)
            stats_dataloader = data_utils.DataLoader(stats_dataset, batch_size=1,
                                                     shuffle=False)
            landsat_mean, landsat_std = cgs.compute_global_stats(
                stats_dataloader, "landsat", mode='std',
                category=category, save_dir=save_dir, channel_num=6
            )

            # 计算当前fold的modis统计量
            stats_dataset = DataLoader(landsat_path, modis_path, label_path,
                                       fold_train_files, num_pairs=num_pairs,
                                       cloud_cover=cloud_cover, Landsat=Landsat,
                                       MODIS=True)
            stats_dataloader = data_utils.DataLoader(stats_dataset, batch_size=1,
                                                     shuffle=False)
            modisQ_mean, modisQ_std = cgs.compute_global_stats(
                stats_dataloader, "modis_Q", mode='std',
                category=category, save_dir=save_dir, channel_num=2
            )
            modisA_mean, modisA_std = cgs.compute_global_stats(
                stats_dataloader, "modis_A", mode='std',
                category=category, save_dir=save_dir, channel_num=9
            )
            mean = {
                "landsat": landsat_mean,
                "modis_Q": modisQ_mean,
                "modis_A": modisA_mean,
            }
            std = {
                "landsat": landsat_std,
                "modis_Q": modisQ_std,
                "modis_A": modisA_std,
            }

        print("\nStatistical Indicators:")
        print("-----------------------------------")
        print("Landsat:\nMean: {0}\nStd: {1}\n".format(mean['landsat'], std['landsat']))
        print("MODIS_Q:\nMean: {0}\nStd: {1}\n".format(mean['modis_Q'], std['modis_Q']))
        print("MODIS_A:\nMean: {0}\nStd: {1}\n".format(mean['modis_A'], std['modis_A']))
        print("-----------------------------------\n")
        stats = {
            "landsat_mean": mean['landsat'], "landsat_std": std['landsat'],
            "modis_Q_mean": mean['modis_Q'], "modis_Q_std": std['modis_Q'],
            "modis_A_mean": mean['modis_A'], "modis_A_std": std['modis_A'],
        }

        train_dataset = DataLoader(
            landsat_path,
            modis_path,
            label_path,
            fold_train_files,
            num_pairs=num_pairs,
            cloud_cover=cloud_cover,
            stats=stats,
            Landsat=Landsat,
            MODIS=True
        )
        train_dataloader = data_utils.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: (
                torch.stack([x['landsat'] for x in batch]),
                torch.stack([x['modis_Q'] for x in batch]),
                torch.stack([x['modis_A'] for x in batch]),
                torch.stack([x['label'] for x in batch]),
            ),
            pin_memory=True,  # 加速CPU到GPU的数据传输
            num_workers=6,  # 使用多进程加载数据
            persistent_workers=True
        )

        valid_dataset = DataLoader(
            landsat_path,
            modis_path,
            label_path,
            fold_val_files,
            num_pairs=num_pairs,
            cloud_cover=cloud_cover,
            stats=stats,
            Landsat=Landsat,
            MODIS=True
        )
        valid_dataloader = data_utils.DataLoader(
            valid_dataset,
            batch_size=batch_size,
            collate_fn=lambda batch: (
                torch.stack([x['landsat'] for x in batch]),
                torch.stack([x['modis_Q'] for x in batch]),
                torch.stack([x['modis_A'] for x in batch]),
                torch.stack([x['label'] for x in batch]),
            ),
            pin_memory=True,
            num_workers=6,
            persistent_workers=True
        )
        aux_steps = 2 * num_pairs + 1  # Fixed MODIS steps for a specific season
        # model = SpatioTemporalViT(t_patch=2, patch_size=4, d_model=512)
        # model = ResNet(in_channels=42, out_channels=7)
        # model = UNet(in_channels=42, out_channels=7)
        # model = ViT_Skip(main_steps=num_pairs * 2, main_spatch=16,
        #                  main_tpatch=2, aux_steps=aux_steps, aux_spatch=2,
        #                  aux_tpatch=1, aux_inchans=(2, 10), attn_drop_rate=0.1,
        #                  embed_dim=768, depth=12, num_heads=12, depth_wise=[3, 6, 9])
        model = SwinTransformer(main_steps=2 * num_pairs, main_spatch=4,
                                main_tpatch=2, main_inchans=6, aux_size=36,
                                aux_steps=aux_steps, aux_spatch=2, aux_tpatch=2,
                                aux_inchans=(2, 9), embed_dim=96, out_chans=6,
                                depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                                window_sizes=[(2, 8, 8), (2, 8, 8),
                                              (2, 4, 4), (2, 4, 4)])

        model.to(device)
        criterion = nn.SmoothL1Loss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                      weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(
                    optimizer,
                    start_factor=0.01,
                    end_factor=1,
                    total_iters=warm_up * len(train_dataloader)
                ),
                CosineAnnealingLR(
                    optimizer,
                    T_max=(epochs - warm_up) * len(train_dataloader) / 2,
                    eta_min=5e-6
                )
            ],
            milestones=[(warm_up) * len(train_dataloader)]
        )

        best_r2 = -np.inf
        best_psnr = -np.inf  # for validation process
        temp_psnr = -np.inf  # for training process
        metrics_dict = {'train_loss': [], 'train_r2': [], 'train_rmse': [],
                        'train_psnr': [], 'train_ssim': [],
                        'valid_loss': [], 'valid_r2': [], 'valid_rmse': [],
                        'valid_psnr': [], 'valid_ssim': []}
        if SECOND_TRAINING:
            checkpoint_path = os.path.join(
                save_dir, "checkpoint",
                f"checkpoint_fold{fold_idx}{category}.pth"
            )
            if os.path.exists(checkpoint_path):
                check_point = torch.load(checkpoint_path)
                model.load_state_dict(check_point['model'])
                optimizer.load_state_dict(check_point['optimizer'])
                scheduler.load_state_dict(check_point['scheduler'])
                start_epoch = check_point['epoch'] + 1
            else:
                print("There is no referred checkpoint, please check!")
                print("Training from initial state...")

        # RF测试
        # rf = RandomForestRegressor(n_estimators=100, max_depth=5)
        # x_trian, y_train = train_dataset[0]
        # x_trian = x_trian.reshape(7 * 256 * 256, 46)
        # y_train = y_train.reshape(7 * 256 * 256, 1)
        # rf.fit(x_trian, y_train)
        # x_val, y_val = valid_dataset[0]
        # x_val = x_val.reshape(7 * 256 * 256, 46)
        # y_val = y_val.reshape(7 * 256 * 256, 1)
        # y_pred = rf.predict(x_val)
        # print(r2_score(y_val.cpu().numpy(), y_pred))
        if TRAIN_MODE:
            for epoch in range(start_epoch, epochs + 1):
                model.train()
                label_list, pred_list = [], []
                count = 0
                sum_y, sum_y2, sum_pred, sum_pred2, sum_y_pred = [0.0] * 5
                n_pixels = 0
                ssim_list, psnr_list = [], []
                loss_metric_train = 0
                optimizer.zero_grad()
                for iter, data in enumerate(train_dataloader):
                    train_landsat, train_modisQ, train_modisA, train_label = data
                    train_landsat = train_landsat.to(device, non_blocking=True)
                    train_modisQ = train_modisQ.to(device, non_blocking=True)
                    train_modisA = train_modisA.to(device, non_blocking=True)
                    train_label = train_label.to(device, non_blocking=True)
                    train_logits = model(train_landsat, train_modisQ, train_modisA)
                    # train_logits = torch.squeeze(train_logits)
                    psnr_loss = PSNRLoss(train_logits, train_label)
                    mse_loss = criterion.forward(train_logits, train_label)
                    train_loss = mse_loss + psnr_factor * (1 / psnr_loss)
                    scaled_loss = train_loss / accumulation_steps

                    scaled_loss.backward()
                    loss_metric_train += train_loss.item()
                    count += 1
                    if ((iter + 1) % accumulation_steps == 0) or \
                        ((iter + 1) == len(train_dataloader)):
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()

                    # Compute statistical indicators per batch
                    sum_y += torch.sum(train_label).item()
                    sum_y2 += torch.sum(train_label ** 2).item()
                    sum_pred += torch.sum(train_logits).item()
                    sum_pred2 += torch.sum(train_logits ** 2).item()
                    sum_y_pred += torch.sum(train_label * train_logits).item()
                    n_pixels += train_label.numel()
                    # Per-image SSIM and PSNR
                    for i in range(train_label.size(0)):
                        batch_label = train_label[i]
                        batch_logits = train_logits[i]
                        label_cpu = batch_label.cpu().detach().numpy()
                        logits_cpu = batch_logits.cpu().detach().numpy()
                        p = psnr(label_cpu, logits_cpu, data_range=1.0)
                        psnr_list.append(p)
                        s = ssim(label_cpu, logits_cpu, data_range=1.0, channel_axis=0)
                        ssim_list.append(s)

                    if iter % 10 == 0:
                        label_list.append(train_label.cpu().detach().numpy())
                        pred_list.append(train_logits.cpu().detach().numpy())
                        current_avg_loss = loss_metric_train / count
                        print("epoch: %d, iter: %d, lr: %g, loss: %g" %
                              (epoch, iter, optimizer.param_groups[0]['lr'],
                               current_avg_loss))
                    del train_landsat, train_modisQ, train_modisA
                    del train_label, train_logits
                    torch.cuda.empty_cache()
                metrics_dict['train_loss'].append(loss_metric_train / count)
                ss_tot = sum_y2 - (sum_y ** 2 / n_pixels)
                ss_res = sum_y2 - 2 * sum_y_pred + sum_pred2
                train_r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
                train_rmse = np.sqrt(ss_res / n_pixels)
                train_psnr = np.mean(psnr_list)
                train_ssim = np.mean(ssim_list)
                metrics_dict['train_psnr'].append(train_psnr)
                metrics_dict['train_ssim'].append(train_ssim)
                if train_psnr > temp_psnr:
                    save_path = os.path.join(
                        save_dir, "train_files",
                        f"train_result_fold{fold_idx}{category}.npz"
                    )
                    np.savez(save_path,
                             label=label_list[:30],
                             pred=pred_list[:30])
                print("epoch: %d, train_r2: %g, train_rmse: %g, "
                      "train_psnr: %g, train_ssim: %g" %
                      (epoch, train_r2, train_rmse, train_psnr, train_ssim))

                if epoch % 2 == 0:
                    model.eval()
                    label_list, pred_list = [], []
                    count = 0
                    loss_metric_val = 0
                    with torch.no_grad():
                        for data in valid_dataloader:
                            valid_landsat, valid_modisQ, valid_modisA, valid_label = data

                            valid_landsat = valid_landsat.to(device, non_blocking=True)
                            valid_modisQ = valid_modisQ.to(device, non_blocking=True)
                            valid_modisA = valid_modisA.to(device, non_blocking=True)
                            valid_label = valid_label.to(device, non_blocking=True)

                            valid_logits = model(valid_landsat, valid_modisQ, valid_modisA)
                            psnr_loss = PSNRLoss(valid_logits, valid_label)
                            mse_loss = criterion.forward(valid_logits, valid_label)
                            valid_loss = mse_loss + psnr_factor * (1 / psnr_loss)

                            loss_metric_val += valid_loss.item()
                            count += 1
                            label_list.append(valid_label.cpu().detach().numpy())
                            pred_list.append(valid_logits.cpu().detach().numpy())

                            del valid_landsat, valid_modisQ, valid_modisA,
                            del valid_label, valid_logits
                            torch.cuda.empty_cache()
                    metrics_dict['valid_loss'].append(loss_metric_val / count)
                    label_list = np.concatenate(label_list, dtype=np.float32)
                    pred_list = np.concatenate(pred_list, dtype=np.float32)
                    valid_psnr = psnr(label_list, pred_list, data_range=1)
                    valid_ssim = ssim(label_list, pred_list,
                                      data_range=1, channel_axis=1)
                    metrics_dict['valid_psnr'].append(valid_psnr)
                    metrics_dict['valid_ssim'].append(valid_ssim)
                    if valid_psnr > best_psnr:
                        save_path = os.path.join(
                            save_dir, "val_files",
                            f"val_result_fold{fold_idx}{category}.npz"
                        )
                        np.savez(save_path,
                                 label=label_list,
                                 pred=pred_list)
                    label_list = label_list.reshape(-1,)
                    pred_list = pred_list.reshape(-1,)
                    valid_r2 = r2(label_list, pred_list)
                    metrics_dict['valid_r2'].append(valid_r2)
                    valid_rmse = rmse(label_list, pred_list)
                    metrics_dict['valid_rmse'].append(valid_rmse)

                    print("epoch: %d, valid_r2: %g, valid_rmse: %g, "
                          "valid_psnr: %g, valid_ssim: %g" %
                          (epoch, valid_r2, valid_rmse, valid_psnr, valid_ssim))
                    if valid_r2 > best_r2:
                        best_r2 = valid_r2
                        model_state = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'epoch': epoch
                        }
                        print("Fold %d: find best model in "
                              "epoch %d with r2 score %f" %
                              (fold_idx, epoch, best_r2))
                        save_path = os.path.join(
                            save_dir, "checkpoint",
                            f"checkpoint_fold{fold_idx}{category}.pth"
                        )
                        torch.save(model_state, save_path)

            fold_metrics.append({'fold': fold_idx, 'best_r2': best_r2})
            save_path = os.path.join(
                save_dir, "metrics",
                f"metrics_fold{fold_idx}{category}.pkl"
            )
            with open(save_path, 'wb') as f:
                pkl.dump(metrics_dict, f)
            key = ["loss", "r2", "rmse", "psnr", "ssim"]
            sp = StatsPlot()
            file_name = f"./loss_pics/metric_plot_fold{fold_idx}{category}.png"
            sp.line_plot(metrics_dict, key, file_name=file_name)

        test_dataset = DataLoader(
            landsat_path,
            modis_path,
            label_path,
            test_files,
            num_pairs=num_pairs,
            cloud_cover=cloud_cover,
            stats=stats,
            Landsat=Landsat,
            MODIS=True
        )
        test_dataloader = data_utils.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: (
                torch.stack([x['landsat'] for x in batch]),
                torch.stack([x['modis_Q'] for x in batch]),
                torch.stack([x['modis_A'] for x in batch]),
                torch.stack([x['label'] for x in batch]),
                # torch.stack([x['doy'] for x in batch]),
            ),
            pin_memory=True,
            num_workers=6,
            persistent_workers=True
        )
        checkpoint_path = os.path.join(
            save_dir, "checkpoint",
            f"checkpoint_fold{fold_idx}{category}.pth"
        )
        check_point = torch.load(checkpoint_path)
        model.load_state_dict(check_point['model'])
        model.eval()
        label_list, pred_list = [], []
        count = 0
        loss_metric_test = 0
        with torch.no_grad():
            for data in test_dataloader:
                test_landsat, test_modisQ, test_modisA, test_label = data

                test_landsat = test_landsat.to(device, non_blocking=True)
                test_modisQ = test_modisQ.to(device, non_blocking=True)
                test_modisA = test_modisA.to(device, non_blocking=True)
                test_label = test_label.to(device, non_blocking=True)
                # doy = doy.to(device)

                test_logits = model(test_landsat, test_modisQ, test_modisA)
                test_logits = torch.squeeze(test_logits)
                psnr_loss = PSNRLoss(test_logits, test_label)
                mse_loss = criterion.forward(test_logits, test_label)
                test_loss = mse_loss + psnr_factor * (1 / psnr_loss)

                loss_metric_test += test_loss.item()
                count += 1
                label_list.append(test_label.cpu().detach().numpy())
                pred_list.append(test_logits.cpu().detach().numpy())

                del test_landsat, test_modisQ, test_modisA
                del test_label, test_logits
                torch.cuda.empty_cache()

        print("Fold %d: The test loss is %.4f." %
              (fold_idx, loss_metric_test / count))
        label_list = np.concatenate(label_list, dtype=np.float32)
        pred_list = np.concatenate(pred_list, dtype=np.float32)
        test_psnr = psnr(label_list, pred_list, data_range=1)
        test_ssim = ssim(label_list, pred_list,
                         data_range=1, channel_axis=1)
        save_path = os.path.join(
            save_dir, "val_files",
            f"val_result_fold{fold_idx}{category}.npz"
        )
        np.savez(save_path, label=label_list, pred=pred_list)
        label_list_all = label_list.reshape(-1,)
        pred_list_all = pred_list.reshape(-1,)
        test_r2 = r2(label_list_all, pred_list_all)
        print("Fold %d: The test r2 is %.4f." % (fold_idx, test_r2))
        test_rmse = rmse(label_list_all, pred_list_all)
        print("Fold %d: The test rmse is %.4f." % (fold_idx, test_rmse))
        print("Fold %d: The test psnr is %.4f." % (fold_idx, test_psnr))
        print("Fold %d: The test ssim is %.4f." % (fold_idx, test_ssim))

    print("\nCross-validation Results:")
    for fm in fold_metrics:
        print(f"Fold {fm['fold']}: Best R2 = {fm['best_r2']:.4f}")
    mean_r2 = np.mean([fm['best_r2'] for fm in fold_metrics])
    print(f"Mean R2 across folds: {mean_r2:.4f}")


if __name__ == "__main__":
    main()
