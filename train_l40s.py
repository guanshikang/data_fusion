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
from torch.cuda.amp import autocast

from FunctionalCode.CommonFuncs import CommonFuncs
from FunctionalCode.GlobalStats import ComputeStats
from FunctionalCode.StatsPlotFuncs import StatsPlot
from Backbone.ResNet import ResNet
from Backbone.UNet import UNet
from Backbone.SpatioTemporalViT import SpatioTemporalViT
from Backbone.STViT import ViTTimeSeriesModel
from Backbone.ViT_SkipConnection import ViT_Skip
from Backbone.SwinTransformer import SwinTransformer


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
    def __init__(self, label_dir, ref_dir):
        self.dataset = self.__get_dataset(label_dir, ref_dir)

    def __get_dataset(self, label_dir, ref_dir):
        sites = os.listdir(ref_dir)
        files = map(lambda x: self._filter_by_season(label_dir, x), sites)
        files = list(itertools.chain.from_iterable(files))
        file_num = len(files)
        tiles = set(map(
            lambda x: re.match("\\w+_(\\d{6})_",
                               os.path.split(x)[-1]).group(1), files
            )
        )
        dataset = {
            "train": [],
            "val": [],
            "test": []
        }
        train_num = int(file_num * .8)
        val_num = train_num + int(file_num * .1)
        num = 0
        for tile in tiles:
            temp_files = filter(
                lambda x: re.match(f"LC09.*_{tile}_2023.*.tif",
                                   os.path.basename(x)),
                files
            )
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


class AlignDataset:
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
                data = np.full((7, 256, 256), -.2, np.float32)
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
                 stats=None, MODIS=False):
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
        self.MODIS = MODIS

    def __get_sitenames(self):
        return [x.split("/")[-2] for x in self.files]

    def __standardization(self, data, flag):
        data = np.where(data < 0, np.nan, data)
        if flag == "label":
            pass
        else:
            data = (data - self.stats[flag + "_mean"][:, None, None, None]) / \
                (self.stats[flag + "_std"][:, None, None, None] + 1e-8)

        return np.nan_to_num(data, nan=0.0)

    def __load_tiff(self, file_path, flag):
        data = np.empty(shape=(7, 0, self.image_size, self.image_size))
        file_path = list(file_path)
        file_path.sort()
        for file in file_path:
            img = gdal.Open(file).ReadAsArray()[:-1]
            img = img.reshape(7, 1, self.image_size, self.image_size)
            img = self.__standardization(img, flag)
            data = np.concatenate((data, img), axis=1)

        return data

    def __get_date_encoding(self, file_names):
        pattern = ".*\\d{6}_(\\d{8})"
        dates = list(map(lambda x: re.match(pattern, x).group(1), file_names))
        dates = list(map(lambda x: datetime.strptime(x, "%Y%m%d"), dates))
        dates.sort()
        doy = list(map(lambda x: x.timetuple().tm_yday, dates))

        return doy

    def __time_index(self, target_date, landsat_dst, modis_dst, landsat_mask):
        units = "days since 1970-01-01 00:00"
        calendar = "standard"
        date = nc.date2num(target_date, units=units, calendar=calendar)
        time_landsat = landsat_dst['time']
        # OPTION 1: Limited Image Pairs
        # START
        # # Time Range
        # SUB OPTION 1: Random Search in Same Season
        def filter_by_season(date, time_dst, units, calendar, same_year=False):
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
                end_date = nc.date2num(
                    datetime.strptime("%d%d" % (target_year, int(months[-1]) + 1),
                                    "%Y%m"),
                    units, calendar
                )
                if season == "DJF":
                    target_year -= 1
                start_date = nc.date2num(
                    datetime.strptime("%d%s" % (target_year, months[0]),
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

        season_idx_landsat = filter_by_season(target_date, time_landsat,
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
        if len(idx) == 0:
            return [], []
        target_idx = np.argwhere(idx == least_idx)  # Need remove target date
        idx = np.delete(idx, target_idx)            # to integrate Landsat-9
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
        season_idx_modis = filter_by_season(target_date, time_modis,
                                            units, calendar, same_year=True)
        while len(season_idx_modis) < 12:  # ~12 MODIS images in a specific season
            season_idx_modis.append(season_idx_modis[-1] + 1)
        sidx, eidx = [np.argmin(np.abs(time_modis.values - x)) for x in
                      [time_landsat[idx[0]].values,
                       time_landsat[idx[-1]].values]]

        # return idx, (sidx, eidx)
        return np.array(season_idx_landsat)[idx], season_idx_modis

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
        sub_landsat = landsat_dst['data']  # [landsat_dst['sat'] == b"8"]
        sub_mask = landsat_dst['mask']  # [landsat_dst['sat'] == b"8"]

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
                                                   modis_dst_Q, sub_mask)
        if len(landsat_idx) < 2 * self.num_pairs:
            print(site_name, date)
        # 提取 landsat
        landsat = sub_landsat[landsat_idx].values.transpose(1, 0, 2, 3)  # ViT: 波段在前
        # imgs = input_dst['data'][idx].values.reshape(-1,  # reshape成三维给CNN网络用
        #                                              self.image_size,
        #                                              self.image_size)
        landsat = landsat / 65535.0

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

                label = self.__load_tiff([label_path], "label")
                label = np.squeeze(label, axis=1)
                label = label / 65535.0
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
        # # Transformer pad for time dimension.
        # t_pad = 24 - imgs.shape[1]
        # if t_pad != 0:
        #     imgs = np.pad(imgs,
        #                   ((0, 0), (0, t_pad), (0, 0), (0, 0)),
        #                   'constant', constant_values=0.0)

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
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))

    return psnr


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
    # label_dir = "/fossfs/skguan/data_fusion/rpt_labels"
    # ref_dir = "/fossfs/skguan/data_fusion/modis/sub_image/rpt_modis/MOD09A1"
    # sd = SplitDataset(label_dir, ref_dir)
    # dst_path = "/fossfs/skguan/data_fusion/dataset_23_rpt(s).csv"
    # sd.to_csv(dst_path)

    # # align dataset
    # input_path = "/fossfs/skguan/data_fusion/landsat"
    # align = AlignDataset(input_path)
    # align.align_dataset()

    # training configuration
    category = "_vit(SEASON_2Sat)"
    SECOND_TRAINING = False
    batch_size = 4  # *** hyper-param: batch size per iteration
    num_pairs = 6  # *** hyper-param: number of images before and after target label
    cloud_cover = 60  # *** hyper-param: maximum cloud cover
    epochs = 200  # ** hyper-param: total training epoches
    warm_up = 10  # * hyper-param: warm_up epoches
    psnr_factor = 1  # ** hyper-param: psnr ratio for loss function
    lr = 3e-5  # *** hyper-parm: initialized learning rate
    accumulation_steps = 2  # *** hyper-parm: avoid cuda out of memory
    n_splits = 5
    start_epoch = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    landsat_path = "/lustre1/g/geog_geors/skguan/landsat"
    modis_path = "/lustre1/g/geog_geors/skguan/modis"
    label_path = "/lustre1/g/geog_geors/skguan/labels"
    dataset_file = "/lustre1/g/geog_geors/skguan/dataset_23(season)_num12.csv"
    df = pd.read_csv(dataset_file)
    train_files = df['train'].dropna().astype(str).to_list()
    train_files += df['val'].dropna().astype(str).to_list()
    # train_files = filter_dataset(train_files)[:19]
    test_files = df['test'].dropna().astype(str).to_list()
    # test_files = filter_dataset(test_files)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_files)):
        print(f"\nTraing Fold {fold_idx + 1} / {n_splits}")
        # 划分当前fold的数据
        fold_train_files = [train_files[i] for i in train_idx]
        fold_val_files = [train_files[i] for i in val_idx]

        # 计算当前fold的landsat统计量
        stats_dataset = DataLoader(landsat_path, modis_path, label_path,
                                   fold_train_files, num_pairs=num_pairs,
                                   cloud_cover=cloud_cover)
        stats_dataloader = data_utils.DataLoader(stats_dataset, batch_size=1,
                                                 shuffle=False)

        cgs = ComputeStats()
        landsat_mean, landsat_std = cgs.compute_global_stats(
            stats_dataloader, "landsat", channel_num=7
        )
        # print(MOD_idx)

        # 计算当前fold的modis统计量
        stats_dataset = DataLoader(landsat_path, modis_path, label_path,
                                   fold_train_files, num_pairs=num_pairs,
                                   cloud_cover=cloud_cover, MODIS=True)
        stats_dataloader = data_utils.DataLoader(stats_dataset, batch_size=1,
                                                 shuffle=False)
        modisQ_mean, modisQ_std = cgs.compute_global_stats(
            stats_dataloader, "modis_Q", channel_num=2
        )
        modisA_mean, modisA_std = cgs.compute_global_stats(
            stats_dataloader, "modis_A", channel_num=10
        )
        stats = {
            "landsat_mean": landsat_mean, "landsat_std": landsat_std,
            "modis_Q_mean": modisQ_mean, "modis_Q_std": modisQ_std,
            "modis_A_mean": modisA_mean, "modis_A_std": modisA_std
        }

        train_dataset = DataLoader(
            landsat_path,
            modis_path,
            label_path,
            fold_train_files,
            num_pairs=num_pairs,
            cloud_cover=cloud_cover,
            stats=stats,
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
                # torch.stack([x['doy'] for x in batch]),
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
                # torch.stack([x['doy'] for x in batch]),
            ),
            pin_memory=True,
            num_workers=6,
            persistent_workers=True
        )
        aux_steps = 12   # Fixed steps for a specific season
        # model = SpatioTemporalViT(t_patch=2, patch_size=4, d_model=512)
        # model = ResNet(in_channels=42, out_channels=7)
        # model = UNet(in_channels=42, out_channels=7)
        model = ViT_Skip(main_steps=num_pairs * 2, main_spatch=16,
                         main_tpatch=2, aux_steps=aux_steps, aux_spatch=2,
                         aux_tpatch=1, aux_inchans=(2, 10), attn_drop_rate=0.1)
        # model = SwinTransformer(main_steps=2 * num_pairs, main_spatch=4,
        #                         main_tpatch=2, main_inchans=7, aux_size=18,
        #                         aux_steps=aux_steps, aux_spatch=4, aux_tpatch=2,
        #                         embed_dim=96, depths=[2, 2, 6, 2],
        #                         num_heads=[3, 6, 12, 24], out_chans=7,
        #                         window_sizes=[(2, 8, 8), (2, 8, 8),
        #                                      (2, 4, 4), (2, 4, 4)])

        model.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                      weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.01,
                         end_factor=1,
                         total_iters=warm_up * len(train_dataloader)),
                CosineAnnealingLR(
                    optimizer,
                    T_max=(epochs - warm_up) * len(train_dataloader),
                    eta_min=1e-6
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
            check_point = torch.load(f"checkpoint_fold{fold_idx}{category}.pth")
            model.load_state_dict(check_point['model'])
            optimizer.load_state_dict(check_point['optimizer'])
            scheduler.load_state_dict(check_point['scheduler'])
            start_epoch = check_point['epoch']

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

        for epoch in range(start_epoch, epochs + 1):
            model.train()
            label_list, pred_list = [], []
            count = 0
            loss_metric_train = 0
            optimizer.zero_grad()
            for iter, data in enumerate(train_dataloader):
                train_landsat, train_modisQ, train_modisA, train_label = data
                train_landsat = train_landsat.to(device, non_blocking=True)
                train_modisQ = train_modisQ.to(device, non_blocking=True)
                train_modisA = train_modisA.to(device, non_blocking=True)
                train_label = train_label.to(device, non_blocking=True)
                # doy = doy.to(device)

                train_logits = model(train_landsat, train_modisQ, train_modisA)
                psnr_loss = PSNRLoss(train_logits, train_label)
                mse_loss = criterion.forward(train_logits, train_label)
                train_loss = mse_loss + psnr_factor * (1 / psnr_loss)
                scaled_loss = train_loss / accumulation_steps

                # train_pred = train_logits.clone().detach()
                scaled_loss.backward()
                loss_metric_train += train_loss.item()
                count += 1
                if (iter + 1) % accumulation_steps == 0 or (iter + 1) == len(train_dataloader):
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                label_list.append(train_label.cpu().detach().numpy())
                pred_list.append(train_logits.cpu().detach().numpy())
                if iter % 10 == 0:
                    current_avg_loss = loss_metric_train / count
                    print("epoch: %d, iter: %d, lr: %g, loss: %g" %
                          (epoch, iter, optimizer.param_groups[0]['lr'],
                           current_avg_loss))
                del train_landsat, train_modisQ, train_modisA
                del train_label, train_logits
                torch.cuda.empty_cache()
            metrics_dict['train_loss'].append(loss_metric_train / count)
            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)
            train_psnr = psnr(label_list, pred_list, data_range=1)
            train_ssim = ssim(label_list, pred_list, data_range=1)
            metrics_dict['train_psnr'].append(train_psnr)
            metrics_dict['train_ssim'].append(train_ssim)
            if train_psnr > temp_psnr:
                np.savez(f"train_result_fold{fold_idx}{category}.npz",
                         label_list[:30], pred_list[:30])
            label_list = label_list.reshape(-1,)
            pred_list = pred_list.reshape(-1,)
            train_r2 = r2(label_list, pred_list)
            metrics_dict['train_r2'].append(train_r2)
            train_rmse = rmse(label_list, pred_list)
            metrics_dict['train_rmse'].append(train_rmse)
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
                        # doy = doy.to(device)

                        valid_logits = model(valid_landsat, valid_modisQ,
                                             valid_modisA)
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
                label_list = np.concatenate(label_list)
                pred_list = np.concatenate(pred_list)
                valid_psnr = psnr(label_list, pred_list, data_range=1)
                valid_ssim = ssim(label_list, pred_list, data_range=1)
                metrics_dict['valid_psnr'].append(valid_psnr)
                metrics_dict['valid_ssim'].append(valid_ssim)
                if valid_psnr > best_psnr:
                    np.savez(f"val_result_fold{fold_idx}{category}.npz",
                             label_list[:30], pred_list[:30])
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
                    torch.save(model_state,
                               f"checkpoint_fold{fold_idx}{category}.pth")

        fold_metrics.append({'fold': fold_idx, 'best_r2': best_r2})
        with open(f"metrics_fold{fold_idx}{category}.pkl", 'wb') as f:
            pkl.dump(metrics_dict, f)
        key = ["loss", "r2", "rmse", "psnr", "ssim"]
        sp = StatsPlot()
        file_name = f"metric_plot_fold{fold_idx}{category}.png"
        sp.line_plot(metrics_dict, key, file_name=file_name)

        test_dataset = DataLoader(
            landsat_path,
            modis_path,
            label_path,
            test_files,
            num_pairs=num_pairs,
            cloud_cover=cloud_cover,
            stats=stats,
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
        check_point = torch.load(f"checkpoint_fold{fold_idx}{category}.pth")
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
        label_list = np.concatenate(label_list)
        pred_list = np.concatenate(pred_list)
        test_psnr = psnr(label_list, pred_list, data_range=1)
        test_ssim = ssim(label_list, pred_list, data_range=1)
        np.savez(f"test_result_fold{fold_idx}{category}.npz",
                 label_list[:30], pred_list[:30])
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
