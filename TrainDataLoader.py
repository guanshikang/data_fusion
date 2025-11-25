# -*- encoding: utf-8 -*-
"""
@brief: Custom dataloader for Landsat and MODIS fusion.

@author: guanshikang

@type: script

Created on Thu Mar 06 20:28:32 2025, HONG KONG
"""
import os
import re
import ast
import numpy as np
import xarray as xr
import netCDF4 as nc
from osgeo import gdal
from datetime import datetime
import torch
import torch.utils.data as data_utils


SEASON = {
    'MAM': ["03", "04", "05"],
    'JJA': ["06", "07", "08"],
    'SON': ["09", "10", "11"],
    'DJF': ["12", "01", "02"]
}
UNIT = "days since 1970-01-01 00:00"
CALENDAR = "standard"


class TrainDataLoader(data_utils.Dataset):
    def __init__(self,
                 landsat_path: str,  # Directory of landsat file (nc, time series).
                 modis_path: str,  # Directory of MODIS file (nc, time series).
                 label_path: str,  # Directory of Lable path (tif, single file).
                 files: list[str]|list[dict],  # Splited full label path (and time index for input dataset) for dataset.
                 image_size: int=256,  # Input image size when training.
                 num_pairs: int=3,  # Number of input images before and after target date.
                 cloud_cover: int=30,  # maximum cloud cover ratio.
                 stats: dict=None,  # if not None, calculated stats will be used for standardization.
                 temporal_mode: str='All',  # 'All' or 'Left', two sides or left side of target date to reconstruct or predict. Case Insensitive.
                 Landsat: str='Single',  # 'Single' or 'Union', Only L8 / L8 & L9. Case Insensitive.
                 modis_upsample: bool=False,  # if upsample MODIS to keep same size with Landsat.
                 update_dataset_file: str=None,  # update dataset_file_name. [dataset_file_name_{'train', 'val', and 'test'}].
                 indexing: bool=False,  # this parameter is used for preindexing.
                 progressive_config: dict=None,  # progressive training configuration.
                ):
        self.landsat_path = landsat_path
        self.modis_path = modis_path
        self.label_path = label_path
        self.files = files
        self.image_size = image_size
        self.num_pairs = num_pairs
        self.cloud_cover = cloud_cover
        self.site_names = self.__get_sitenames() if len(files) > 0 else []
        self.stats = stats
        self.temporal_mode = temporal_mode or 'All'
        self.Landsat = Landsat or "Single"
        self.modis_upsample = modis_upsample

        # for searching input index
        self.update_dataset_file = update_dataset_file
        self.indexing = indexing

        self.progressive_config = progressive_config or {
            'current_epoch': 1,
            'total_epochs': 200,
            'strict_ratio': 0.5,
            'moderate_ratio': 0.8,
            'relaxed_ratio': 1.0
        }

    def __get_sitenames(self):
        if isinstance(self.files[0], dict):
            return [x['label_path'].split("/")[-2] for x in self.files]
        elif isinstance(self.files[0], str):
            return [x.split("/")[-2] for x in self.files]

    def update_progressive_epoch(self, current_epoch):
        self.progressive_config['current_epoch'] = current_epoch

    def _get_progressive_threshold(self):
        current_epoch = self.progressive_config['current_epoch']
        total_epochs = self.progressive_config['total_epochs']
        strict_ratio = self.progressive_config['strict_ratio']
        moderate_ratio = self.progressive_config['moderate_ratio']
        progress = current_epoch / total_epochs
        if progress < strict_ratio:
            return {
                'reflectance_min': 0.0,
                'reflectance_max': 1.0
            }
        elif progress < moderate_ratio:
            return {
                'reflectance_min': -0.1,
                'reflectance_max': 1.2
            }
        else:
            return {
                'reflectance_min': -0.2,
                'reflectance_max': 1.5
            }

    def _data_cleaning(self, data):
        threshold = self._get_progressive_threshold()
        valid_mask = (data >= threshold['reflectance_min']) & \
            (data <= threshold['reflectance_max'])
        data = np.clip(data, threshold['reflectance_min'], threshold['reflectance_max'])

        return data, valid_mask

    def _standardization(self, data, flag):
        data = (data - np.array(self.stats[flag + "_mean"])[:, None, None, None]) / \
            (np.array(self.stats[flag + "_std"])[:, None, None, None] + 1e-8)

        return data

    def __load_tiff(self, file_path):
        data = np.empty(shape=(8, 0, self.image_size, self.image_size))
        file_path = list(file_path)
        file_path.sort()
        for file in file_path:
            if not os.path.exists(file):
                raise FileNotFoundError(f"{file} not found.")
            img = gdal.Open(file).ReadAsArray()
            img = img.reshape(8, 1, self.image_size, self.image_size)
            data = np.concatenate((data, img), axis=1)

        return data

    def __filter_by_season(self,
                           date: datetime,
                           time_dst: xr.Dataset,
                           same_year: bool=False
                           ) -> list:
        """
        Filter seasonal index. If only support same year (MODIS),
        should find background field for winter.
        """
        target_month = date.month
        target_season = {
            k: v for k, v in SEASON.items()
            if str(target_month).zfill(2) in v
        }
        season = [x for x in target_season.keys()][0]
        months = [x for x in target_season.values()][0]
        ori_date = nc.num2date(time_dst.values, units=UNIT, calendar=CALENDAR)
        target_year = date.year
        if same_year:
            syear, eyear = [target_year] * 2
            if season == "DJF":
                if target_month == 12:
                    eyear += 1  # if month is 12, end year should + 1.
                else:
                    syear -= 1  # if month is 1 or 2, start year should - 1.
            start_date = nc.date2num(
                datetime.strptime("%d%s" % (syear, months[0]), "%Y%m"),
                UNIT, CALENDAR
            )
            end_date = nc.date2num(
                datetime.strptime("%d%d" % (eyear, int(months[-1]) + 1), "%Y%m"),
                UNIT, CALENDAR
            )
            temp_date = ori_date[
                (start_date <= time_dst.values) & (time_dst.values < end_date)
            ]
            idx = np.argwhere(np.isin(ori_date, temp_date))
            return [x[0] for x in idx]
        else:
            ori_month = list(map(lambda x: str(x.month).zfill(2), ori_date))
            idx = np.argwhere(np.isin(ori_month, months))

            return [x[0] for x in idx]

    def __nearest_couple_idx(self,
                       idx: np.ndarray,
                       target: int
                       ) -> tuple[int, int]:
        """
        Due to the need of two nearest elements for Landsat, the logic of this is
        to find left and right element of target using two pointer to find them.
        """
        left, right = 0, len(idx) - 1

        # Cross edge process.
        if target <= idx[0] or target >= idx[-1]:
            return np.nan, np.nan

        while left < right - 1:
            mid = (left + right) // 2
            if idx[mid] < target:
                left = mid
            else:
                right = mid

        return left, right

    def _time_index(self,
                    target_date: datetime,
                    landsat_dst: xr.Dataset,
                    modis_dst: xr.Dataset,
                    landsat_mask: xr.Dataset
                    ) -> tuple[np.ndarray, np.ndarray, list, int]:
        """
        Search input data index. Logic is to filter desired dates (seasonal and less
        than maximum cloud ratio), find idx1 nearest to target date, and find idx2
        nearest to idx1, then find min_idx and max_idx according to idx2. MODIS idx
        is corresponding index of Landsat on its own dataset.

        ltarget_idx and mtarget_idx are used for handling boundary situation.
        """
        date = nc.date2num(target_date, units=UNIT, calendar=CALENDAR)
        landsat_time = landsat_dst['time']
        season_idx_landsat = self.__filter_by_season(target_date, landsat_time)
        season_landsat = landsat_time[season_idx_landsat]
        season_mask = landsat_mask[season_idx_landsat]
        llength = 2 * self.num_pairs  # * Here is for the num of image pairs.
        least_idx = np.argmin(np.abs(season_landsat.values - date))  # idx nearest to target date.
        idx = []  # idx less than maximum cloud cover ratio, idx is same level with least_idx.
        for i, clear_data in enumerate(season_mask):
            clear_data = clear_data.values
            clear_ratio = np.sum(clear_data == 1) * 100 / \
                (clear_data.shape[0] * clear_data.shape[1])
            if clear_ratio >= 100 - self.cloud_cover:
                idx.append(i)
        idx = np.array(idx)
        if self.Landsat.lower() == "union":
            temp_idx = np.argwhere(idx == least_idx)
            if temp_idx.size > 0:
                idx = np.delete(idx, temp_idx)  # if "Union", idx list contains [least_idx].
        if len(idx) == 0:
            return [], []
        couple_idx = self.__nearest_couple_idx(idx, least_idx)  # two idx nearest to least_idx.
        temp_len = len(idx)
        min_idx = couple_idx[0] - (llength // 2 - 1)  # left have contained one element
        max_idx = couple_idx[1] + (llength // 2 - 1)  # right have contained one element
        # Ensure edge date find enough input data.
        if min_idx < 0:
            min_idx = 0
            max_idx = min_idx + llength
        elif max_idx > temp_len - 1:
            max_idx = temp_len
            min_idx = max_idx - llength
        lidx = idx[min_idx:max_idx + 1]
        landsat_idx = np.array(season_idx_landsat)[lidx]  # target index of landsat
        ltarget_idx = [i for i, x in enumerate(lidx) if x in idx[list(couple_idx)]]

        # * Same logic for MODIS
        # * find landsat date for modis searching
        sub_landsat_time = landsat_time[landsat_idx].values
        modis_time = modis_dst['time']
        season_idx_modis = self.__filter_by_season(target_date,
                                                   modis_time,
                                                   same_year=False)
        season_modis = modis_time[season_idx_modis]
        mlength = llength + 1
        while len(season_idx_modis) < mlength:  # MODIS needs include one more than Landsat
            season_idx_modis.append(season_idx_modis[-1] + 1)
        least_midx = np.argmin(np.abs(season_modis.values - date))
        # if least_midx not in season_idx_modis:  # If [least_midx] is get from [modis_time]
        #     if np.abs(least_midx - season_idx_modis[0]) == 1:  # The [least_midx] may be outside of target season.
        #         season_idx_modis = np.concatenate(([least_midx], season_idx_modis))  # So find nearest 1 idx to complement
        #     elif np.abs(least_midx - season_idx_modis[-1]) == 1:  # the [seaon_idx_modis], if date distance is larger than 1,
        #         season_idx_modis = np.concatenate((season_idx_modis, [least_midx]))  # there is a problem.
        #     else:  # this logic is also acceptable, but to keep align with Landsat, so still choose [leaset_idx] from [season_modis].
        #         raise IndexError(f"{site_name} did not obey the seasonal MODIS searching logic.")
        """
        Comments below are the logic for finding neareast modis of target date.
        But dates corresponding to landsat is more rational.
        """
        # min_idx, max_idx = least_midx - mlength // 2, least_midx + mlength // 2
        # if min_idx < 0:
        #     min_idx = 0
        #     max_idx = min_idx + mlength
        # elif max_idx > len(season_idx_modis) - 1:
        #     max_idx = len(season_idx_modis)
        #     min_idx = max_idx - mlength
        # midx = list(range(min_idx, max_idx))
        # if len(midx) != mlength:
        #     midx = list(range(min_idx, max_idx + 1))
        midx = []
        for time in sub_landsat_time:
            temp_idx = np.argmin(np.abs(season_modis.values - time))
            midx.append(temp_idx)
        _, mtarget_idx = self.__nearest_couple_idx(midx, least_midx)  # right idx is insertion place.
        midx.insert(mtarget_idx, least_midx)
        modis_idx = np.array(season_idx_modis)[midx]

        return landsat_idx, modis_idx, ltarget_idx, mtarget_idx

    def _get_idx(self,
                 file: str|dict,  # label path or dict [label path, searched dataset index, optional].
                 landsat_dst: xr.Dataset,  # full landsat dataset opened with xarray.
                 modisQ_dst: xr.Dataset,  # full modisQ dataset (A is allowed also).
                 landsat_mask: xr.Dataset,  # full landsat FMask dataset.
                 site_name: str  # site name of this label.
                 ) -> tuple[str, list|np.ndarray, list|np.ndarray]:
        """find input index of Landsat and MODIS. If found before, load it or search it now."""
        if isinstance(file, dict):
            label_path = file['label_path']
            landsat_idx = file['landsat_idx']
            modis_idx = file['modis_idx']
            ltarget_idx = file['ltarget_idx']
            mtarget_idx = file['mtarget_idx']
            if len(landsat_idx) != self.num_pairs * 2 and \
                len(modis_idx) != self.num_pairs * 2 + 1:
                    raise ValueError("Provided index are mismatch with desired length.")
        elif isinstance(file, str):
            if 'idx' in file:
                file = ast.literal_eval(file)
                label_path = file['label_path']
                landsat_idx = file['landsat_idx']
                modis_idx = file['modis_idx']
                ltarget_idx = file['ltarget_idx']
                mtarget_idx = file['mtarget_idx']
                if len(landsat_idx) != self.num_pairs * 2 and \
                    len(modis_idx) != self.num_pairs * 2 + 1:
                        raise ValueError("Provided index are mismatch with desired length.")
            else:
                """
                Once searching finished, results will be saved in a csv file according
                to training strategy, please manually mergy these files, replace original
                [dataset_file] and delete these temporary index files to aviod overwrite.
                """
                label_path = file
                if not os.path.exists(file):
                    raise FileNotFoundError(f"Label: {file} doesn't exist.")
                date = datetime.strptime(re.findall("\\d{8}", file)[0], "%Y%m%d")
                landsat_idx, modis_idx, ltarget_idx, mtarget_idx = self._time_index(date,
                                                                                    landsat_dst,
                                                                                    modisQ_dst,
                                                                                    landsat_mask)
                if len(landsat_idx) < 2 * self.num_pairs:
                    print(site_name, date)

        # Determine whether to predict or reconstruct
        if self.temporal_mode.lower() != "all":
            middle_idx = len(landsat_idx) // 2
            if self.temporal_mode.lower() == "left":
                landsat_idx = landsat_idx[:middle_idx]
                modis_idx = modis_idx[:middle_idx + 1]
            else:
                raise ValueError("Temporal_mode only supports 'ALL' or 'LEFT', please modify its value.")

        return label_path, landsat_idx, modis_idx, ltarget_idx, mtarget_idx

    def __load_data(self,
                    site_name: str
                    ) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset]:
        """Load each input data. return xr.Dataset of Landsat, Landsat_Mask, MOD09Q1, MOD09A1"""
        # load landsat
        landsat_path = os.path.join(self.landsat_path, site_name, site_name + ".nc")
        if not os.path.exists(landsat_path):
            raise FileNotFoundError(f"{landsat_path} not found.")
        landsat_dst = xr.open_dataset(landsat_path)
        if self.Landsat.lower() == "single":
            sub_landsat_dst = landsat_dst['data'][landsat_dst['sat'] == b"8"]
            sub_landsat_mask = landsat_dst['mask'][landsat_dst['sat'] == b"8"]
        elif self.Landsat.lower() == "union":
            sub_landsat_dst = landsat_dst['data']
            sub_landsat_mask = landsat_dst['mask']
        else:
            raise ValueError("Type of Landsat data should be pointed out!")
        # load MODIS09Q / A
        modis_path = [os.path.join(self.modis_path, "MOD09%s1" % x, site_name)
                      for x in ["Q", "A"]]
        modis_file = [os.path.join(x, site_name + ".nc") for x in modis_path]
        modisQ_dst = xr.open_dataset(modis_file[0])
        modisA_dst = xr.open_dataset(modis_file[1])

        return sub_landsat_dst, sub_landsat_mask, modisQ_dst, modisA_dst

    def __getitem__(self, item):
        site_name = self.site_names[item]
        # load raw input dataset
        landsat_dst, landsat_mask, modisQ_dst, modisA_dst = self.__load_data(site_name)
        # find input index
        file = self.files[item]
        label_path, landsat_idx, modis_idx, ltarget_idx, mtarget_idx = self._get_idx(file,
                                                                                     landsat_dst,
                                                                                     modisQ_dst,
                                                                                     landsat_mask,
                                                                                     site_name)
        if self.indexing:  # In indexing process, only iter the dataset and no further process.
            return label_path, landsat_idx, modis_idx, ltarget_idx, mtarget_idx

        # Extract landsat
        sub_landsat = landsat_dst[landsat_idx]
        landsat = sub_landsat.values.transpose(1, 0, 2, 3)  # band first
        band_order = [1, 2, 3, 4, 5, 6]
        landsat = landsat[band_order, ...]
        landsat = landsat * 2.75e-5 - 0.2
        landsat, valid_mask = self._data_cleaning(landsat)

        # Extract MOD09Q1
        sub_modis_Q = modisQ_dst['data'][modis_idx]
        modis_Q = sub_modis_Q.values.transpose(1, 0, 2, 3)
        modis_Q = modis_Q * 1e-4
        modis_Q, _ = self._data_cleaning(modis_Q)
        # Extract MOD09A1
        sub_modis_A = modisA_dst['data'][modis_idx]
        if self.modis_upsample:
            sub_modis_A = sub_modis_A.interp(lon=sub_landsat['lon'],
                                             lat=sub_landsat['lat'],
                                             method='linear',
                                             kwargs={'fill_value': 'extrapolate'})
        modis_A = sub_modis_A.values.transpose(1, 0, 2, 3)
        band_order = [2, 3, 0, 1, 5, 6, 7, 8, 9]
        modis_A = modis_A[band_order, ...].astype(np.float64)
        modis_A[:6], _ = self._data_cleaning(modis_A[:6] * 1e-4)
        modis_A[6:] = modis_A[6:] * 1e-2

        if self.stats is not None:
            landsat = self._standardization(landsat, "landsat")
            landsat = torch.from_numpy(landsat)
            landsat = landsat.to(torch.float32)

            modis_Q = self._standardization(modis_Q, "modis_Q")
            modis_Q = torch.from_numpy(modis_Q)
            modis_Q = modis_Q.to(torch.float32)

            modis_A = self._standardization(modis_A, "modis_A")
            modis_A = torch.from_numpy(modis_A)
            modis_A = modis_A.to(torch.float32)

            label = self.__load_tiff([label_path])
            band_order = [1, 2, 3, 4, 5, 6]
            label = label[band_order, ...]
            label = label * 2.75e-5 - 0.2
            label, gt_mask = self._data_cleaning(label)
            label = torch.from_numpy(label)
            label = torch.squeeze(label).to(torch.float32)
            gt_mask = torch.from_numpy(gt_mask)
            gt_mask = torch.squeeze(gt_mask).to(torch.float32)

            return {
                "landsat": landsat,
                "modis_Q": modis_Q,
                "modis_A": modis_A,
                "label": label,
                "gt_mask": gt_mask,
                "ltarget_idx": ltarget_idx,
                "mtarget_idx": mtarget_idx
            }

        return {
            "landsat": landsat,
            "modis_Q": modis_Q,
            "modis_A": modis_A,
            "valid_mask": valid_mask
        }

    def __len__(self):
        return len(self.files)
