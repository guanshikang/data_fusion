# -*- encoding: utf-8 -*-
"""
@brief: DataLoader for Inference.

@author: guanshikang

@type: script

Created on Mon Oct 13 11:37:40 2025, HONG KONG
"""
import os
import ast
import torch
import numpy as np
import xarray as xr
from TrainDataLoader import TrainDataLoader
from datetime import datetime, timedelta


class InferenceDataLoader(TrainDataLoader):
    def __init__(self,
                 landsat_path: str,  # Directory of landsat file (nc).
                 modis_path: str,  # Directory of MODIS file (nc).
                 dates: list[str, str, int]=[],  # Target prediction date, 'yyyymmdd'. [start_date, end_date, interval]
                 num_pairs: int=1,  # Time pairs of used data.
                 cloud_cover: float=0,  # maximum cloud cover ratio.
                 temporal_mode: str='All',  # 'All' or 'Left', two sides or left side of target date to reconstruct or predict. Case Insensitive.
                 Landsat: str='Single',  # "Single" or "Union", Only L8 / L8 & L9.
                 modis_upsample: bool=False,  # if upsample MODIS to keep same size with Landsat.
                 stats: dict={},  # calculated statistical indictors of training dataset.
                 ):
        super().__init__(landsat_path, modis_path, label_path=None, files=[], stats=stats)
        self.input_files = self.__get_files(landsat_path, modis_path)
        self.dates = self.__date_generate(dates)
        self.num_pairs = num_pairs
        self.cloud_cover = cloud_cover
        self.temporal_mode = temporal_mode
        self.Landsat = Landsat
        self.modis_upsample = modis_upsample

    def __get_files(self, landsat_path, modis_path) -> {str, list}:
        """Initialization of input files. Determine length of DataLoader"""
        self.site_names = os.listdir(landsat_path)
        landsat_files = [os.path.join(landsat_path, x) for x in self.site_names]
        modisA_files = [os.path.join(modis_path, "MOD09A1", x) for x in self.site_names]
        modisQ_files = [os.path.join(modis_path, "MOD09Q1", x) for x in self.site_names]
        input_files = {
            'landsat': landsat_files,
            'modis_A': modisA_files,
            'modis_Q': modisQ_files
        }
        self.__file_len = len(landsat_files)

        return input_files

    def __date_generate(self, dates) -> list[datetime]:
        """Get a list for time indexing."""
        if isinstance(dates, str):
            dates = ast.literal_eval(dates)
        start_date, end_date, interval = dates
        start_date = datetime.strptime(start_date, "%Y%m%d")
        end_date = datetime.strptime(end_date, "%Y%m%d")
        time_interval = timedelta(days=interval)
        time_series = [start_date + time_interval * i \
            for i in range(0, (end_date - start_date).days // interval + 1)]
        self.__date_len = len(time_series)

        return time_series

    def __load_data(self, file_item) -> None:
        """Save each input data until finishing iteration of date list."""
        landsat_dst = xr.open_dataset(
            self.input_files['landsat'][file_item]
        )
        if self.Landsat == "Single":
            self.sub_landsat_dst = landsat_dst['data'][landsat_dst['sat'] == b"8"]
            self.sub_landsat_mask = landsat_dst['mask'][landsat_dst['sat'] == b"8"]
        elif self.Landsat == "Union":
            self.sub_landsat_dst = landsat_dst['data']
            self.sub_landsat_mask = landsat_dst['mask']
        else:
            raise ValueError("Type of Landsat data should be pointed out!")

        self.modisQ_dst = xr.open_dataset(
            self.input_files['modis_Q'][file_item]
        )

        self.modisA_dst = xr.open_dataset(
            self.input_files['modis_A'][file_item]
        )

    def _data_standard(self, landsat_idx: list, modis_idx: list) -> tuple[list, list, list]:
        """Prepocess input data for next stage."""
        sub_landsat = self.sub_landsat_dst[landsat_idx]
        landsat = sub_landsat.values.transpose(1, 0, 2, 3)
        band_order = [1, 2, 3, 4, 5, 6]
        landsat = landsat[band_order, ...]
        landsat = landsat * 2.75e-5 - 0.2
        landsat, _ = super()._data_cleaning(landsat)
        landsat = super()._standardization(landsat, "landsat")

        sub_modis_Q = self.modisQ_dst['data'][modis_idx]
        modis_Q = sub_modis_Q.values.transpose(1, 0, 2, 3)
        modis_Q = modis_Q * 1e-4
        modis_Q, _ = super()._data_cleaning(modis_Q)
        modis_Q = super()._standardization(modis_Q, "modis_Q")

        sub_modis_A = self.modisA_dst['data'][modis_idx]
        if self.modis_upsample:
            sub_modis_A = sub_modis_A.interp(lon=sub_landsat['lon'],
                                             lat=sub_landsat['lat'],
                                             method='linear',
                                             kwargs={'fill_value': 'extrapolate'})
        modis_A = sub_modis_A.values.transpose(1, 0, 2, 3)
        band_order = [2, 3, 0, 1, 5, 6, 7, 8, 9]
        modis_A = modis_A[band_order, ...].astype(np.float64)
        modis_A[:6], _ = super()._data_cleaning(modis_A[:6] * 1e-4)
        modis_A[6:] = modis_A[6:] * 1e-2
        modis_A = super()._standardization(modis_A, "modis_A")

        return landsat, modis_Q, modis_A

    def __getitem__(self, item):
        file_item, date_item = divmod(item, self.__date_len)
        date = self.dates[date_item]
        site_name = self.site_names[file_item].replace(".nc", "")
        format_date = self.dates[date_item].strftime("%Y%m%d")
        self.__load_data(file_item)
        landsat_idx, modis_idx = super()._time_index(date,
                                                     self.sub_landsat_dst,
                                                     self.modisQ_dst,
                                                     self.sub_landsat_mask)
        if len(landsat_idx) < 2 * self.num_pairs:
            print("No target input for %s on %s" % (site_name, format_date))
            return {
                'landsat': torch.ones(0),
                'modis_Q': torch.ones(0),
                'modis_A': torch.ones(0),
                'out_name': None
            }
        if self.temporal_mode.lower() != "all":
            middle_idx = len(landsat_idx) // 2
            if self.temporal_mode.lower() == "left":
                landsat_idx = landsat_idx[:middle_idx]
                modis_idx = modis_idx[:middle_idx + 1]
            else:
                raise ValueError("Temporal_mode only supports 'ALL' or 'LEFT', please modify its value.")

        landsat, modis_Q, modis_A = self._data_standard(landsat_idx, modis_idx)

        landsat = torch.from_numpy(landsat)
        landsat = landsat.to(torch.float32)

        modis_Q = torch.from_numpy(modis_Q)
        modis_Q = modis_Q.to(torch.float32)

        modis_A = torch.from_numpy(modis_A)
        modis_A = modis_A.to(torch.float32)

        return {
            'landsat': landsat,
            'modis_Q': modis_Q,
            'modis_A': modis_A,
            'out_name': "%s_%s" % (site_name, format_date)
        }

    def __len__(self):
        return self.__file_len * self.__date_len