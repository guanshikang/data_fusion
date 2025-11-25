# -*- encoding: utf-8 -*-
"""
@type: module

@brief: data visulization functions.

@author: guanshikang

Created on Wed Oct 15 22:36:19 2025, HONG KONG
"""
import os
import re
import ast
import sys
sys.path.append(sys.path[0] + "/..")
import numpy as np
import xarray as xr
from osgeo import gdal
from datetime import datetime, timedelta
from FunctionalCode.CommonFuncs import CommonFuncs
from FunctionalCode.StatsPlotFuncs import StatsPlot


class DataVisualization:
    def __init__(self):
        pass

    def save_pred(self, file_dir, file_name, index):
        """
        Avoid file so targe that the process be killed.

        Args:
            file_path (str): e.g., "/fossfs/skguan/output/data_fusion/val_files"
            file_name (str): e.g., "val_result_fold0_swin(free_cloud).npz"
            index (int | list[int]): index of data to be saved
            out_name (str): e.g., "r234"
        """
        file_path = os.path.join(file_dir, file_name)
        data = np.load(file_path)
        label = data['label']
        pred = data['pred']
        if [len(label.shape), len(pred.shape)] == [1, 1]:
            label = label.reshape(-1, 6, 256, 256)
            pred = pred.reshape(-1, 6, 256, 256)
        if isinstance(index, int):
            index = [index]
        for i in index:
            CommonFuncs.save_image(f"label_{i}.tif", label[i])
            CommonFuncs.save_image(f"pred_{i}.tif", pred[i])
        print("Have Saved All Image.")

    def pixel_trend_plot(self, site_dir, compare_dir, ref_dates, label_dir=None, index=None):
        """
        line chart of sr time series.

        Args:
            file_dir (str): predict file directory.
            label_path (str, optional): ground truth file path (.nc).
            index (int or tupele[start_point, end_point]): point index of time series.
        """
        ref_dates = self.__date_generate(ref_dates)
        sites = os.listdir(site_dir)
        for site in sites:
            site_path = os.path.join(site_dir, site)
            compare_path = os.path.join(compare_dir, site)
            label_path = os.path.join(label_dir, site + ".nc")
            label_dst = xr.open_dataset(label_path)
            label_time = CommonFuncs.num2date(label_dst['time'])
            files = os.listdir(site_path)
            files.sort()
            date_pattern = r"^.*_(?P<date>\d{8})\.\w+$"
            metrics = {
                'x_pred': [],
                'x_label': [],
                'y_pred': [],
                'y_compare': [],
                'y_label': [],
                'doy': []
            }
            if index is None:
                index = [0, 256]  # Whole image
            elif isinstance(index, int):
                index = [index, index + 1]
            for i, file in enumerate(files):
                date = re.match(date_pattern, file).group("date")
                pred_dst = gdal.Open(os.path.join(site_path, file))
                compare_dst = gdal.Open(os.path.join(compare_path, file))
                pred = pred_dst.ReadAsArray()
                compare = compare_dst.ReadAsArray()
                date_idx = np.argwhere(np.isin(ref_dates, date))
                metrics['x_pred'].append(date_idx[0][0])
                sub_pred = pred[:, index[0]:index[1], index[0]:index[1]]
                sub_compare = compare[:, index[0]:index[1], index[0]:index[1]]
                if len(sub_pred.shape) > 1:
                    sub_pred = np.mean(sub_pred, axis=(1, 2))
                    sub_compare = np.mean(sub_compare, axis=(1, 2))
                metrics['y_pred'].append(sub_pred)
                metrics['y_compare'].append(sub_compare)
                metrics['doy'].append(CommonFuncs.date2doy(date)[1])
                if date in label_time:
                    band_order = [1, 2, 3, 4, 5, 6]
                    idx = np.argwhere(np.isin(label_time, date))
                    metrics['x_label'].append(np.argwhere(np.isin(ref_dates, date))[0][0])
                    label = label_dst['data'][idx[0]].squeeze().values * 2.75e-5 - 0.2
                    sub_label = label[band_order, index[0]:index[1], index[0]:index[1]]
                    if len(sub_label.shape) > 1:
                        sub_label = np.mean(sub_label, axis=(1, 2))
                    metrics['y_label'].append(sub_label)
            names = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
            sp = StatsPlot()
            file_name = f"{site}.png"
            sp.doy_line_plot(metrics, names, file_name=file_name)

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
        time_series = [datetime.strftime(x, "%Y%m%d") for x in time_series]

        return time_series


def main():
    dv = DataVisualization()
    # file_dir = "/fossfs/skguan/output/data_fusion/val_files"
    # file_name = "val_result_swin(No_PPM).npz"
    # index = list(range(0, 10))
    # dv.save_pred(file_dir, file_name, index)
    ref_dates = ["20230101", "20231230", 8]
    label_dir = "/fossfs/skguan/data_fusion/inference/landsat"
    site_dir = "/fossfs/skguan/output/data_fusion/inference/OurModel"
    compare_dir = "/fossfs/skguan/output/data_fusion/inference/SwinSTFM"
    index = [127, 130]
    dv.pixel_trend_plot(site_dir, compare_dir, ref_dates, label_dir, index)


if __name__ == "__main__":
    main()
