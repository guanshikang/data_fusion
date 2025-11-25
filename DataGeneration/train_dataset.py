# -*- encoding: utf-8 -*-
"""
@type: module

@brief: auxiliary functions for generating training dataset.

@author: guanshikang

Created on Fri Oct 17 14:13:53 2025, HONG KONG
"""
import os
import re
import sys
sys.path.append(sys.path[0] + "/..")
import glob
import random
import itertools
import numpy as np
import pandas as pd
from FunctionalCode.CommonFuncs import CommonFuncs

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

