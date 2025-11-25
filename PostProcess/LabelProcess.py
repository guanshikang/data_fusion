# -*- encoding: utf-8 -*-
"""
@type: module

@brief: operations related to label selection, label purification and so on.

@author: guanshikang

Created on Wed Oct 15 21:59:45 2025, HONG KONG
"""
import os
import sys
sys.path.append(sys.path[0] + "/..")
import re
import glob
import tqdm
import numpy as np
import pandas as pd
from FunctionalCode.CommonFuncs import CommonFuncs


class LabelProcess(CommonFuncs):
    def __init__(self):
        super().__init__()

    def same_patch_check(self):
        # 根据左上角点检查label是否是同一个patch
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
            min_x, max_y, _, _ = super().get_extent(file)
            site_name = file.split("/")[-2]
            point_dict['site_id'][i] = site_name
            point_dict['lon_ul'][i] = min_x
            point_dict['lat_ul'][i] = max_y
        df = pd.DataFrame(point_dict)
        df = df.groupby([df['lon_ul'], df['lat_ul']]).aggregate({'site_id': 'first'})
        df.to_csv("/fossfs/skguan/data_fusion/check.csv")

    def label2RGB(self):
        # 检查label里面是否含云
        label_dir = "/fossfs/skguan/output/data_fusion/benchmark"
        pattern = "LC09_*_*_2023*_202*_*_*.tif"
        out_dir = "/fossfs/skguan/output/data_fusion/benchmark/rgb"
        files = glob.iglob(os.path.join(label_dir, "ESTARFM", pattern))
        for file in tqdm.tqdm(files):
            # site_name = file.split("/")[-2]
            # date = re.match(".*LC09_.*_\\d{6}_(\\d{8})_\\d{8}", file).group(1)
            # output_path = os.path.join(out_dir, site_name + f"_{date}.tif")
            name = file.split("/")[-1]
            output_path = os.path.join(out_dir, name)
            # if os.path.exists(output_path):
                # continue
            file_info = super().get_fileinfo(file)
            data = file_info['src_ds'].ReadAsArray()[:3]
            min_value = np.percentile(data, 2)
            max_value = np.percentile(data, 98)
            data = np.clip((data - min_value) / (max_value - min_value) * 255, 0, 255).astype(np.uint8)
            super().save_image(file, output_path, data)

    def rm_cloudy_label(self):
        # 根据csv文件转移label RGB 然后删除含云的标签, 先label2RGB后本函数
        work_dir = "/fossfs/skguan/data_fusion/"
        label_path = os.path.join(work_dir, "dataset_23lr_c0n1.csv")
        labels = pd.read_csv(label_path)
        input_dir = os.path.join(work_dir, "ref_image", "rgb_n1")
        pattern = ".*LC09_.*_\\d{6}_(\\d{8})_\\d{8}"
        func = lambda x: x.split("/")[-2] + f"_{re.match(pattern, x).group(1)}.tif"

        # ! PART 1: Copy CSV Label to Another Directory.
        # if not os.path.exists(input_dir):
            # os.mkdir(input_dir)
        # label_values = list(itertools.chain.from_iterable(
        #     [labels[x].dropna().values for x in ["train", "val", "test"]]
        # ))
        # if len(os.listdir(input_dir)) != len(label_values):
        #     rgb_files = glob.iglob(os.path.join(work_dir, "ref_image", "labels_rgb", "*.tif"))
        #     df = list(map(func, label_values))
        #     for rgb_file in tqdm.tqdm(rgb_files):
        #         file_name = rgb_file.split("/")[-1]
        #         if os.path.exists(os.path.join(input_dir, file_name)):
        #             continue
        #         if file_name in df:
        #             shutil.copy(rgb_file, input_dir)
        #     exit()

        # ! PART 2: Save Filtered CSV File.
        input_files = os.listdir(input_dir)
        headers = ["train", "val", "test"]
        temp_dict = pd.DataFrame()
        for header in headers:
            series = labels[header].dropna()
            df = series.to_frame(name=header)
            df['site'] = series.apply(func)
            flag = df['site'].isin(input_files)
            temp_dict = pd.concat([temp_dict, df[header][flag]], axis=1)
        output_path = label_path.replace(".csv", "_new.csv")
        pd.DataFrame(temp_dict).to_csv(output_path)


def main():
    lp = LabelProcess()
    lp.same_patch_check()
    lp.label2RGB()
    lp.rm_cloudy_label()

if __name__ == "__main__":
    main()
