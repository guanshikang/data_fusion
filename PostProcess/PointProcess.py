# -*- encoding: utf-8 -*-
"""
@type: module

@brief: mainly for single file or csv table of training points.

@author: guanshikang

Created on Wed Oct 15 21:43:07 2025, HONG KONG
"""
import os
import glob
import shutil
import numpy as np
import pandas as pd
from osgeo import gdal

class PointProcess:
    def __init__(self):
        pass

    @staticmethod
    def cross_tile(glob_pattern: str) -> None:
        """
        Check if input images are near the edge of per raw tile.
        If cross tile, the ignore value (0) will be contained in SWIR1 and SWIR2 bands.
        These labels should be cleaned.

        Args:
            glob_pattern: e.g. "/fossfs/skguan/data_fusion/labels/*/*.tif"
        """
        files = glob.iglob(glob_pattern)
        for file in files:
            data = gdal.Open(file).ReadAsArray()
            if np.any(data[5:7, :, :] == 0):
                os.remove(file)
                print(f"{file} has been reomved.")
        print("All process done.")

    @staticmethod
    def empty_folder(file_dir: str) -> None:
        """
        delete the empty folder.

        Args:
            file_dir (str): e.g. "/fossfs/skguan/data_fusion/landsat"
        """
        sites = os.listdir(file_dir)
        for site in sites:
            files = glob.glob(os.path.join(file_dir, site, "*.tif"))
            if len(files) == 0:
                file_path = os.path.join(file_dir, site)
                shutil.rmtree(file_path)
                print(f"{file_path} has been removed.")

    def size_unit_convertion(self, fsize):
        if fsize < 1024:
            return(round(fsize, 4), 'Byte')
        else:
            KBX = fsize / 1024
            if KBX < 1024:
                return(round(KBX, 4), 'K')
            else:
                MBX = KBX / 1024
                if MBX < 1024:
                    return(round(MBX, 4), 'M')
                else:
                    GBX = MBX / 1024
                    if GBX < 1024:
                        return(round(GBX, 4), 'G')
                    else:
                        return(round(GBX / 1024, 4), 'T')


    @staticmethod
    def file_size_summary(dataset_csv, check_dir):
        """
        get total file size for training dataset.
        """
        df = pd.read_csv(dataset_csv, usecols=['train', 'val', 'test'])
        sites = set([x.split("/")[-2] for x in df.dropna().values.reshape(-1,)])
        for sensor in ['landsat', 'modis/MOD09A1', 'modis/MOD09Q1', 'labels']:
            file_dir = os.path.join(check_dir, sensor)
            total_size = 0
            if sensor != "labels":
                for site in sites:
                    file = os.path.join(file_dir, site, site + ".nc")
                    size = os.path.getsize(file)
                    total_size += size
                total_size, unit = PointProcess.size_unit_convertion(PointProcess, total_size)
                print(f"{sensor} used {total_size} {unit}.")
            else:
                for site in sites:
                    files = glob.glob(os.path.join(file_dir, site, "*.tif"))
                    for file in files:
                        size = os.path.getsize(file)
                        total_size += size
                total_size, unit = PointProcess.size_unit_convertion(PointProcess, total_size)
                print(f"{sensor} used {total_size} {unit}.")

def main():
    dataset = "/fossfs/skguan/data_fusion/dataset_23lr_c0n1_new.csv"
    check_dir = "/fossfs/skguan/data_fusion"
    pp = PointProcess()
    pp.file_size_summary(dataset, check_dir)


if __name__ == "__main__":
    main()



