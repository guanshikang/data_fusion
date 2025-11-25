# -*- encoding: utf-8 -*-
"""
@brief: 图像没有重叠做镶嵌、区域相同做合成

@author: guanshikang

@type: script

Created on Thu Aug 11 16:08:52 2022, Beijing
"""
import os
import re
import glob
import math
import tqdm
import numpy as np
from osgeo import gdal


def GetExtent(in_fn):
    """
    获取影像的左上角和右下角坐标

    Args:
        in_fn: 输入影像

    Returns:
        左上角坐标和右下角坐标
    """
    ds = gdal.Open(in_fn)
    geotrans = list(ds.GetGeoTransform())
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    min_x = geotrans[0]
    max_y = geotrans[3]
    max_x = geotrans[0] + xsize * geotrans[1]
    min_y = geotrans[3] + ysize * geotrans[5]

    del ds
    return min_x, max_y, max_x, min_y


def main():
    path = "/fossfs/skguan/DATA/7.GLC30_p"
    os.chdir(path)
    in_files = glob.glob("*Annual*.tif")

    in_fn = in_files[0]
    filename = "/fossfs/skguan/data_fusion/GLC30_2022.tif"
    if os.path.exists(filename):
        os.remove(filename)
    # 获取待镶嵌栅格的最大最小的坐标值
    min_x, max_y, max_x, min_y = GetExtent(in_fn)
    for in_fn in in_files[1:]:
        minx, maxy, maxx, miny = GetExtent(in_fn)
        min_x = min(min_x, minx)
        min_y = min(min_y, miny)
        max_x = max(max_x, maxx)
        max_y = max(max_y, maxy)
    # 计算镶嵌后影像的行列号
    in_ds = gdal.Open(in_files[0])
    geotrans = list(in_ds.GetGeoTransform())
    width = geotrans[1]
    height = geotrans[5]

    columns = math.ceil((max_x - min_x) / width)
    rows = math.ceil((max_y - min_y) / abs(height))

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(filename, columns, rows, 1, gdal.GDT_Float32)
    out_ds.SetProjection(in_ds.GetProjection())
    geotrans[0] = min_x
    geotrans[3] = max_y
    out_ds.SetGeoTransform(geotrans)
    out_band = out_ds.GetRasterBand(1)

    # 定义仿射逆变换
    inv_geotrans = gdal.InvGeoTransform(geotrans)

    # 开始逐渐写入

    iter = tqdm.tqdm(in_files)
    for in_fn in iter:
        print("Processing {}".format(in_fn))
        in_ds = gdal.Open(in_fn)
        in_gt = in_ds.GetGeoTransform()
        # 仿射逆变换
        offset = gdal.ApplyGeoTransform(inv_geotrans, in_gt[0], in_gt[3])
        x, y = map(int, offset)
        trans = gdal.Transformer(in_ds, out_ds, [])  # in_ds是源栅格，out_ds是目标栅格
        _, xyz = trans.TransformPoint(False, 0, 0)  # 计算in_ds中左上角像元对应out_ds中的行列号
        x, y, _ = map(math.ceil, xyz)
        in_da = in_ds.GetRasterBand(23).ReadAsArray()
        out_da = out_band.ReadAsArray(x, y, in_ds.RasterXSize, in_ds.RasterYSize)
        in_da = np.where((in_da == 0) & (out_da != 0), out_da, in_da)
        try:
            out_band.WriteArray(in_da, x, y)  # 镶嵌操作
        except Exception as e:
            print(e)
            print("Current x, y are {0}, {1}".format(x, y))
            print("Total rows and cols are {0}, {1}".format(rows, columns))
            print("Total file number is {}".format(len(in_files)))

    # print("%s has finished the mean overlay" % date)
    del in_ds, out_band, out_ds


if __name__ == "__main__":
    main()
