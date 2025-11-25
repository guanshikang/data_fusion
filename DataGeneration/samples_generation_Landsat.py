# -*- encoding: utf-8 -*-
"""
@brief: find Landsat samples.

@author: guanshikang

@type: script

Created on Wed Jan 08 15:17:06 2025, HONG KONG
"""
import os
import re
import ast
import glob
import tqdm
import json
import shutil
import random
import argparse
import numpy as np
import pandas as pd
import netCDF4 as nc
import geopandas as gpd
from osgeo import gdal, osr
from shapely.geometry import Point
from datetime import datetime
from FunctionalCode.CommonFuncs import CommonFuncs


class DataCleaning(CommonFuncs):
    """
    对目标区域的landsat和modis数据进行清洗，形成有效的配对数据.
    """
    def __init__(self, landsat_dir=None, modis_dir=None, ref_point_path=None, output_dir=None):
        super().__init__()
        self.landsat_dir = landsat_dir
        self.modis_dir = modis_dir
        self.points = self._read_csv(ref_point_path)
        self.output_dir = output_dir
        # shp_path = "/fossfs/skguan/data_fusion/ref_image/WRS2_descending.shp"
        # self.wrs = gpd.read_file(shp_path)  # self._calculate_tile
        # self.points['lcheck'] = self.points.apply(
        #     lambda df: self._check_tile(df['tile'], df['ref_ltile'], siteid=df['siteid']),
        #     axis=1
        # )
        # self.points.to_csv(ref_point_path.replace(".csv", "_new.csv"))

    def _read_csv(self, path):
        """
        对点坐标数据进行处理.

        Args:
            path (str): point file path.
        """
        points = pd.read_csv(path)
        # points = points.dropna()
        if 'tile' in points.keys():
            pattern = "\\d+[\\.]\\d+"
            string = str(points['tile'].iloc[0])
            regex = re.match(pattern, string)
            if regex:
                points['tile'] = points['tile'].astype(int)

            points['tile'] = points['tile'].astype(str)
        elif 'htile' in points.keys():
            points['tile'] = points['htile'].map(str) + \
                points['vtile'].map(lambda x: str(x).zfill(3))

        return points

    def _filter_tiles(self, log_files):
        pattern = r"There are bad lines in LC0[89]_L2SP_(?P<tile1>\d{6})|Went wrong for QA File of LC0[89]_L2SP_(?P<tile2>\d{6})"
        tiles = []
        for log_file in log_files:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    regex = re.match(pattern, line)
                    if regex is not None:
                        tile = regex.group("tile1") or regex.group("tile2")
                        tiles.append(tile)

        return tiles

    def _query_tile(self, site_id, lon, lat):
        """
        根据降轨文件计算tile (高纬可能存在部分重轨).

        Args:
            lon (float): longitude of POI.
            lat (float): latitude of POI.
        """

        point = Point(lon, lat)
        in_tile = self.wrs[self.wrs.contains(point)]
        if not in_tile.empty:
            tiles = []
            for i in range(len(in_tile)):
                path = in_tile.iloc[i]['PATH']
                row = in_tile.iloc[i]['ROW']
                tiles.append("{}{:0>3}".format(path, row))
            return tiles
        else:
            print(f"Did not find any tile for the {site_id}.")

    def _check_tile(self, tile, ref_tile, lcheck=np.nan, siteid=None, flag=0):
        """
        由于错误tile也提出了图像(重叠部分有一个tile全为0)，所以需要逐一排查.

        Args:
            tile (str): current tile.
            ref_tile (list): reference tile list. str.
            siteid (str): target site id.
            flag (int): preprocessing (0) or filtering stage (1).
        """
        if flag == 0:  # 预处理阶段
            if tile in ref_tile and len(ast.literal_eval(ref_tile)) == 1:
                return ""
            else:
                ref_dir = os.path.join(self.output_dir, "random_landsat", siteid)
                if not os.path.exists(ref_dir):
                    return ""
                else:
                    os.chdir(ref_dir)
                ref_path = os.listdir()[0]
                ref_data = gdal.Open(ref_path).ReadAsArray()[5:7, :, :]
                if np.all(ref_data == 0):
                    return "other"
                elif np.any(ref_data == 0):
                    return "cross"
                else:
                    return ""
        elif flag == 1:  # 数据筛选阶段
            if not pd.notna(lcheck):
                return tile
            else:
                tile_ls = ast.literal_eval(ref_tile)
                return [x for x in tile_ls if x != tile][0]
        else:
            """
            0: Preprocessing stage for determining how many tiles the site need.
            1: Filtering stage for tackling the zero site.
            """
            raise ValueError(f"Expected 0 or 1, but got {flag}")

    def cloud_process_landsat(self, log_file=None):
        """
        使用cloud mask对landsat数据进行云处理 QA_PIXEL
        """
        tiles = []
        if log_file is not None:
            tiles = self._filter_tiles(log_file)  # 进入内部进行筛选自定义
        # self.points = self.points[self.points['lcheck'] == "cross"]
        self.points['tile'] = self.points.apply(
            lambda df: self._check_tile(df['tile'], df['ref_ltile'], df['lcheck'], flag=1),
            axis=1
        )
        keys = set(self.points['tile'])
        key_num = len(keys)
        progressor = tqdm.tqdm(enumerate(keys))
        for i, key in progressor:
            if (key == "") or (pd.isna(key)) or (key not in tiles):
                continue

            progressor.set_description(
                "Processing tile of {0}: {1} / {2}".format(
                    key, i + 1, key_num
                )
            )
            # 对Landsat数据进行去云和光谱拼接处理.
            landsat_path = os.path.join(self.landsat_dir, key[:3], key[3:], "202[2-4]", "*")
            directories = glob.iglob(landsat_path)
            for directory in directories:
                files = os.listdir(directory)
                if len(files) == 0:
                    continue
                sr_pattern = "^.*B[1-7]+.TIF$"
                qa_pattern = "^.*QA_PIXEL.TIF$"
                qa_path = list(map(
                    lambda x: os.path.join(directory, x), filter(
                        lambda x: re.match(qa_pattern, x), files)
                ))[0]
                sr_path = list(map(
                    lambda x: os.path.join(directory, x), filter(
                        lambda x: re.match(sr_pattern, x), files)
                ))
                sr_path.sort()
                try:
                    file_info = self.get_fileinfo(sr_path[0])
                    samples = self.sampling(qa_path, sr_path, file_info, key=key)
                    prefix_name, date = self.acquire_productID(sr_path[0])

                except Exception as e:
                    print(e)
                    continue
                # doy = super().date2doy(date)
                # modis_path = os.path.join(self.output_dir, "modis", "proj")
                # pattern = f".*{doy}.*"
                # files = list(map(
                #     lambda x: os.path.join(modis_path, x), filter(
                #         lambda x: re.match(pattern, x),
                #         os.listdir(modis_path))
                # ))
                # if len(files) == 0:
                #     files = self.cloud_process_modis(doy)

                for sample in samples:
                    sub_img = sample['sub_img']
                    # lon_seed, lat_seed = sample['lon_seed'], sample['lat_seed']
                    site_id = sample['site_id']

                    landsat_dir = os.path.join(self.output_dir, "random_landsat", site_id)
                    if not os.path.exists(landsat_dir):
                        os.makedirs(landsat_dir)
                    landsat_path = os.path.join(landsat_dir, f"{prefix_name}.tif")
                    # if os.path.exists(landsat_path):
                    #     continue
                    super().save_image(
                        sr_path[0], landsat_path, sub_img,
                        sample['scol'], sample['srow']
                    )

    def modis_paired(self, src_path, tar_ls):
        """
        找到landsat ample对应的modis图像上的sample.

        Args:
            src_path (str): source file path.
        """
        min_lon, max_lat, max_lon, min_lat = super().get_extent(src_path)
        # file_info = super().get_fileinfo(src_path)
        # pcs = file_info['pcs']
        # gcs = file_info['gcs']
        # min_lon, max_lat = super().xy2lonlat(pcs, gcs, min_x, max_y)
        # max_lon, min_lat = super().xy2lonlat(pcs, gcs, max_x, min_y)
        for tar_file in tar_ls:
            min_lon_tar, max_lat_tar, max_lon_tar, min_lat_tar = super().get_extent(tar_file)
            file_info = super().get_fileinfo(tar_file)
            geo_trans = file_info['geotrans']

            is_contains = list(map(
                lambda x, y: x > y,
                [min_lon, min_lat, max_lon_tar, max_lat_tar],
                [min_lon_tar, min_lat_tar, max_lon, max_lat]
            ))
            if all(is_contains):
                min_col, max_col = list(map(
                    lambda x: int((x - min_lon_tar) / geo_trans[1]),
                    [min_lon, max_lon]
                ))
                min_row, max_row = list(map(
                    lambda x: int((x - max_lat_tar) / geo_trans[5]),
                    [max_lat, min_lat]
                ))
                sub_ds = file_info['src_ds']
                data = sub_ds.ReadAsArray()
                sub_data = data[:, min_row:max_row, min_col:max_col]
                sub_data *= 0.0001
                if np.sum(sub_data) == 0:
                    continue

                return {
                    'tar_file': tar_file,
                    'sub_data': sub_data,
                    'min_lon': min_lon,
                    'max_lat': max_lat
                }
            else:
                continue

    def cloud_process_modis(self, date):
        """
        对MODIS(MO/YD09)数据进行预处理.

        Args:
            date (str): day of year.
        """
        input_dir = os.path.join(self.modis_path, date[:4], date[4:])
        file_ls = os.listdir(input_dir)
        china_tiles = [
            'h23v04', 'h23v05', 'h24v04', 'h24v05', 'h25v03',
            'h25v04', 'h25v05', 'h25v06', 'h26v03', 'h26v04',
            'h26v05', 'h26v06', 'h27v04', 'h27v05', 'h27v06',
            'h28v05', 'h28v06', 'h28v07', 'h28v08', 'h29v06',
            'h29v07', 'h29v08'
        ]
        pattern = ".*(h\\d+v\\d+).*"
        file_ls = list(filter(
            lambda x: re.match(pattern, x).group(1) in china_tiles,
            file_ls
        ))
        output_dir = os.path.join(self.output_dir, "modis", "proj")
        for file in file_ls:
            input_path = os.path.join(input_dir, file)
            self.geo_correction(input_path, output_dir)

        files = os.listdir(output_dir)
        files = list(map(
            lambda x: os.path.join(output_dir, x), filter(
                lambda x: re.match(".*%s.*" % date, x), files
            )
        ))

        return files

    def geo_correction(self, input_file, output_dir, height=2400, width=2400, band_num=6):
        """
        对MODIS数据进行投影坐标系的校正.

        Args:
            input_file (str): corresponding MODIS file.
            output_dir (str): output file directory.
            height (int): output array height.
            width (int): output array width.
            band_num (int): output array bands.
        """
        dst = gdal.Open(input_file)
        sub_dsts = dst.GetSubDatasets()
        sub_dsts_sr = list(filter(
            lambda x: re.match(".*b0[1-467].*", x[0]), sub_dsts
        ))
        sr_data = np.zeros((band_num, height, width))
        # qa_data = np.zeros((height, width))
        for i, sub_dst in enumerate(sub_dsts_sr):
            sub_dst_name = sub_dst[0]
            dst = gdal.Open(sub_dst_name)
            sr_data[i, :, :] = dst.ReadAsArray()

        band_seq = [2, 3, 0, 1, 4, 5]
        sr_data = sr_data[band_seq, :, :]

        base_name = os.path.basename(input_file).replace("hdf", "tif")
        output_path = os.path.join(output_dir, base_name)

        super().save_image(sub_dst_name, output_path, sr_data)
        proj_path = output_path.replace(".tif", "_proj.tif")
        self.reproject_to_geographic(output_path, proj_path)

        os.remove(output_path)

    def acquire_productID(self, src_path):
        """
        处理输入数据的文件名获得日期，行列号等属性.

        Args:
            src_path (str): reference data.
        """
        # LC08_L2SP_145030_20200509_20200820_02_T1_QA_PIXEL.TIF
        pattern = ".*(\\d{8}).\\d{8}.*T\\d"
        base_name = os.path.basename(src_path)
        regex = re.match(pattern, base_name)
        prefix_name = regex.group(0)
        date = regex.group(1)

        return prefix_name, date

    def sampling(self, qa_path, sr_path, file_info, key=None, image_size=256, seed_num=10, cloud_cover=95):
        """
        对Landsat数据以指定窗口大小进行采样.
        或以经纬度中心膨胀.
        cloud_cover需要满足一定要求.

        Args:
            qa_path (str): quality control file path.
            sr_path (str): surface reflectance file path.
            file_info (list): reference file info.
            key (str): tile of point location.
            image_size (int):  image size of samples for SR.
            random_seed (int): number of subimgs.
        """
        if key is None:
            data_array = file_info['src_ds'].ReadAsArray()
            width = data_array.shape[1]
            length = data_array.shape[2]
            lon_num = int(length / image_size)
            lat_num = int(width / image_size)

            lon_seeds = random.sample(range(0, lon_num), k=seed_num)
            lat_seeds = random.sample(range(0, lat_num), k=seed_num)

            for lon_seed, lat_seed in zip(lon_seeds, lat_seeds):
                lon_seed, lat_seed = 18, 26
                srow, scol = [i * image_size for i in [lat_seed, lon_seed]]
                erow, ecol = [i + image_size for i in [srow, scol]]

                sub_img = data_array[:, srow:erow, scol:ecol]

                cloud_cover = self.cloud_coverage(sub_img)

                if cloud_cover > 100:
                    continue
                else:
                    sub_img = sub_img
                    yield {
                        'sub_img': sub_img,
                        'lon_seed': lon_seed,
                        'lat_seed': lat_seed
                    }
        else:
            df = self.points['tile']
            df = self.points[self.points['tile'] == key]
            step = int(image_size / 2)
            size = max(file_info['width'], file_info['length'])
            for id, lon, lat in zip(df['siteid'], df['lon'], df['lat']):
                x, y, _ = self.lonlat2xy(file_info['pcs'], file_info['gcs'], lon, lat)
                row, col = self.xy2rowcol(file_info['geotrans'], x, y)
                srow, erow = [row - step, row + step]
                scol, ecol = [col - step, col + step]
                flag = [(x < 0) | (x > size) for x in [srow, erow, scol, ecol]]
                if any(flag):
                    continue
                sub_offset = {'xoff': scol, 'yoff': srow,
                              'xsize': ecol - scol, 'ysize': erow - srow}
                cloud_mask = self.cloud_mask(qa_path, **sub_offset)
                if cloud_mask is None:
                    continue
                clear_mask = ~cloud_mask[np.newaxis, :, :]
                sub_img = self.spectral_concate(sr_path, **sub_offset)
                if sub_img is None:
                    print(f"{id} has been skipped for the bad lines of source data.")
                    continue

                # clear_sky = sr_data * ~c_mask
                sub_img = np.concatenate((sub_img, clear_mask), axis=0)

                # cloud_cover = self.cloud_coverage(sub_img)

                # if cloud_cover > 100:
                #     continue

                yield {
                        'site_id': id,
                        'sub_img': sub_img,
                        'srow': srow,
                        'scol': scol
                }

    def cloud_coverage(self, data_array):
        """
        calculate the cloud coverage (clear pixel ratio).

        Args:
            data_array (ndarray): the data array to be calculated.
        """
        bg_pixels = np.sum(data_array[0, :, :] == 0)
        cloud_ratio = bg_pixels / (data_array.shape[1] * data_array.shape[2])
        cloud_ratio *= 100

        return cloud_ratio

    def reproject_to_geographic(self, src_path, output_path):
        """
        将原始数据重投影并采样到地理坐标系.

        Args:
            sr_path (str): reflectance file path.
            output_path (str): output file path.
        """
        # 参考图像信息
        src_fileinfo = self.get_fileinfo(src_path)

        src_ds = src_fileinfo['src_ds']
        width = src_fileinfo['width']
        length = src_fileinfo['length']

        src_srs = osr.SpatialReference()
        src_srs.ImportFromWkt(src_fileinfo['pcs'].ExportToWkt())

        # 目标图像信息
        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromEPSG(4326)

        gdal.Warp(
            output_path,
            src_ds,
            width=width,
            height=length,
            srcSRS=src_srs,
            dstSRS=dst_srs,
            resampleAlg=gdal.GRA_Bilinear
        )

    def spectral_concate(self, sr_path, xoff=0, yoff=0, xsize=None, ysize=None):
        """
        对landsat数据进行波段拼接

        Args:
            sr_path (str): reflectance file path.
            xoff (int): offset for x (col) to read.
            yoff (int): offset for y (row) to read.
            xsize (int): length of x direction.
            ysize (int): width of y direction.
        """
        sr_ds = map(lambda x: gdal.Open(x), sr_path)
        try:
            sr_data = list(map(
                lambda x: x.GetRasterBand(1).ReadAsArray(xoff, yoff, xsize, ysize),
                sr_ds
            ))
        except AttributeError:
            print(f"There are 0 kb files for {os.path.split(sr_path[0])[-2]}")
            return None
        if any(x is None for x in sr_data):
            for i, band in enumerate(sr_data):
                if band is None:
                    print(f"There are bad lines in {os.path.split(sr_path[i])[-1]}")
            return None
        else:
            data = np.array(sr_data)
            return data


    def cloud_mask(self, qa_path, xoff=0, yoff=0, xsize=None, ysize=None):
        """
        返回landsat数据的云掩膜

        Args:
            qa_path (str): cloud mask file path.
            xoff (int): offset for x (col) to read.
            yoff (int): offset for y (row) to read.
            xsize (int): length of x direction.
            ysize (int): width of y direction.
        """
        try:
            qa_ds = gdal.Open(qa_path)
            qa_band = qa_ds.GetRasterBand(1)
            qa_data = qa_band.ReadAsArray(xoff=xoff, yoff=yoff,
                                          win_xsize=xsize, win_ysize=ysize)
            cloud_mask = qa_data & 0b11111
        except Exception:
            print(f"Went wrong for QA File of {os.path.split(qa_path)[-1]}.")
            return None
        else:
            return cloud_mask.astype(bool)

    def find_landsat_tiles(self, ref_dir, image_size: int = 256):
        """
        根据经纬度找到所在的landsat tiles.

        Args:
            ref_dir (str): reference file directory.
            image_size (int, optional): desired image size. Defaults to 256.
        """
        if not os.path.exists(ref_dir):
            os.makedirs(ref_dir)
        if len(os.listdir(ref_dir)) == 0:
            print("Searching reference images!")
            self._create_ref_dir(ref_dir)
        ref_files = map(
            lambda x: os.path.join(ref_dir, x),
            os.listdir(ref_dir)
        )
        point_num = len(self.points)
        temp_dict = {
            "siteid": [""] * point_num,
            "tile": [""] * point_num,
            "ltile": [""] * point_num,  # 高纬地区可能有重叠
            "lon": self.points['lon'],
            "lat": self.points['lat'],
            "row": [0] * point_num,
            "col": [0] * point_num
        }
        flag = []
        progressor = tqdm.tqdm(ref_files)
        for file in progressor:
            progressor.set_description("Processing {}.".format(file))
            min_lon, max_lat, max_lon, min_lat = self.get_extent(file)
            file_info = self.get_fileinfo(file)
            progressor1 = tqdm.tqdm(enumerate(zip(
                self.points['siteid'], self.points['lat'], self.points['lon']
            )))
            for i, (siteid, lat, lon) in progressor1:
                progressor1.set_description(
                    "{0} / {1}".format(i + 1, point_num)
                )
                x, y, _ = self.lonlat2xy(file_info['pcs'], file_info['gcs'], lon, lat)
                is_contains = list(map(
                    lambda m, n: m > n,
                    [x, max_lat, max_lon, y],
                    [min_lon, y, x, min_lat]
                ))
                if all(is_contains):
                    row, col = self.xy2rowcol(file_info['geotrans'], x, y)
                    dst = gdal.Open(file)
                    data = dst.GetRasterBand(1).ReadAsArray()
                    if data is None:
                        continue
                    size = max(data.shape)
                    step = int(image_size / 2)
                    srow, erow = [row - step, row + step]
                    scol, ecol = [col - step, col + step]
                    is_illegal = [(x < 0) | (x > size) for x in [srow, erow, scol, ecol]]
                    if any(is_illegal) | (i in flag):
                        continue
                    sub_data = data[srow:erow, scol:ecol]
                    ratio = np.sum((sub_data != 65535) & (np.max(sub_data) != 0)) / (image_size * image_size)
                    if ratio > 0.95:
                        try:
                            tile = re.match(".*LC0[89]_L2SP_(\\d+).*", file).group(1)
                            temp_dict['siteid'][i] = siteid
                            if temp_dict['tile'][i] == "":
                                temp_dict['tile'][i] = tile
                            else:
                                temp_dict['ltile'][i] = tile
                            temp_dict['row'][i] = row  # 不采用，提点时重新计算row_col，但保留作为参考.
                            temp_dict["col"][i] = col
                            flag.append(i)
                        except Exception as e:
                            print(e)
        df = pd.DataFrame(temp_dict)
        df.to_csv("/fossfs/skguan/data_fusion/sites_landsat.csv")

    def _create_ref_dir(self, ref_dir: str, band_num: int = 1,
                        year: int = 2024, month: int = 7):
        """
        使用指定波段创建图像参考库.

        Args:
            ref_dir (str): directory of reference images.
            band_num (int, optional): selected reference band. Defaults to 1.
            year (int, optional): reference year. Defaults to 2024.
            month (int, optional): reference month. Defaults to 7.
        """
        target_name = "*{0}{1}*_{2}*_B{3}.TIF".format(  # 文件名
                year, str(month).zfill(2), year, band_num
        )
        for x in range(100, 200):
            for y in range(0, 100):
                pattern = os.path.join(
                    self.landsat_dir, str(x), str(y).zfill(3),
                    str(year), "*", target_name
                )
                files = glob.glob(pattern)
                if len(files) == 0:
                    continue
                else:
                    file = files[0]
                    file_name = os.path.basename(file)
                    output_path = os.path.join(ref_dir, file_name)
                    self.reproject_to_geographic(file, output_path)

    def create_labels(self, output_dir: str):
        """
        寻找landsat9晴空标签.

        Args:
            output_dir (str): output directory for labels.
        """
        input_dir = os.path.join(self.output_dir, "random_landsat")
        files = glob.iglob(os.path.join(input_dir, "*", "LC09*"))
        for file in tqdm.tqdm(files):
            dst = gdal.Open(file)
            data = dst.GetRasterBand(8).ReadAsArray()
            bg_pixels = np.sum(data[-1] == 0)
            cloud_ratio = bg_pixels / (data.shape[0] * data.shape[1])
            cloud_ratio *= 100
            if cloud_ratio == 0:
                site_id = file.split("/")[-2]
                output_path = os.path.join(output_dir, site_id)
                if not os.path.exists(output_path):
                    os.mkdir(output_path)
                shutil.copy(file, output_path)

    def concat2nc(self, input_dir):
        """
        把输入数据按时间顺序叠加在一起保存成nc文件.

        Args:
            input_dir (str): input file directory.
        """
        paths = map(
            lambda x: os.path.join(input_dir, x),
            filter(lambda y: re.match(r"r\d+?", y), os.listdir(input_dir))
        )
        progressor = tqdm.tqdm(paths)
        for i, path in enumerate(progressor):
            os.chdir(path)
            site_name = os.path.split(path)[-1]
            progressor.set_description(
                "{}: {} has been processed".format(
                    i + 1, site_name
                )
            )
            output_name = site_name + ".nc"
            # if os.path.exists(output_name):
            #     continue
            files = list(filter(
                lambda x: re.match("LC0[89].*_\\d{6}_202[2-4]\\w*", x),
                os.listdir(path)
            ))
            if len(files) == 0:
                print(f"{site_name} is skipped for the empty directory.")
                continue
            files.sort(key=lambda x: re.findall("\\d{8}", x)[0])
            flag = [x[3] for x in files]
            data = list(map(lambda x: gdal.Open(x).ReadAsArray(), files))
            time = list(map(
                lambda x: datetime.strptime(x, "%Y%m%d"),
                map(lambda y: re.findall("\\d{8}", y)[0],
                    files)
            ))
            file_info = self.get_fileinfo(files[0])
            lon = np.arange(
                file_info['geotrans'][0],
                file_info['geotrans'][0] +
                file_info['length'] * file_info['geotrans'][1],
                file_info['geotrans'][1]
            )
            lat = np.arange(
                file_info['geotrans'][3],
                file_info['geotrans'][3] +
                file_info['width'] * file_info['geotrans'][5],
                file_info['geotrans'][5]
            )

            band_num = data[0].shape[0]
            try:
                self.save_nc(output_name, data, lon, lat, time, band_num,
                             mask=-1, flag=flag)
            except Exception as e:
                print(f"{site_name} arised an error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--category', required=True, help='Requested Funcation')
    args = parser.parse_args()
    category = args.category

    IMAGE_SIZE = 256
    point_path = "/fossfs/skguan/data_fusion/split_dataset/original_sites/random_sites_new.csv"
    ref_dir = "/fossfs/skguan/data_fusion/ref_image"
    landsat_dir = "/fossfs/skguan/DATA/10.Landsat_L2_p"
    output_dir = "/fossfs/skguan/data_fusion"
    label_dir = "/fossfs/skguan/data_fusion/labels"
    dc = DataCleaning(landsat_dir, ref_point_path=point_path, output_dir=output_dir)
    # dc.find_landsat_tiles(ref_dir, image_size=IMAGE_SIZE)  # 可以直接计算所在tile
    # dc.cloud_process_landsat(["slurm-2361279.out", "slurm-2367512.out", "slurm-2372150.out", "slurm-2375787.out", "slurm-2381215.out", "slurm-2382009.out"])
    if category == "L":
        dc.create_labels(label_dir)
    elif category == "C":
        concat_path = "/fossfs/skguan/data_fusion/random_landsat"
        dc.concat2nc(concat_path)


if __name__ == "__main__":
    main()
