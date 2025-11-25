# -*- encoding: utf-8 -*-
import os
import re
import glob
import tqdm
import math
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from osgeo import gdal, osr
from datetime import datetime
from samples_generation_Landsat import DataCleaning


class ExtractMODSamples(DataCleaning):
    def __init__(self, modis_dir, ref_point_path, ref_image_dir, ref_out_dir, output_dir):
        super().__init__(ref_point_path=ref_point_path)
        self.modis_dir = modis_dir
        self.ref_image_dir = ref_image_dir
        self.ref_out_dir = ref_out_dir
        self.output_dir = output_dir
        self.points = pd.read_csv(ref_point_path)
        # 下面为直接计算modis的tile，实测准确率较高，因为原来的搜寻点只更新一次
        # self.points['ref_mtile'] = self.points.apply(
            # lambda df: self._calculate_tile(df['lon'], df['lat']), axis=1
        # )
        # self.points.to_csv(ref_point_path.replace(".csv", "_new.csv"))

    def _lonlat2xy(self, pcs, gcs, lon, lat):
        """
        经纬度坐标转投影坐标.
        """
        ct = osr.CoordinateTransformation(gcs, pcs)
        coordinates = ct.TransformPoint(lon, lat)  # UTM needs lat first, lon second.
        return coordinates[0], coordinates[1], coordinates[2]

    def get_modis_info(self, modis_path):
        dst = gdal.Open(modis_path)
        sub_dsts = dst.GetSubDatasets()
        sub_dsts_sr = list(filter(
            lambda x: re.match(".*(?:b0[1-7]|[sv]zen|raz).*", x[0]),
            sub_dsts
        ))
        sub_dst_name = sub_dsts_sr[0]
        file_info = super().get_fileinfo(sub_dst_name[0])

        return sub_dsts_sr, file_info

    def _calculate_tile(self, lon, lat, resolution=1113194):
        """
        计算MODIS所在的tile.

        Args:
            lon (float): longitude of POI.
            lat (float): latitude of POI.
        """
        wgs_proj = osr.SpatialReference()
        wgs_proj.ImportFromEPSG(4326)
        sin_str = 'PROJCS["World_Sinusoidal",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Sinusoidal"],PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",0.0],UNIT["Meter",1.0]]'
        sin_proj = osr.SpatialReference()
        sin_proj.ImportFromProj4(sin_str)

        ct = osr.CoordinateTransformation(wgs_proj, sin_proj)
        sin_x, sin_y, _ = ct.TransformPoint(lat, lon)
        sin_row = math.floor((sin_y - 10001965.729312722) * -1 / resolution)
        sin_col = math.floor((sin_x + 20037508.342789244) / resolution)

        return f"h{sin_col}v{str(sin_row).zfill(2)}"

    def extract_samples(self, image_size: int = 256, category: str = None):
        """
        提取样本主函数.

        Args:
            image_size (int, optional): 样本窗口大小. Defaults to 256.
        """
        existing_sites = os.listdir(self.output_dir)
        self.points = self.points[~self.points['siteid'].isin(existing_sites)]
        keys = set(self.points['ref_mtile'])
        key_num = len(keys)
        progressor = tqdm.tqdm(enumerate(keys))
        for i, key in progressor:
            progressor.set_description(
                "Processing tile of {0}: {1} / {2}".format(
                    key, i + 1, key_num
                )
            )
            if key == "":
                continue
            df = self.points[self.points['ref_mtile'] == key]
            files = glob.iglob(
                os.path.join(self.modis_dir, f"202[2-4]/*/*{key}*.hdf")
            )
            for file in files:
                try:
                    sub_dsts_sr, file_info = self.get_modis_info(file)
                except Exception as e:
                    print("It occurs a crash, and the cause is {}".format(e))
                    print(files, type(files))
                    continue

                for id, lon, lat in zip(df['siteid'], df['lon'], df['lat']):
                    x, y, _ = self._lonlat2xy(file_info['pcs'], file_info['gcs'], lon, lat)
                    row, col = super().xy2rowcol(file_info['geotrans'], x, y)
                    scale = 4 if category == "A" else 2
                    step = int(image_size / scale)
                    srow, erow = [row - step, row + step]
                    scol, ecol = [col - step, col + step]
                    flag = [(x < 0) | (x > file_info['width']) for x in [srow, erow, scol, ecol]]
                    if any(flag):
                        continue
                    sr_data = list(map(
                        lambda x: gdal.Open(x[0]).ReadAsArray(
                            xoff=scol, yoff=srow,
                            xsize=ecol - scol, ysize=erow - srow
                        ),
                        sub_dsts_sr
                    ))
                    sr_data = np.array(sr_data)
                    doy = re.match(".*A(\\d{7}).*", file).group(1)
                    site_dir = os.path.join(self.output_dir, id)
                    # print(f"\n{id}")
                    if not os.path.exists(site_dir):
                        os.mkdir(site_dir)
                    # name = "{0}_{1}_ori.tif" if category == "A" else "{0}_{1}.tif"
                    name = "{0}_{1}.tif"
                    output_path = os.path.join(site_dir, name.format(id, doy))
                    if os.path.exists(output_path):
                        continue
                    geotrans = list(file_info['geotrans'])
                    geotrans[0] = geotrans[0] + geotrans[1] * scol + geotrans[2] * srow
                    geotrans[3] = geotrans[3] + geotrans[4] * scol + geotrans[5] * srow
                    try:
                        super().save_image(sub_dsts_sr[0][0], output_path, sr_data, geo_trans=geotrans)
                        # if category == "A":
                        #     resample_path = output_path.replace("_ori.tif", ".tif")
                        #     self.post_resample(output_path, resample_path, scale_factor=2)
                        #     os.remove(output_path)
                    except Exception as e:
                        print("%s.\n%s is invalid." % (e, file))
                        continue

    # 重投影参考影像（寻找tiles）
    def resample_ref_images(self):
        ref_path = os.path.join(self.ref_image_dir, "2014", "073")
        files = list(map(
            lambda x: os.path.join(ref_path, x),
            os.listdir(ref_path)
        ))
        for file in tqdm.tqdm(files):
            dst = gdal.Open(file)
            sub_dsts = dst.GetSubDatasets()
            sub_dsts_sr = list(filter(
                lambda x: re.match(".*b0[1-467].*", x[0]), sub_dsts
            ))
            sr_data = np.zeros((2, 4800, 4800))
            base_name = os.path.basename(file).replace(".hdf", "_ori.tif")
            for i, sub_dst in enumerate(sub_dsts_sr):
                sub_dst_name = sub_dst[0]
                dst = gdal.Open(sub_dst_name)
                sr_data[i, :, :] = dst.ReadAsArray()
            ref_out_path = os.path.join(self.ref_out_dir, base_name)
            super().save_image(sub_dst_name, ref_out_path, sr_data)

            final_path = ref_out_path.replace("_ori.tif", ".tif")
            super().reproject_to_geographic(ref_out_path, final_path)
            os.remove(ref_out_path)

    # 提取后重采样
    def post_resample(self, input_path, output_path, scale_factor=1):
        ref_fileinfo = super().get_fileinfo(input_path)
        input_dst = gdal.Open(input_path)
        in_proj = input_dst.GetProjection()
        band_num = input_dst.RasterCount
        in_band = input_dst.GetRasterBand(1)
        dtype = in_band.DataType

        driver = gdal.GetDriverByName("GTiff")
        out_dst = driver.Create(
            output_path,
            ref_fileinfo['length'] * scale_factor,
            ref_fileinfo['width'] * scale_factor,
            band_num,
            dtype
        )
        geo_trans = list(ref_fileinfo["geotrans"])
        geo_trans[1] = geo_trans[1] / scale_factor
        geo_trans[5] = geo_trans[5] / scale_factor
        out_dst.SetGeoTransform(geo_trans)
        out_dst.SetProjection(ref_fileinfo['pcs'].ExportToWkt())

        gdal.ReprojectImage(
            input_dst, out_dst,
            in_proj,
            ref_fileinfo['pcs'].ExportToWkt(),
            gdal.GRA_Bilinear
        )

    # 寻找每个点需要用到的tiles
    def find_tiles(self, image_size):
        files = list(map(
            lambda x: os.path.join(self.ref_out_dir, x),
            os.listdir(self.ref_out_dir)
        ))
        point_num = len(self.points)
        temp_dict = {
            "siteid": [""] * point_num,
            "lat": self.points['lat'],
            "lon": self.points['lon'],
            "mtile": [""] * point_num,
        }
        flag = []
        progressor = tqdm.tqdm(files)
        for file in progressor:
            progressor.set_description("Processing {}.".format(file))
            min_lon, max_lat, max_lon, min_lat = super().get_extent(file)
            file_info = super().get_fileinfo(file)
            band = file_info['src_ds'].GetRasterBand(1)
            progressor1 = tqdm.tqdm(enumerate(zip(
                self.points['siteid'], self.points['lat'], self.points['lon']
            )))
            for i, (siteid, lat, lon) in progressor1:
                is_contains = list(map(
                    lambda x, y: x > y,
                    [lon, max_lat, max_lon, lat],
                    [min_lon, lat, lon, min_lat]
                ))
                if all(is_contains):
                    x, y, _ = self._lonlat2xy(file_info['pcs'], file_info['gcs'], lon, lat)
                    row, col = super().xy2rowcol(file_info['geotrans'], x, y)

                    # sub_dsts = dst.GetSubDatasets()
                    # sub_dsts_sr = list(filter(
                    #     lambda x: re.match(".*b01.*", x[0]), sub_dsts
                    # ))
                    # sub_dst_name = sub_dsts_sr[0]
                    # dst = gdal.Open(sub_dst_name)
                    # sr_data = dst.ReadAsArray()
                    step = int(image_size / 2)
                    srow, erow = [row - step, row + step]
                    scol, ecol = [col - step, col + step]
                    is_illegal = [(x < 0) | (x > 4800) for x in [srow, erow, scol, ecol]]
                    if any(is_illegal) | (i in flag):
                        continue
                    sub_data = band.ReadAsArray(
                        xoff=scol, yoff=srow,
                        win_xsize=ecol - scol, win_ysize=erow - srow
                    )
                    ratio = np.sum(sub_data[:, :] != 0) / (image_size * image_size)
                    if ratio > 0.5 or temp_dict['siteid'][i] == "":
                        tile = re.match(".*(h\\d{2}v\\d{2}).*", file).group(1)
                        temp_dict['siteid'][i] = siteid
                        temp_dict['mtile'][i] = tile
                        flag.append(i)

        df = pd.DataFrame(temp_dict)
        df.to_csv("/fossfs/skguan/data_fusion/split_dataset/original_sites/random_sits_modis.csv")

    def pair_extraction(self, src_dir, ref_dir, sub_img_dir, category=None):
        sites = os.listdir(ref_dir)
        for site in sites:
            try:
                ref_path = os.path.join(ref_dir, site)
                ref_file = os.path.join(ref_path, os.listdir(ref_path)[1])
                # ref_info = super().get_fileinfo(ref_file)
                min_x, max_y, _, _ = super().get_extent(ref_file)

                src_path = os.path.join(src_dir, site)
                # src_file = os.path.join(src_path, os.listdir(src_path)[0])
                # proj_file = src_file.split(".")[0] + "_proj.tif"
                # super().reproject_image(src_file, ref_file, proj_file)
                # proj_info = super().get_fileinfo(proj_file)
                # left_row, left_col = super().xy2rowcol(
                #     proj_info['geotrans'],
                #     min_x,
                #     max_y
                # )
                # right_row, right_col = super().xy2rowcol(
                #     proj_info['geotrans'],
                #     max_x,
                #     min_y
                # )
                # os.remove(proj_file)
                pattern = ".+_202[2-4]\\d+.tif"
                files = list(map(
                    lambda x: os.path.join(src_path, x), filter(
                        lambda x: re.match(pattern, x), os.listdir(src_path))
                    ))
            except Exception as e:
                print(e)
                continue
            output_dir = os.path.join(sub_img_dir, site)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            for file in files:
                proj_file = file.split(".")[0] + "_proj.tif"
                super().reproject_image1(file, ref_file, proj_file)
                proj_info = super().get_fileinfo(proj_file)
                output_name = os.path.basename(file)
                output_path = os.path.join(output_dir, output_name)
                # gdal.Warp(
                #     output_path,
                #     proj_file,
                #     outputBounds=(min_x, min_y, max_x, max_y),
                #     xRes=proj_info['geotrans'][1],
                #     yRes=-proj_info['geotrans'][5],
                #     targetAlignedPixels=True,
                #     resampleAlg=gdal.GRA_NearestNeighbour
                # )
                row, col = super().xy2rowcol(proj_info['geotrans'],
                                             min_x, max_y)
                sub_data = gdal.Open(proj_file).ReadAsArray()
                if category == "A":
                    sub_data = sub_data[:, row:row + 18, col:col + 18]  # A
                    x_off = col
                    y_off = row
                elif category == "Q":
                    sub_data = sub_data[:, row - 1:row + 35, col - 1:col + 35]  # Q
                    x_off = col - 1
                    y_off = row - 1
                else:
                    print("Check the category of pair extraction.")
                super().save_image(proj_file, output_path, sub_data,
                                   x_off=x_off, y_off=y_off)
                os.remove(proj_file)

    def AvgPool(self, data):
        x = torch.tensor(data).to(torch.float)
        x = nn.AvgPool2d(kernel_size=2, stride=1)(x)
        x = x.to(torch.int16)

        return x.numpy()

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
        with tqdm.tqdm(paths) as pbar:
            for path in paths:
                os.chdir(path)
                site_name = os.path.split(path)[-1]
                output_name = site_name + ".nc"
                # if os.path.exists(output_name):
                #     pbar.update(1)
                #     continue
                files = list(filter(
                    lambda x: re.match(".+_202[2-4]\\w*.tif", x),
                    os.listdir(path)
                ))
                if len(files) == 0:
                    pbar.update(1)
                    continue
                files.sort()
                data = list(map(lambda x: gdal.Open(x).ReadAsArray(), files))
                # data = list(map(lambda x: self.AvgPool(x), data))  # 完全对齐一个像素的漂移进行池化操作
                time = list(map(
                    lambda x: datetime.strptime(x, "%Y%j"),
                    map(lambda x: re.findall("\\d{7}", x)[0],
                        files)
                ))
                file_info = super().get_fileinfo(files[0])
                lon = np.arange(
                    file_info['geotrans'][0],
                    file_info['geotrans'][0] +
                    file_info['length'] * file_info['geotrans'][1],
                    file_info['geotrans'][1]
                )[:file_info['length']]
                lat = np.arange(
                    file_info['geotrans'][3],
                    file_info['geotrans'][3] +
                    file_info['width'] * file_info['geotrans'][5],
                    file_info['geotrans'][5]
                )[:file_info['width']]

                band_num = data[0].shape[0]
                try:
                    super().save_nc(output_name, data, lon, lat, time, band_num)
                except Exception as e:
                    print(f"{site_name} arised an error: {e}")
                pbar.update(1)


def main():
    # user input
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--category', required=True, help='MODIS Data Type')
    args = parser.parse_args()
    category = args.category

    IMAGE_SIZE = 128
    # category = "A"
    modis_dir = "/fossfs/DATAPOOL/MODIS/MOD09{}1_C61".format(category)
    points_path = "/fossfs/skguan/data_fusion/split_dataset/original_sites/random_sites_new.csv"
    output_dir = "/fossfs/skguan/data_fusion/original_modis/MOD09{}1".format(category)
    ref_path = "/fossfs/DATAPOOL/MODIS/MOD09Q1_C61"  # for find_tiles
    ref_out_dir = "/fossfs/skguan/MOD_Samples/ref_image"  # for find_tiles
    landsat_dir = "/fossfs/skguan/data_fusion/random_landsat"
    sub_img_path = "/fossfs/skguan/data_fusion/modis/MOD09{}1".format(category)
    es = ExtractMODSamples(modis_dir, points_path, ref_path, ref_out_dir, output_dir)
    # es.extract_samples(IMAGE_SIZE, category)
    # es.find_tiles(IMAGE_SIZE)
    # es.pair_extraction(output_dir, landsat_dir, sub_img_path, category=category)
    # es.concat2nc(sub_img_path)


if __name__ == "__main__":
    main()
