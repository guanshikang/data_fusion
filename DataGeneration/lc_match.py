# -*- encoding: utf-8 -*-
"""
@brief: Statistics for Land Cover in Labels

@author: guanshikang

@type: script

Created on Wed Aug 13 11:05:33 2025, HONG KONG
"""
import os
import re
import glob
import math
import random
import numpy as np
import pandas as pd
from osgeo import gdal
from global_land_mask import globe
from FunctionalCode.CommonFuncs import CommonFuncs

def search_by_site():
    GLC_dir = "/fossfs/skguan/DATA/7.GLC30_p"
    ref_files = glob.glob(os.path.join(GLC_dir, "./*/*Annual*.tif"))[300:600]
    output_dir = "/fossfs/skguan/data_fusion/glc"
    src_path = "/fossfs/skguan/data_fusion/landsat"
    sites_name = os.listdir(src_path)
    cf = CommonFuncs()
    src_files = list(map(
        lambda y: os.path.join(y, os.listdir(y)[1]), map(
            lambda x: os.path.join(src_path, x), sites_name)
    ))
    length = len(src_files)
    attri = {'site': [''] * length, 'point': [''] * length,
             'pcs': [''] * length, 'gcs': [''] * length}
    for i, (site, src_file) in enumerate(zip(sites_name, src_files)):
        attri['site'][i] = (site, src_file)
        attri['point'][i] = cf.get_extent(os.path.join(src_path, src_file))
        file_info = cf.get_fileinfo(src_file)
        attri['pcs'][i] = file_info['pcs']
        attri['gcs'][i] = file_info['gcs']
    for i, ref_file in enumerate(ref_files):
        file_info = cf.get_fileinfo(ref_file)
        data = file_info['src_ds'].GetRasterBand(23).ReadAsArray()
        for site, point, pcs, gcs in zip(attri['site'], attri['point'], attri['pcs'], attri['gcs']):
            lat1, lon1 = cf.xy2lonlat(pcs, gcs, point[0], point[1])
            lat2, lon2 = cf.xy2lonlat(pcs, gcs, point[2], point[3])
            row1, col1 = cf.xy2rowcol(file_info['geotrans'], lon1, lat1)
            row2, col2 = cf.xy2rowcol(file_info['geotrans'], lon2, lat2)
            if (row1 < 0) or (row2 > file_info['width']) or \
                (col1 < 0) or (col2 > file_info['length']):
                    continue
            else:
                print(f"Processing {site[0]} at {i}th file.")
                row_buffer = 10 if row1 - 10 >= 0 else row1
                col_buffer = 10 if col1 - 10 >= 0 else col1
                sub_data = data[row1 - row_buffer:row2 + 10, col1 - col_buffer:col2 + 10]
                proj_path = os.path.join(output_dir, site[0] + "_proj.tif")
                cf.save_image(ref_file, proj_path, sub_data,
                              x_off=col1 - col_buffer, y_off=row1 - row_buffer)
                output_path = os.path.join(output_dir, site[0] + ".tif")
                cf.reproject_image(proj_path, site[1], output_path, gdal.GRA_NearestNeighbour)
                os.remove(proj_path)

def search_by_lonlat():
    point_csv = pd.read_csv("/fossfs/skguan/data_fusion/split_dataset/original_sites/rpt_2025(1).csv")
    cf = CommonFuncs()
    glc_dir = "/fossfs/skguan/DATA/7.GLC30_p"
    output_dir = "/fossfs/skguan/data_fusion/rpt_glc"
    for i in range(len(point_csv)):
        lon = point_csv['lon'][i]
        lat = point_csv['lat'][i]
        siteid = point_csv['siteid'][i]
        if "glcfile" in point_csv.keys():
            glc_file = point_csv['glcfile'][i]
        else:
            lon_zone = 5 * math.floor(int(lon) / 5)
            lat_zone = 5 * math.ceil(int(lat) / 5)
            EorW = "E" if lon_zone > 0 else "W"
            NorS = "N" if lat_zone > -5 else "S"
            lon_zone0 = lon_zone - 5 if (lon_zone % 10 == 5) & (EorW == "E") else lon_zone
            # lat_zone = abs(lat_zone) + 5 if NorS == "N" else lat_zone
            directory = f"GLC_FCS30D_19852022maps_{EorW}{abs(lon_zone0)}-{EorW}{abs(lon_zone0) + 5}"
            file = f"GLC_FCS30D_20002022_{EorW}{abs(lon_zone)}{NorS}{abs(lat_zone)}_Annual_V1.1.tif"
            glc_file = os.path.join(glc_dir, directory, file)
            if not os.path.exists(glc_file):
                print(f"{siteid} did not be extracted because GLC file does not exist.")
                continue
        file_info = cf.get_fileinfo(glc_file)
        row, col = cf.xy2rowcol(file_info['geotrans'], lon, lat)
        row1, col1 = [x - 128 for x in [row, col]]
        row2, col2 = [x + 128 for x in [row, col]]
        try:
            sub_data = file_info['src_ds'].GetRasterBand(23).ReadAsArray(col1, row1, col2 - col1, row2 - row1)
            output_path = os.path.join(output_dir, f"{siteid}.tif")
            cf.save_image(glc_file, output_path, sub_data, col1, row1)
        except AttributeError:
            print(f"{siteid} is out of range in target file.")

def random_generation_points():
    image_size = 256
    glc_dir = "/fossfs/skguan/DATA/7.GLC30_p"
    meta_dirs = glob.iglob("/fossfs/DATAPOOL/Landsat_L2/*/*/2023")
    output_path = "/fossfs/skguan/data_fusion/split_dataset/original_sites"
    cf = CommonFuncs()
    map_dict = {
        # 71: ([71, 72], [80, 100]),  # Evergreen Needleleaf Forests
        # 81: ([81, 82], [75, 100]),  # Deciduous Needleleaf Forests
        # 120: ([120, 121, 122], [50, 100]),  # Shrublands
        # 181: ([181, 182, 184, 185, 186, 187], [30, 100]),  # Permanent Wetlands
        # 190: ([190], [30, 100]),  # Impervious Surfaces
        # 200: ([200, 201, 202], [60, 70]),  # Bare Areas
        # 210: ([210], [30, 100]),  # Water Body
        91: ([91, 92], [30, 100]),  # Mixed Forests
        220: ([220], [30, 100])  # Snow and Ice
    }
    save_dict = {
        'siteid': [],
        'lon': [],
        'lat': [],
        'htile': [],
        'vtile': [],
        'glcid': [],
        'glcprop': [],
        'glcfile': [],
    }
    count = 1
    for i, meta_dir in enumerate(meta_dirs):
        meta_dir = os.path.join(meta_dir, os.listdir(meta_dir)[0])
        meta_file = glob.glob(os.path.join(meta_dir, "*MTL.txt"))[0]
        htile, vtile = meta_file.split("/")[-5:-3]
        with open(meta_file, 'r') as f:
            lines = f.readlines()
            points = {}
            pattern_lonlat = r".*CORNER_(?P<flag1>(?:UL|LR)_(?:LAT|LON))_PRODUCT.?=.?(?P<value1>-?\d+\.?\d+)"
            pattern_widlen = r".*REFLECTIVE_(?P<flag2>(?:LINES|SAMPLES)).?=.?(?P<value2>\d+\.?\d+)"
            for line in lines[60:]:
                regex = re.match(f"{pattern_lonlat}|{pattern_widlen}", line)
                if regex is not None:
                    if regex.group("flag1") is not None:
                        points[regex.group("flag1")] = float(regex.group("value1"))
                        if regex.group("flag1") == "LR_LON":
                            break
                    else:
                        points[regex.group("flag2")] = int(regex.group("value2"))

            lon_zone = 5 * math.floor(int(points['UL_LON']) / 5)
            lat_zone = 5 * math.ceil(int(points['UL_LAT']) / 5)
            EorW = "E" if lon_zone > 0 else "W"
            NorS = "N" if lat_zone > -5 else "S"
            lon_zone0 = lon_zone - 5 if (lon_zone % 10 == 5) & (EorW == "E") else lon_zone
            # lat_zone = abs(lat_zone) + 5 if NorS == "N" else lat_zone
            directory = f"GLC_FCS30D_19852022maps_{EorW}{lon_zone0}-{EorW}{lon_zone0 + 5}"
            file = f"GLC_FCS30D_20002022_{EorW}{lon_zone}{NorS}{abs(lat_zone)}_Annual_V1.1.tif"
            glc_file = os.path.join(glc_dir, directory, file)
            if not os.path.exists(glc_file):
                continue
            glc_extent = cf.get_extent(glc_file)
            if points['UL_LON'] > glc_extent[0] and points['UL_LAT'] < glc_extent[1]:
                file_info = cf.get_fileinfo(glc_file)
                glc_band = file_info['src_ds'].GetRasterBand(23)
                try:
                    width = points['LINES']
                    length = points['SAMPLES']
                    x_res = (points['LR_LON'] - points['UL_LON']) / length
                    y_res = (points['LR_LAT'] - points['UL_LAT']) / width
                except KeyError:
                    continue
                lon_num = int(length / image_size)
                lat_num = int(width / image_size)

                lon_seeds = random.choices(range(0, lon_num), k=200)
                lat_seeds = random.choices(range(0, lat_num), k=200)
                for lon_seed, lat_seed in zip(lon_seeds, lat_seeds):
                    row, col = [i * image_size for i in [lat_seed, lon_seed]]
                    srow, scol = [i - int(image_size / 2) for i in [row, col]]
                    erow, ecol = [i + image_size for i in [srow, scol]]

                    lon1, lon, lon2 = [points['UL_LON'] + x_res * x for x in [scol, col, ecol]]
                    lat1, lat, lat2 = [points['UL_LAT'] + y_res * y for y in [srow, row, erow]]
                    if globe.is_land(lat, lon):
                        row1, col1 = cf.xy2rowcol(file_info['geotrans'], lon1, lat1)
                        row2, col2 = cf.xy2rowcol(file_info['geotrans'], lon2, lat2)
                        if row1 > 0 and col1 > 0 and row2 < file_info['width'] and col2 < file_info['length']:
                            sub_data = glc_band.ReadAsArray(col1, row1, col2 - col1, row2 - row1)
                            for k, (v, p) in map_dict.items():
                                sub_data = np.where(np.isin(sub_data, v), k, sub_data)
                                prop = np.sum(sub_data == k) * 100 / (sub_data.shape[0] * sub_data.shape[1])
                                if p[0] < prop < p[1]:
                                    siteid = "r%d" % (count)
                                    save_dict['siteid'].append(siteid)
                                    save_dict['lon'].append(lon)
                                    save_dict['lat'].append(lat)
                                    save_dict['htile'].append(htile)
                                    save_dict['vtile'].append(vtile)
                                    save_dict['glcid'].append(k)
                                    save_dict['glcprop'].append(prop)
                                    save_dict['glcfile'].append(glc_file)

                                    count += 1

                if count % 500 == 0:
                    print(f"Have found {count} points. Current Progress: {i * 100 / 1893:.2f}%")

    print(f"Have Searched all meta files.")
    df = pd.DataFrame(save_dict)
    df.to_csv(os.path.join(output_path, "random_sites_Snow_and_Ice.csv"))



if __name__ == "__main__":
    search_by_lonlat()


