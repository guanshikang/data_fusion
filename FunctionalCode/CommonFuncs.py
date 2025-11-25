# -*- encoding: utf-8 -*-
"""
@brief: commonly used image processing functions.

@author: guanshikang

@type: class

Created on Wed Jan 08 15:17:06 2025, HONG KONG
"""

import os
import glob
import math
import numpy as np
import netCDF4 as nc
from datetime import datetime
from osgeo import gdal, osr

os.environ['PROJ_LIB'] = '/home/gskk/miniconda3/envs/guanshikang/share/proj'


class CommonFuncs:
    """
    常使用的数据处理函数.
    """
    def __init__(self):
        pass

    # *************** 基础功能函数区 *************** #
    @staticmethod
    def date2doy(date) -> tuple[str, str]:
        """
        convert date to doy.

        Args:
            date (str): YYMMDD.

        returns:
            tuple(str, str): (yyyydoy, doy)
        """
        fmt = '%Y%m%d'
        dt = datetime.strptime(date, fmt)
        tt = dt.timetuple()

        return str(tt.tm_year * 1000 + tt.tm_yday), str(tt.tm_yday)

    @staticmethod
    def doy2date(doy):
        """
        将doy转换成日期.

        Args:
            doy (str): day of year.
        """
        dt = datetime.strptime(doy, '%Y%j').date()
        fmt = '%Y%m%d'

        return dt.strftime(fmt)

    @staticmethod
    def date2num(date: str|datetime|list[datetime]):
        if isinstance(date, str):
            date = datetime.strptime(date, "%Y%m%d")
        calendar = "standard"
        units = "days since 1970-01-01 00:00"
        num = nc.date2num(date, units, calendar)

        return num

    @staticmethod
    def num2date(num: int|list[int], str_format: int=1):
        calendar = "standard"
        units = "days since 1970-01-01 00:00"
        date = nc.num2date(num, units, calendar)
        if str_format:
            date = [datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S") for x in date]
            date = [x.strftime("%Y%m%d") for x in date]

        return date

    def get_fileinfo(self, ref_img):
        """
        获取参考图像的形状，投影，六参数.

        Args:
            ref_img (str): file path of reference image.
        """
        src_ds = gdal.Open(ref_img)
        width = src_ds.RasterYSize
        length = src_ds.RasterXSize
        band_num = src_ds.RasterCount
        geotrans = src_ds.GetGeoTransform()
        pcs = osr.SpatialReference()
        pcs.ImportFromWkt(src_ds.GetProjection())
        gcs = pcs.CloneGeogCS()

        return {
            'src_ds': src_ds,
            'width': width,
            'length': length,
            'band_num': band_num,
            'geotrans': geotrans,
            'pcs': pcs,
            'gcs': gcs
        }

    def get_extent(self, src_path):
        """
        获取图像的左上角和右下角坐标.

        Args:
            src_path (str): reference image file path.

        Returns:
            min_x, max_y, max_x, min_y
        """
        ds = gdal.Open(src_path)
        geo_trans = list(ds.GetGeoTransform())
        x_size = ds.RasterXSize
        y_size = ds.RasterYSize
        min_x = geo_trans[0]
        max_y = geo_trans[3]
        max_x = geo_trans[0] + x_size * geo_trans[1]
        min_y = geo_trans[3] + y_size * geo_trans[5]

        del ds
        return min_x, max_y, max_x, min_y

    def xy2lonlat(self, pcs, gcs, x, y):
        """
        投影坐标转经纬度坐标.
        """
        ct = osr.CoordinateTransformation(pcs, gcs)
        lat, lon, _ = ct.TransformPoint(x, y)  # lat, lon, height
        return lat, lon

    def lonlat2xy(self, pcs, gcs, lon, lat):
        """
        经纬度坐标转投影坐标.
        """
        ct = osr.CoordinateTransformation(gcs, pcs)
        coordinates = ct.TransformPoint(lat, lon)  # UTM needs lat first, lon second.
        return coordinates[0], coordinates[1], coordinates[2]

    def xy2rowcol(self, geotrans, x, y):
        """
        投影或地理坐标转行列号.
        """
        a = np.array([[geotrans[1], geotrans[2]], [geotrans[4], geotrans[5]]])
        b = np.array([x - geotrans[0], y - geotrans[3]])
        row_col = np.linalg.solve(a, b)  # 使用numpy的1ina1g,so1ve进行二元一次方程的求解
        row = int(np.floor(row_col[1]))
        col = int(np.floor(row_col[0]))
        return row, col

    # *************** 基础图像操作区 *************** #
    def resample_image(self, src_path, ref_path, output_path):
        """
        重采样图像.

        Args:
            src_path (str): image path to be resampled.
            ref_path (str): reference image path.
            output_path (str): output image path.
        """
        ref_fileinfo = self.get_fileinfo(ref_path)

        driver = gdal.GetDriverByName("GTiff")
        output_dst = driver.Create(
            output_path,
            ref_fileinfo['length'],
            ref_fileinfo['width'],
            ref_fileinfo['band_num'],
            ref_fileinfo['dtype']
        )
        output_dst.SetGeoTransform(ref_fileinfo['geotrans'])
        output_dst.SetProjection(ref_fileinfo['pcs'].ExportToWkt())

        input_dst = gdal.Open(src_path)
        input_proj = input_dst.GetProjection()
        gdal.ReprojectImage(
            input_dst, output_dst,
            input_proj,
            ref_fileinfo['pcs'].ExportToWkt(),
            gdal.GRA_Cubic
        )

        del input_dst, output_dst

    def reproject_image1(self, src_path, ref_path, output_path):
        """
        重投影图像，可自定义.

        Args:
            src_path (str): input file path.
            ref_path (str): reference file path.
            output_path (str): output file path.
        """
        # 获取输入图像和参考图像的元数据
        src_info = self.get_fileinfo(src_path)
        dtype = src_info['src_ds'].GetRasterBand(1).DataType
        ref_info = self.get_fileinfo(ref_path)

        # 建立两个投影之间的关系
        ct = osr.CoordinateTransformation(src_info['pcs'], ref_info['pcs'])

        # 计算输出图像的四角坐标
        (ulx, uly, _) = ct.TransformPoint(
            src_info['geotrans'][0],
            src_info['geotrans'][3]
        )
        (urx, ury, _) = ct.TransformPoint(
            src_info['geotrans'][0] + src_info['geotrans'][1] * src_info['length'],
            src_info['geotrans'][3]
        )
        (llx, lly, _) = ct.TransformPoint(
            src_info['geotrans'][0],
            src_info['geotrans'][3] + src_info['geotrans'][5] * src_info['width']
        )
        (lrx, lry, _) = ct.TransformPoint(
            src_info['geotrans'][0] +
            src_info['geotrans'][1] * src_info['length'] +
            src_info['geotrans'][2] * src_info['width'],
            src_info['geotrans'][3] +
            src_info['geotrans'][4] * src_info['length'] +
            src_info['geotrans'][5] * src_info['width']
        )

        min_x = min(ulx, urx, llx, lrx)
        max_x = max(ulx, urx, llx, lrx)
        min_y = min(uly, ury, lly, lry)
        max_y = max(uly, ury, lly, lry)

        # 创建输出图像，计算输出尺寸
        # 重投影可根据输入，输出图像分辨率重采样(单位一致)
        # 建议以目标分辨率为目标
        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(
            output_path,
            int((max_x - min_x) / src_info['geotrans'][1]),
            int((max_y - min_y) / -src_info['geotrans'][5]),
            src_info['band_num'],
            dtype
        )
        dst_trans = (min_x, src_info['geotrans'][1], src_info['geotrans'][2],
                     max_y, src_info['geotrans'][4], src_info['geotrans'][5])
        dst_ds.SetGeoTransform(dst_trans)
        dst_ds.SetProjection(ref_info['pcs'].ExportToWkt())

        # 投影转换
        gdal.ReprojectImage(
            src_info['src_ds'],
            dst_ds,
            src_info['pcs'].ExportToWkt(),
            ref_info['pcs'].ExportToWkt(),
            gdal.GRA_Bilinear
        )

        del src_info, ref_info, dst_ds

    def reproject_image2(self, src_path, ref_path, output_path, method=gdal.GRA_Bilinear):
        """
        重投影图像，完全按照参考图像.

        Args:
            src_path (str): input file path.
            ref_path (str): reference file path.
            output_path (str): output file path.
        """
        # 获取输入图像和参考图像的元数据
        src_info = self.get_fileinfo(src_path)
        ref_info = self.get_fileinfo(ref_path)

        geotrans = list(ref_info['geotrans'])
        xmin = geotrans[0]
        ymax = geotrans[3]
        xmax = xmin + geotrans[1] * ref_info['width']
        ymin = ymax + geotrans[5] * ref_info['length']

        warp_options = gdal.WarpOptions(
            srcSRS=src_info['pcs'].ExportToWkt(),
            dstSRS=ref_info['pcs'].ExportToWkt(),
            outputBounds=[xmin, ymin, xmax, ymax],
            xRes=ref_info['geotrans'][1],
            yRes=abs(ref_info['geotrans'][5]),
            targetAlignedPixels=True,
            resampleAlg=method
        )

        # Create the output dataset
        output_ds = gdal.Warp(output_path, src_path, options=warp_options)

        del src_info, ref_info, output_ds

    def mosaic_image(self, src_dir, pattern, output_path, tar_band=1):
        """
        图像镶嵌.

        Args:
            src_dir (str): directory of input data.
            pattern (str): glob pattern to filter files.
            output_path (str): file path of output file.
            tar_band (str): target mosaic band. Default to 1.
        """
        os.chdir(src_dir)
        in_files = glob.glob(pattern)
        in_fn = in_files[0]
        if os.path.exists(output_path):
            os.remove(output_path)
        # 获取待镶嵌栅格的最大最小的坐标值
        min_x, max_y, max_x, min_y = self.get_extent(in_fn)
        for in_fn in in_files[1:]:
            minx, maxy, maxx, miny = self.get_extent(in_fn)
            min_x = min(min_x, minx)
            min_y = min(min_y, miny)
            max_x = max(max_x, maxx)
            max_y = max(max_y, maxy)
        # 计算镶嵌后影像的行列号
        in_ds = gdal.Open(in_files[0])
        geotrans = list(in_ds.GetGeoTransform())
        width = geotrans[1]
        height = geotrans[5]
        dtype = in_ds.GetRasterBand(1).DataType

        columns = math.ceil((max_x - min_x) / width)
        rows = math.ceil((max_y - min_y) / abs(height))

        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(output_path, columns, rows, 1, dtype)
        out_ds.SetProjection(in_ds.GetProjection())
        geotrans[0] = min_x
        geotrans[3] = max_y
        out_ds.SetGeoTransform(geotrans)
        out_band = out_ds.GetRasterBand(1)

        # 定义仿射逆变换
        inv_geotrans = gdal.InvGeoTransform(geotrans)

        # 开始逐渐写入

        for in_fn in in_files:
            print("Processing {}".format(in_fn))
            in_ds = gdal.Open(in_fn)
            in_gt = in_ds.GetGeoTransform()
            # 仿射逆变换
            offset = gdal.ApplyGeoTransform(inv_geotrans, in_gt[0], in_gt[3])
            x, y = map(int, offset)
            trans = gdal.Transformer(in_ds, out_ds, [])  # in_ds是源栅格，out_ds是目标栅格
            _, xyz = trans.TransformPoint(False, 0, 0)  # 计算in_ds中左上角像元对应out_ds中的行列号
            x, y, _ = map(math.ceil, xyz)
            in_da = in_ds.GetRasterBand(tar_band).ReadAsArray()
            out_da = out_band.ReadAsArray(x, y, in_ds.RasterXSize, in_ds.RasterYSize)
            in_da = np.where((in_da == 0) & (out_da != 0), out_da, in_da)
            try:
                out_band.WriteArray(in_da, x, y)
            except Exception as e:
                print(e)
                print("Current x, y are {0}, {1}".format(x, y))
                print("Total rows and cols are {0}, {1}".format(rows, columns))
                print("Total file number is {}".format(len(in_files)))

        del in_ds, out_band, out_ds

    @staticmethod
    def save_image(output_path, data_array, x_off=0, y_off=0, ref_path=None,
                   geo_trans=None, proj=None, meta_data=None, color_table=None,
                   no_data=None):
        """
        Save Full Image.

        Args:
            output_path (str): output image save path.
            data_array (ndarray): data array to be saved.
            x_off (int): longitude offset relative to reference data. If patched iamge, x_off should be seed_num x patch_size.
            y_off (int): latitude offset relative to reference data. If patched image, y_off should be seed_num x patch_size.
            ref_path (str): reference image path that provide coordinate system and geotransform.
            geo_trans (list): specified six parameters.
            proj (str or osr.SpatialReference or EPSG number): specified projection.
            meta_data (): specified meta data.
            color_table (list or gdal.ColorTable): specified color table.
            no_data (number): ignore data. int or float.

        Usage of color table:
            A color sequence should be specified first.
            Then instantiate a object of gdal.ColorTable().
            Set color entry for each unique value.
            For example:
                palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
                ct = gdal.ColorTable()
                for i, color in enumerate(palette):
                    ct.SetColorEntry(i, color)  # the colored unique value.

            This operation should be perfomed before write array to file.
        """
        driver = gdal.GetDriverByName("GTiff")
        DTYPE = {
            'uint8': gdal.GDT_Byte, 'int8': gdal.GDT_Byte,
            'uint16': gdal.GDT_UInt16, 'int16': gdal.GDT_Int16,
            'uint32': gdal.GDT_UInt32, 'int32': gdal.GDT_Int32,
            'uint64': gdal.GDT_Int64, 'int64': gdal.GDT_Int64,
            'float32': gdal.GDT_Float32, 'float64': gdal.GDT_Float64
        }
        dtype = data_array.dtype.name
        if dtype in DTYPE:
            dtype = DTYPE[dtype]
        else:
            dtype = gdal.GDT_Float32

        if len(data_array.shape) == 3:
            band_num, width, length = data_array.shape
        elif len(data_array.shape) == 2:
            data_array = np.array([data_array], dtype=dtype)
            band_num, width, length = data_array.shape
        else:
            raise ValueError("GDAL only supports 2 and 3 dimensional data.")
        out_ds = driver.Create(output_path,
                               length,
                               width,
                               band_num,
                               dtype)

        if ref_path is not None:
            file_info = CommonFuncs.get_fileinfo(ref_path)
            proj = file_info['pcs'].ExportToWkt()
            if geo_trans is None:
                geo_trans = list(file_info['geotrans'])
                geo_trans[0] += geo_trans[1] * x_off
                geo_trans[3] += geo_trans[5] * y_off
            out_ds.SetGeoTransform(geo_trans)
        if proj is not None:
            if isinstance(proj, osr.SpatialReference):
                proj = proj.ExportToWkt()
            elif isinstance(proj, int):
                srs = osr.SpatialReference()
                srs.ImportFromEPSG(proj)
                proj = srs.ExportToWkt()
            elif isinstance(proj, str):
                out_ds.SetProjection(proj)
            else:
                raise ValueError("Format of projection is wrong.")
        if meta_data is not None:
            out_ds.SetMetaData(meta_data)
        if color_table is not None:
            out_ds.GetRasterBand(1).SetColorTable(color_table)
        for i in range(band_num):
            out_ds.GetRasterBand(i + 1).WriteArray(data_array[i])
        if no_data is not None:
            for i in range(band_num):
                out_ds.GetRasterBand(i + 1).WriteArray(data_array[i])

        del out_ds

    def save_nc(self, output_path, data, lon, lat, time, bandnum, mask=None,
                  flag=None):
        """
        保存时间序列图像为nc文件.

        Args:
            output_path (str): the output file path.
            data (list): data list open with map.
            lon (ndarray): coordinate for longitude.
            lat (ndarray): coordinate for latitude.
            time (list): list of datetime.
            bandnum (int): band num.
            mask (int): mask index of data. default to None.
        """
        Landsat = None
        dst = nc.Dataset(output_path, "w", format='NETCDF4')
        dst.createDimension("lon", len(lon))
        dst.createDimension("lat", len(lat))
        if bandnum > 1:
            bandnum = bandnum - 1 if mask is not None else bandnum
            dst.createDimension("band", bandnum)
        dst.createDimension("time", len(time))

        nc_lon = dst.createVariable("lon", "f", "lon")
        nc_lat = dst.createVariable("lat", "f", "lat")
        nc_time = dst.createVariable("time", "f8", "time")
        if bandnum > 1:
            dims = ("time", "band", "lat", "lon")
        else:
            dims = ("time", "lat", "lon")
        DTYPE = {
            'uint8': "u1", 'uint16': "u2",
            'int8': "i1", 'int16': "i2",
            'float32': "f4", 'float64': "f8"
        }
        if data[0].dtype.name in DTYPE.keys():
            dtype = DTYPE[data[0].dtype.name]
        else:
            dtype = 'f'
        nc_data = dst.createVariable("data", dtype, dims)
        if mask is not None:
            Landsat = True
            if data[mask].dtype.name in DTYPE.keys():
                dtype = DTYPE[data[mask].dtype.name]
            else:
                dtype = "u2"
            nc_mask = dst.createVariable("mask", dtype, ("time", "lat", "lon"))

        nc_lon[:] = lon
        nc_lat[:] = lat
        calendar = "standard"
        units = "days since 1970-01-01 00:00"
        nc_time[:] = nc.date2num(time, units, calendar)
        if flag is not None:
            nc_flag = dst.createVariable("sat", "S1", "time")
            nc_flag[:] = flag
        if Landsat:
            if bandnum > 1:
                for i in range(len(time)):
                    if mask is not None:
                        nc_mask[i, :, :] = data[i][mask, :, :].squeeze()
                        nc_data[i, :, :, :] = data[i][:mask, :, :]
                    else:
                        nc_data[i, :, :, :] = data[i]
            else:
                for i in range(len(time)):
                    nc_data[i, :, :] = data[i]
        else:
            if bandnum > 1:
                for i in range(len(time)):
                    nc_data[i, :, :, :] = data[i]
            else:
                for i in range(len(time)):
                    nc_data[i, :, :] = data[i]

        dst.close()
