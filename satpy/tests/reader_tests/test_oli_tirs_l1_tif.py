#!/usr/bin/python
# Copyright (c) 2018 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Unittests for generic image reader."""

import os

import dask.array as da
import numpy as np
import pytest
import xarray as xr

metadata_text = b"""<?xml version="1.0" encoding="UTF-8"?>
<LANDSAT_METADATA_FILE>
  <PRODUCT_CONTENTS>
    <ORIGIN>Image courtesy of the U.S. Geological Survey</ORIGIN>
    <DIGITAL_OBJECT_IDENTIFIER>https://doi.org/10.5066/P975CC9B</DIGITAL_OBJECT_IDENTIFIER>
    <LANDSAT_PRODUCT_ID>LC08_L1GT_026200_20240502_20240513_02_T2</LANDSAT_PRODUCT_ID>
    <PROCESSING_LEVEL>L1GT</PROCESSING_LEVEL>
    <COLLECTION_NUMBER>02</COLLECTION_NUMBER>
    <COLLECTION_CATEGORY>T2</COLLECTION_CATEGORY>
    <OUTPUT_FORMAT>GEOTIFF</OUTPUT_FORMAT>
    <FILE_NAME_BAND_1>LC08_L1GT_026200_20240502_20240513_02_T2_B1.TIF</FILE_NAME_BAND_1>
    <FILE_NAME_BAND_2>LC08_L1GT_026200_20240502_20240513_02_T2_B2.TIF</FILE_NAME_BAND_2>
    <FILE_NAME_BAND_3>LC08_L1GT_026200_20240502_20240513_02_T2_B3.TIF</FILE_NAME_BAND_3>
    <FILE_NAME_BAND_4>LC08_L1GT_026200_20240502_20240513_02_T2_B4.TIF</FILE_NAME_BAND_4>
    <FILE_NAME_BAND_5>LC08_L1GT_026200_20240502_20240513_02_T2_B5.TIF</FILE_NAME_BAND_5>
    <FILE_NAME_BAND_6>LC08_L1GT_026200_20240502_20240513_02_T2_B6.TIF</FILE_NAME_BAND_6>
    <FILE_NAME_BAND_7>LC08_L1GT_026200_20240502_20240513_02_T2_B7.TIF</FILE_NAME_BAND_7>
    <FILE_NAME_BAND_8>LC08_L1GT_026200_20240502_20240513_02_T2_B8.TIF</FILE_NAME_BAND_8>
    <FILE_NAME_BAND_9>LC08_L1GT_026200_20240502_20240513_02_T2_B9.TIF</FILE_NAME_BAND_9>
    <FILE_NAME_BAND_10>LC08_L1GT_026200_20240502_20240513_02_T2_B10.TIF</FILE_NAME_BAND_10>
    <FILE_NAME_BAND_11>LC08_L1GT_026200_20240502_20240513_02_T2_B11.TIF</FILE_NAME_BAND_11>
    <FILE_NAME_QUALITY_L1_PIXEL>LC08_L1GT_026200_20240502_20240513_02_T2_QA_PIXEL.TIF</FILE_NAME_QUALITY_L1_PIXEL>
    <FILE_NAME_QUALITY_L1_RADIOMETRIC_SATURATION>LC08_L1GT_026200_20240502_20240513_02_T2_QA_RADSAT.TIF</FILE_NAME_QUALITY_L1_RADIOMETRIC_SATURATION>
    <FILE_NAME_ANGLE_COEFFICIENT>LC08_L1GT_026200_20240502_20240513_02_T2_ANG.txt</FILE_NAME_ANGLE_COEFFICIENT>
    <FILE_NAME_ANGLE_SENSOR_AZIMUTH_BAND_4>LC08_L1GT_026200_20240502_20240513_02_T2_VAA.TIF</FILE_NAME_ANGLE_SENSOR_AZIMUTH_BAND_4>
    <FILE_NAME_ANGLE_SENSOR_ZENITH_BAND_4>LC08_L1GT_026200_20240502_20240513_02_T2_VZA.TIF</FILE_NAME_ANGLE_SENSOR_ZENITH_BAND_4>
    <FILE_NAME_ANGLE_SOLAR_AZIMUTH_BAND_4>LC08_L1GT_026200_20240502_20240513_02_T2_SAA.TIF</FILE_NAME_ANGLE_SOLAR_AZIMUTH_BAND_4>
    <FILE_NAME_ANGLE_SOLAR_ZENITH_BAND_4>LC08_L1GT_026200_20240502_20240513_02_T2_SZA.TIF</FILE_NAME_ANGLE_SOLAR_ZENITH_BAND_4>
    <FILE_NAME_METADATA_ODL>LC08_L1GT_026200_20240502_20240513_02_T2_MTL.txt</FILE_NAME_METADATA_ODL>
    <FILE_NAME_METADATA_XML>LC08_L1GT_026200_20240502_20240513_02_T2_MTL.xml</FILE_NAME_METADATA_XML>
    <DATA_TYPE_BAND_1>UINT16</DATA_TYPE_BAND_1>
    <DATA_TYPE_BAND_2>UINT16</DATA_TYPE_BAND_2>
    <DATA_TYPE_BAND_3>UINT16</DATA_TYPE_BAND_3>
    <DATA_TYPE_BAND_4>UINT16</DATA_TYPE_BAND_4>
    <DATA_TYPE_BAND_5>UINT16</DATA_TYPE_BAND_5>
    <DATA_TYPE_BAND_6>UINT16</DATA_TYPE_BAND_6>
    <DATA_TYPE_BAND_7>UINT16</DATA_TYPE_BAND_7>
    <DATA_TYPE_BAND_8>UINT16</DATA_TYPE_BAND_8>
    <DATA_TYPE_BAND_9>UINT16</DATA_TYPE_BAND_9>
    <DATA_TYPE_BAND_10>UINT16</DATA_TYPE_BAND_10>
    <DATA_TYPE_BAND_11>UINT16</DATA_TYPE_BAND_11>
    <DATA_TYPE_QUALITY_L1_PIXEL>UINT16</DATA_TYPE_QUALITY_L1_PIXEL>
    <DATA_TYPE_QUALITY_L1_RADIOMETRIC_SATURATION>UINT16</DATA_TYPE_QUALITY_L1_RADIOMETRIC_SATURATION>
    <DATA_TYPE_ANGLE_SENSOR_AZIMUTH_BAND_4>INT16</DATA_TYPE_ANGLE_SENSOR_AZIMUTH_BAND_4>
    <DATA_TYPE_ANGLE_SENSOR_ZENITH_BAND_4>INT16</DATA_TYPE_ANGLE_SENSOR_ZENITH_BAND_4>
    <DATA_TYPE_ANGLE_SOLAR_AZIMUTH_BAND_4>INT16</DATA_TYPE_ANGLE_SOLAR_AZIMUTH_BAND_4>
    <DATA_TYPE_ANGLE_SOLAR_ZENITH_BAND_4>INT16</DATA_TYPE_ANGLE_SOLAR_ZENITH_BAND_4>
  </PRODUCT_CONTENTS>
  <IMAGE_ATTRIBUTES>
    <SPACECRAFT_ID>LANDSAT_8</SPACECRAFT_ID>
    <SENSOR_ID>OLI_TIRS</SENSOR_ID>
    <WRS_TYPE>2</WRS_TYPE>
    <WRS_PATH>26</WRS_PATH>
    <WRS_ROW>200</WRS_ROW>
    <NADIR_OFFNADIR>NADIR</NADIR_OFFNADIR>
    <TARGET_WRS_PATH>26</TARGET_WRS_PATH>
    <TARGET_WRS_ROW>200</TARGET_WRS_ROW>
    <DATE_ACQUIRED>2024-05-02</DATE_ACQUIRED>
    <SCENE_CENTER_TIME>18:00:24.6148649Z</SCENE_CENTER_TIME>
    <STATION_ID>LGN</STATION_ID>
    <CLOUD_COVER>0.85</CLOUD_COVER>
    <CLOUD_COVER_LAND>-1</CLOUD_COVER_LAND>
    <IMAGE_QUALITY_OLI>9</IMAGE_QUALITY_OLI>
    <IMAGE_QUALITY_TIRS>9</IMAGE_QUALITY_TIRS>
    <SATURATION_BAND_1>N</SATURATION_BAND_1>
    <SATURATION_BAND_2>N</SATURATION_BAND_2>
    <SATURATION_BAND_3>N</SATURATION_BAND_3>
    <SATURATION_BAND_4>Y</SATURATION_BAND_4>
    <SATURATION_BAND_5>N</SATURATION_BAND_5>
    <SATURATION_BAND_6>N</SATURATION_BAND_6>
    <SATURATION_BAND_7>N</SATURATION_BAND_7>
    <SATURATION_BAND_8>N</SATURATION_BAND_8>
    <SATURATION_BAND_9>N</SATURATION_BAND_9>
    <ROLL_ANGLE>-0.000</ROLL_ANGLE>
    <SUN_AZIMUTH>-39.71362413</SUN_AZIMUTH>
    <SUN_ELEVATION>-41.46228969</SUN_ELEVATION>
    <EARTH_SUN_DISTANCE>1.0079981</EARTH_SUN_DISTANCE>
    <TRUNCATION_OLI>UPPER</TRUNCATION_OLI>
    <TIRS_SSM_MODEL>FINAL</TIRS_SSM_MODEL>
    <TIRS_SSM_POSITION_STATUS>ESTIMATED</TIRS_SSM_POSITION_STATUS>
  </IMAGE_ATTRIBUTES>
  <PROJECTION_ATTRIBUTES>
    <MAP_PROJECTION>UTM</MAP_PROJECTION>
    <DATUM>WGS84</DATUM>
    <ELLIPSOID>WGS84</ELLIPSOID>
    <UTM_ZONE>40</UTM_ZONE>
    <GRID_CELL_SIZE_PANCHROMATIC>15.00</GRID_CELL_SIZE_PANCHROMATIC>
    <GRID_CELL_SIZE_REFLECTIVE>30.00</GRID_CELL_SIZE_REFLECTIVE>
    <GRID_CELL_SIZE_THERMAL>30.00</GRID_CELL_SIZE_THERMAL>
    <PANCHROMATIC_LINES>200</PANCHROMATIC_LINES>
    <PANCHROMATIC_SAMPLES>200</PANCHROMATIC_SAMPLES>
    <REFLECTIVE_LINES>100</REFLECTIVE_LINES>
    <REFLECTIVE_SAMPLES>100</REFLECTIVE_SAMPLES>
    <THERMAL_LINES>100</THERMAL_LINES>
    <THERMAL_SAMPLES>100</THERMAL_SAMPLES>
    <ORIENTATION>NORTH_UP</ORIENTATION>
    <CORNER_UL_LAT_PRODUCT>24.18941</CORNER_UL_LAT_PRODUCT>
    <CORNER_UL_LON_PRODUCT>58.17657</CORNER_UL_LON_PRODUCT>
    <CORNER_UR_LAT_PRODUCT>24.15493</CORNER_UR_LAT_PRODUCT>
    <CORNER_UR_LON_PRODUCT>60.44878</CORNER_UR_LON_PRODUCT>
    <CORNER_LL_LAT_PRODUCT>22.06522</CORNER_LL_LAT_PRODUCT>
    <CORNER_LL_LON_PRODUCT>58.15819</CORNER_LL_LON_PRODUCT>
    <CORNER_LR_LAT_PRODUCT>22.03410</CORNER_LR_LAT_PRODUCT>
    <CORNER_LR_LON_PRODUCT>60.39501</CORNER_LR_LON_PRODUCT>
    <CORNER_UL_PROJECTION_X_PRODUCT>619500.000</CORNER_UL_PROJECTION_X_PRODUCT>
    <CORNER_UL_PROJECTION_Y_PRODUCT>2675700.000</CORNER_UL_PROJECTION_Y_PRODUCT>
    <CORNER_UR_PROJECTION_X_PRODUCT>850500.000</CORNER_UR_PROJECTION_X_PRODUCT>
    <CORNER_UR_PROJECTION_Y_PRODUCT>2675700.000</CORNER_UR_PROJECTION_Y_PRODUCT>
    <CORNER_LL_PROJECTION_X_PRODUCT>619500.000</CORNER_LL_PROJECTION_X_PRODUCT>
    <CORNER_LL_PROJECTION_Y_PRODUCT>2440500.000</CORNER_LL_PROJECTION_Y_PRODUCT>
    <CORNER_LR_PROJECTION_X_PRODUCT>850500.000</CORNER_LR_PROJECTION_X_PRODUCT>
    <CORNER_LR_PROJECTION_Y_PRODUCT>2440500.000</CORNER_LR_PROJECTION_Y_PRODUCT>
  </PROJECTION_ATTRIBUTES>
  <LEVEL1_PROCESSING_RECORD>
    <ORIGIN>Image courtesy of the U.S. Geological Survey</ORIGIN>
    <DIGITAL_OBJECT_IDENTIFIER>https://doi.org/10.5066/P975CC9B</DIGITAL_OBJECT_IDENTIFIER>
    <REQUEST_ID>1885324_00001</REQUEST_ID>
    <LANDSAT_SCENE_ID>LC80262002024123LGN00</LANDSAT_SCENE_ID>
    <LANDSAT_PRODUCT_ID>LC08_L1GT_026200_20240502_20240513_02_T2</LANDSAT_PRODUCT_ID>
    <PROCESSING_LEVEL>L1GT</PROCESSING_LEVEL>
    <COLLECTION_CATEGORY>T2</COLLECTION_CATEGORY>
    <OUTPUT_FORMAT>GEOTIFF</OUTPUT_FORMAT>
    <DATE_PRODUCT_GENERATED>2024-05-13T15:32:54Z</DATE_PRODUCT_GENERATED>
    <PROCESSING_SOFTWARE_VERSION>LPGS_16.4.0</PROCESSING_SOFTWARE_VERSION>
    <FILE_NAME_BAND_1>LC08_L1GT_026200_20240502_20240513_02_T2_B1.TIF</FILE_NAME_BAND_1>
    <FILE_NAME_BAND_2>LC08_L1GT_026200_20240502_20240513_02_T2_B2.TIF</FILE_NAME_BAND_2>
    <FILE_NAME_BAND_3>LC08_L1GT_026200_20240502_20240513_02_T2_B3.TIF</FILE_NAME_BAND_3>
    <FILE_NAME_BAND_4>LC08_L1GT_026200_20240502_20240513_02_T2_B4.TIF</FILE_NAME_BAND_4>
    <FILE_NAME_BAND_5>LC08_L1GT_026200_20240502_20240513_02_T2_B5.TIF</FILE_NAME_BAND_5>
    <FILE_NAME_BAND_6>LC08_L1GT_026200_20240502_20240513_02_T2_B6.TIF</FILE_NAME_BAND_6>
    <FILE_NAME_BAND_7>LC08_L1GT_026200_20240502_20240513_02_T2_B7.TIF</FILE_NAME_BAND_7>
    <FILE_NAME_BAND_8>LC08_L1GT_026200_20240502_20240513_02_T2_B8.TIF</FILE_NAME_BAND_8>
    <FILE_NAME_BAND_9>LC08_L1GT_026200_20240502_20240513_02_T2_B9.TIF</FILE_NAME_BAND_9>
    <FILE_NAME_BAND_10>LC08_L1GT_026200_20240502_20240513_02_T2_B10.TIF</FILE_NAME_BAND_10>
    <FILE_NAME_BAND_11>LC08_L1GT_026200_20240502_20240513_02_T2_B11.TIF</FILE_NAME_BAND_11>
    <FILE_NAME_QUALITY_L1_PIXEL>LC08_L1GT_026200_20240502_20240513_02_T2_QA_PIXEL.TIF</FILE_NAME_QUALITY_L1_PIXEL>
    <FILE_NAME_QUALITY_L1_RADIOMETRIC_SATURATION>LC08_L1GT_026200_20240502_20240513_02_T2_QA_RADSAT.TIF</FILE_NAME_QUALITY_L1_RADIOMETRIC_SATURATION>
    <FILE_NAME_ANGLE_COEFFICIENT>LC08_L1GT_026200_20240502_20240513_02_T2_ANG.txt</FILE_NAME_ANGLE_COEFFICIENT>
    <FILE_NAME_ANGLE_SENSOR_AZIMUTH_BAND_4>LC08_L1GT_026200_20240502_20240513_02_T2_VAA.TIF</FILE_NAME_ANGLE_SENSOR_AZIMUTH_BAND_4>
    <FILE_NAME_ANGLE_SENSOR_ZENITH_BAND_4>LC08_L1GT_026200_20240502_20240513_02_T2_VZA.TIF</FILE_NAME_ANGLE_SENSOR_ZENITH_BAND_4>
    <FILE_NAME_ANGLE_SOLAR_AZIMUTH_BAND_4>LC08_L1GT_026200_20240502_20240513_02_T2_SAA.TIF</FILE_NAME_ANGLE_SOLAR_AZIMUTH_BAND_4>
    <FILE_NAME_ANGLE_SOLAR_ZENITH_BAND_4>LC08_L1GT_026200_20240502_20240513_02_T2_SZA.TIF</FILE_NAME_ANGLE_SOLAR_ZENITH_BAND_4>
    <FILE_NAME_METADATA_ODL>LC08_L1GT_026200_20240502_20240513_02_T2_MTL.txt</FILE_NAME_METADATA_ODL>
    <FILE_NAME_METADATA_XML>LC08_L1GT_026200_20240502_20240513_02_T2_MTL.xml</FILE_NAME_METADATA_XML>
    <FILE_NAME_CPF>LC08CPF_20240429_20240630_02.03</FILE_NAME_CPF>
    <FILE_NAME_BPF_OLI>LO8BPF20240502162846_20240502181430.01</FILE_NAME_BPF_OLI>
    <FILE_NAME_BPF_TIRS>LT8BPF20240502144307_20240510102926.01</FILE_NAME_BPF_TIRS>
    <FILE_NAME_RLUT>LC08RLUT_20150303_20431231_02_01.h5</FILE_NAME_RLUT>
    <DATA_SOURCE_TIRS_STRAY_LIGHT_CORRECTION>TIRS</DATA_SOURCE_TIRS_STRAY_LIGHT_CORRECTION>
    <DATA_SOURCE_ELEVATION>GLS2000</DATA_SOURCE_ELEVATION>
  </LEVEL1_PROCESSING_RECORD>
  <LEVEL1_MIN_MAX_RADIANCE>
    <RADIANCE_MAXIMUM_BAND_1>748.04883</RADIANCE_MAXIMUM_BAND_1>
    <RADIANCE_MINIMUM_BAND_1>-61.77412</RADIANCE_MINIMUM_BAND_1>
    <RADIANCE_MAXIMUM_BAND_2>766.01111</RADIANCE_MAXIMUM_BAND_2>
    <RADIANCE_MINIMUM_BAND_2>-63.25745</RADIANCE_MINIMUM_BAND_2>
    <RADIANCE_MAXIMUM_BAND_3>705.87274</RADIANCE_MAXIMUM_BAND_3>
    <RADIANCE_MINIMUM_BAND_3>-58.29120</RADIANCE_MINIMUM_BAND_3>
    <RADIANCE_MAXIMUM_BAND_4>595.23163</RADIANCE_MAXIMUM_BAND_4>
    <RADIANCE_MINIMUM_BAND_4>-49.15442</RADIANCE_MINIMUM_BAND_4>
    <RADIANCE_MAXIMUM_BAND_5>364.25208</RADIANCE_MAXIMUM_BAND_5>
    <RADIANCE_MINIMUM_BAND_5>-30.08006</RADIANCE_MINIMUM_BAND_5>
    <RADIANCE_MAXIMUM_BAND_6>90.58618</RADIANCE_MAXIMUM_BAND_6>
    <RADIANCE_MINIMUM_BAND_6>-7.48064</RADIANCE_MINIMUM_BAND_6>
    <RADIANCE_MAXIMUM_BAND_7>30.53239</RADIANCE_MAXIMUM_BAND_7>
    <RADIANCE_MINIMUM_BAND_7>-2.52137</RADIANCE_MINIMUM_BAND_7>
    <RADIANCE_MAXIMUM_BAND_8>673.63843</RADIANCE_MAXIMUM_BAND_8>
    <RADIANCE_MINIMUM_BAND_8>-55.62928</RADIANCE_MINIMUM_BAND_8>
    <RADIANCE_MAXIMUM_BAND_9>142.35797</RADIANCE_MAXIMUM_BAND_9>
    <RADIANCE_MINIMUM_BAND_9>-11.75597</RADIANCE_MINIMUM_BAND_9>
    <RADIANCE_MAXIMUM_BAND_10>22.00180</RADIANCE_MAXIMUM_BAND_10>
    <RADIANCE_MINIMUM_BAND_10>0.10033</RADIANCE_MINIMUM_BAND_10>
    <RADIANCE_MAXIMUM_BAND_11>22.00180</RADIANCE_MAXIMUM_BAND_11>
    <RADIANCE_MINIMUM_BAND_11>0.10033</RADIANCE_MINIMUM_BAND_11>
  </LEVEL1_MIN_MAX_RADIANCE>
  <LEVEL1_MIN_MAX_REFLECTANCE>
    <REFLECTANCE_MAXIMUM_BAND_1>1.210700</REFLECTANCE_MAXIMUM_BAND_1>
    <REFLECTANCE_MINIMUM_BAND_1>-0.099980</REFLECTANCE_MINIMUM_BAND_1>
    <REFLECTANCE_MAXIMUM_BAND_2>1.210700</REFLECTANCE_MAXIMUM_BAND_2>
    <REFLECTANCE_MINIMUM_BAND_2>-0.099980</REFLECTANCE_MINIMUM_BAND_2>
    <REFLECTANCE_MAXIMUM_BAND_3>1.210700</REFLECTANCE_MAXIMUM_BAND_3>
    <REFLECTANCE_MINIMUM_BAND_3>-0.099980</REFLECTANCE_MINIMUM_BAND_3>
    <REFLECTANCE_MAXIMUM_BAND_4>1.210700</REFLECTANCE_MAXIMUM_BAND_4>
    <REFLECTANCE_MINIMUM_BAND_4>-0.099980</REFLECTANCE_MINIMUM_BAND_4>
    <REFLECTANCE_MAXIMUM_BAND_5>1.210700</REFLECTANCE_MAXIMUM_BAND_5>
    <REFLECTANCE_MINIMUM_BAND_5>-0.099980</REFLECTANCE_MINIMUM_BAND_5>
    <REFLECTANCE_MAXIMUM_BAND_6>1.210700</REFLECTANCE_MAXIMUM_BAND_6>
    <REFLECTANCE_MINIMUM_BAND_6>-0.099980</REFLECTANCE_MINIMUM_BAND_6>
    <REFLECTANCE_MAXIMUM_BAND_7>1.210700</REFLECTANCE_MAXIMUM_BAND_7>
    <REFLECTANCE_MINIMUM_BAND_7>-0.099980</REFLECTANCE_MINIMUM_BAND_7>
    <REFLECTANCE_MAXIMUM_BAND_8>1.210700</REFLECTANCE_MAXIMUM_BAND_8>
    <REFLECTANCE_MINIMUM_BAND_8>-0.099980</REFLECTANCE_MINIMUM_BAND_8>
    <REFLECTANCE_MAXIMUM_BAND_9>1.210700</REFLECTANCE_MAXIMUM_BAND_9>
    <REFLECTANCE_MINIMUM_BAND_9>-0.099980</REFLECTANCE_MINIMUM_BAND_9>
  </LEVEL1_MIN_MAX_REFLECTANCE>
  <LEVEL1_MIN_MAX_PIXEL_VALUE>
    <QUANTIZE_CAL_MAX_BAND_1>65535</QUANTIZE_CAL_MAX_BAND_1>
    <QUANTIZE_CAL_MIN_BAND_1>1</QUANTIZE_CAL_MIN_BAND_1>
    <QUANTIZE_CAL_MAX_BAND_2>65535</QUANTIZE_CAL_MAX_BAND_2>
    <QUANTIZE_CAL_MIN_BAND_2>1</QUANTIZE_CAL_MIN_BAND_2>
    <QUANTIZE_CAL_MAX_BAND_3>65535</QUANTIZE_CAL_MAX_BAND_3>
    <QUANTIZE_CAL_MIN_BAND_3>1</QUANTIZE_CAL_MIN_BAND_3>
    <QUANTIZE_CAL_MAX_BAND_4>65535</QUANTIZE_CAL_MAX_BAND_4>
    <QUANTIZE_CAL_MIN_BAND_4>1</QUANTIZE_CAL_MIN_BAND_4>
    <QUANTIZE_CAL_MAX_BAND_5>65535</QUANTIZE_CAL_MAX_BAND_5>
    <QUANTIZE_CAL_MIN_BAND_5>1</QUANTIZE_CAL_MIN_BAND_5>
    <QUANTIZE_CAL_MAX_BAND_6>65535</QUANTIZE_CAL_MAX_BAND_6>
    <QUANTIZE_CAL_MIN_BAND_6>1</QUANTIZE_CAL_MIN_BAND_6>
    <QUANTIZE_CAL_MAX_BAND_7>65535</QUANTIZE_CAL_MAX_BAND_7>
    <QUANTIZE_CAL_MIN_BAND_7>1</QUANTIZE_CAL_MIN_BAND_7>
    <QUANTIZE_CAL_MAX_BAND_8>65535</QUANTIZE_CAL_MAX_BAND_8>
    <QUANTIZE_CAL_MIN_BAND_8>1</QUANTIZE_CAL_MIN_BAND_8>
    <QUANTIZE_CAL_MAX_BAND_9>65535</QUANTIZE_CAL_MAX_BAND_9>
    <QUANTIZE_CAL_MIN_BAND_9>1</QUANTIZE_CAL_MIN_BAND_9>
    <QUANTIZE_CAL_MAX_BAND_10>65535</QUANTIZE_CAL_MAX_BAND_10>
    <QUANTIZE_CAL_MIN_BAND_10>1</QUANTIZE_CAL_MIN_BAND_10>
    <QUANTIZE_CAL_MAX_BAND_11>65535</QUANTIZE_CAL_MAX_BAND_11>
    <QUANTIZE_CAL_MIN_BAND_11>1</QUANTIZE_CAL_MIN_BAND_11>
  </LEVEL1_MIN_MAX_PIXEL_VALUE>
  <LEVEL1_RADIOMETRIC_RESCALING>
    <RADIANCE_MULT_BAND_1>1.2357E-02</RADIANCE_MULT_BAND_1>
    <RADIANCE_MULT_BAND_2>1.2654E-02</RADIANCE_MULT_BAND_2>
    <RADIANCE_MULT_BAND_3>1.1661E-02</RADIANCE_MULT_BAND_3>
    <RADIANCE_MULT_BAND_4>9.8329E-03</RADIANCE_MULT_BAND_4>
    <RADIANCE_MULT_BAND_5>6.0172E-03</RADIANCE_MULT_BAND_5>
    <RADIANCE_MULT_BAND_6>1.4964E-03</RADIANCE_MULT_BAND_6>
    <RADIANCE_MULT_BAND_7>5.0438E-04</RADIANCE_MULT_BAND_7>
    <RADIANCE_MULT_BAND_8>1.1128E-02</RADIANCE_MULT_BAND_8>
    <RADIANCE_MULT_BAND_9>2.3517E-03</RADIANCE_MULT_BAND_9>
    <RADIANCE_MULT_BAND_10>3.3420E-04</RADIANCE_MULT_BAND_10>
    <RADIANCE_MULT_BAND_11>3.3420E-04</RADIANCE_MULT_BAND_11>
    <RADIANCE_ADD_BAND_1>-61.78647</RADIANCE_ADD_BAND_1>
    <RADIANCE_ADD_BAND_2>-63.27010</RADIANCE_ADD_BAND_2>
    <RADIANCE_ADD_BAND_3>-58.30286</RADIANCE_ADD_BAND_3>
    <RADIANCE_ADD_BAND_4>-49.16426</RADIANCE_ADD_BAND_4>
    <RADIANCE_ADD_BAND_5>-30.08607</RADIANCE_ADD_BAND_5>
    <RADIANCE_ADD_BAND_6>-7.48213</RADIANCE_ADD_BAND_6>
    <RADIANCE_ADD_BAND_7>-2.52188</RADIANCE_ADD_BAND_7>
    <RADIANCE_ADD_BAND_8>-55.64041</RADIANCE_ADD_BAND_8>
    <RADIANCE_ADD_BAND_9>-11.75832</RADIANCE_ADD_BAND_9>
    <RADIANCE_ADD_BAND_10>0.10000</RADIANCE_ADD_BAND_10>
    <RADIANCE_ADD_BAND_11>0.10000</RADIANCE_ADD_BAND_11>
    <REFLECTANCE_MULT_BAND_1>2.0000E-05</REFLECTANCE_MULT_BAND_1>
    <REFLECTANCE_MULT_BAND_2>2.0000E-05</REFLECTANCE_MULT_BAND_2>
    <REFLECTANCE_MULT_BAND_3>2.0000E-05</REFLECTANCE_MULT_BAND_3>
    <REFLECTANCE_MULT_BAND_4>2.0000E-05</REFLECTANCE_MULT_BAND_4>
    <REFLECTANCE_MULT_BAND_5>2.0000E-05</REFLECTANCE_MULT_BAND_5>
    <REFLECTANCE_MULT_BAND_6>2.0000E-05</REFLECTANCE_MULT_BAND_6>
    <REFLECTANCE_MULT_BAND_7>2.0000E-05</REFLECTANCE_MULT_BAND_7>
    <REFLECTANCE_MULT_BAND_8>2.0000E-05</REFLECTANCE_MULT_BAND_8>
    <REFLECTANCE_MULT_BAND_9>2.0000E-05</REFLECTANCE_MULT_BAND_9>
    <REFLECTANCE_ADD_BAND_1>-0.100000</REFLECTANCE_ADD_BAND_1>
    <REFLECTANCE_ADD_BAND_2>-0.100000</REFLECTANCE_ADD_BAND_2>
    <REFLECTANCE_ADD_BAND_3>-0.100000</REFLECTANCE_ADD_BAND_3>
    <REFLECTANCE_ADD_BAND_4>-0.100000</REFLECTANCE_ADD_BAND_4>
    <REFLECTANCE_ADD_BAND_5>-0.100000</REFLECTANCE_ADD_BAND_5>
    <REFLECTANCE_ADD_BAND_6>-0.100000</REFLECTANCE_ADD_BAND_6>
    <REFLECTANCE_ADD_BAND_7>-0.100000</REFLECTANCE_ADD_BAND_7>
    <REFLECTANCE_ADD_BAND_8>-0.100000</REFLECTANCE_ADD_BAND_8>
    <REFLECTANCE_ADD_BAND_9>-0.100000</REFLECTANCE_ADD_BAND_9>
  </LEVEL1_RADIOMETRIC_RESCALING>
  <LEVEL1_THERMAL_CONSTANTS>
    <K1_CONSTANT_BAND_10>774.8853</K1_CONSTANT_BAND_10>
    <K2_CONSTANT_BAND_10>1321.0789</K2_CONSTANT_BAND_10>
    <K1_CONSTANT_BAND_11>480.8883</K1_CONSTANT_BAND_11>
    <K2_CONSTANT_BAND_11>1201.1442</K2_CONSTANT_BAND_11>
  </LEVEL1_THERMAL_CONSTANTS>
  <LEVEL1_PROJECTION_PARAMETERS>
    <MAP_PROJECTION>UTM</MAP_PROJECTION>
    <DATUM>WGS84</DATUM>
    <ELLIPSOID>WGS84</ELLIPSOID>
    <UTM_ZONE>40</UTM_ZONE>
    <GRID_CELL_SIZE_PANCHROMATIC>15.00</GRID_CELL_SIZE_PANCHROMATIC>
    <GRID_CELL_SIZE_REFLECTIVE>30.00</GRID_CELL_SIZE_REFLECTIVE>
    <GRID_CELL_SIZE_THERMAL>30.00</GRID_CELL_SIZE_THERMAL>
    <ORIENTATION>NORTH_UP</ORIENTATION>
    <RESAMPLING_OPTION>CUBIC_CONVOLUTION</RESAMPLING_OPTION>
  </LEVEL1_PROJECTION_PARAMETERS>
</LANDSAT_METADATA_FILE>
"""


class TestOLITIRSL1:
    """Test generic image reader."""

    def setup_method(self):
        """Create temporary images and metadata to test on."""
        import tempfile
        from datetime import datetime, timezone

        from pyresample.geometry import AreaDefinition

        from satpy.scene import Scene

        self.date = datetime(2024, 5, 12, tzinfo=timezone.utc)

        self.filename_info = dict(observation_date=datetime(2024, 5, 3),
                                  platform_type="L",
                                  process_level_correction="L1TP",
                                  spacecraft_id="08",
                                  data_type="C")
        self.ftype_info = {"file_type": "granule_b04"}

        # Create area definition
        pcs_id = "WGS 84 / UTM zone 40N"
        proj4_dict = {"proj": "utm", "zone": 40, "datum": "WGS84", "units": "m", "no_defs": None, "type": "crs"}
        self.x_size = 100
        self.y_size = 100
        area_extent = (619485., 2440485., 850515., 2675715.)
        self.area_def = AreaDefinition("geotiff_area", pcs_id, pcs_id,
                                       proj4_dict, self.x_size, self.y_size,
                                       area_extent)

        # Create datasets for L, LA, RGB and RGBA mode images
        self.test_data__1 = da.random.randint(12000, 16000,
                                              size=(self.y_size, self.x_size),
                                              chunks=(50, 50)).astype(np.uint16)
        self.test_data__2 = da.random.randint(8000, 14000,
                                              size=(self.y_size, self.x_size),
                                              chunks=(50, 50)).astype(np.uint16)
        self.test_data__3 = da.random.randint(1, 10000,
                                              size=(self.y_size, self.x_size),
                                              chunks=(50, 50)).astype(np.uint16)

        ds_b4 = xr.DataArray(self.test_data__1,
                             dims=("y", "x"),
                             attrs={"name": "b04",
                                    "start_time": self.date})

        ds_b11 = xr.DataArray(self.test_data__2,
                              dims=("y", "x"),
                              attrs={"name": "b04",
                                     "start_time": self.date})

        ds_sza = xr.DataArray(self.test_data__3,
                              dims=("y", "x"),
                              attrs={"name": "sza",
                                     "start_time": self.date})

        # Temp dir for the saved images
        self.base_dir = tempfile.mkdtemp()

        # Filenames to be used during testing
        self.fnames = [f"{self.base_dir}/LC08_L1GT_026200_20240502_20240513_02_T2_B4.TIF",
                       f"{self.base_dir}/LC08_L1GT_026200_20240502_20240513_02_T2_B11.TIF",
                       f"{self.base_dir}/LC08_L1GT_026200_20240502_20240513_02_T2_MTL.xml",
                       f"{self.base_dir}/LC08_L1GT_026200_20240502_20240513_02_T2_SZA.TIF"]

        self.bad_fname_plat = self.fnames[0].replace("LC08", "BC08")
        self.bad_fname_plat2 = self.fnames[2].replace("LC08", "BC08")

        self.bad_fname_chan = self.fnames[0].replace("B4", "B5")

        # Put the datasets to Scene for easy saving
        scn = Scene()
        scn["b4"] = ds_b4
        scn["b4"].attrs["area"] = self.area_def
        scn["b11"] = ds_b11
        scn["b11"].attrs["area"] = self.area_def
        scn["sza"] = ds_sza
        scn["sza"].attrs["area"] = self.area_def

        # Save the images.  Two images in PNG and two in GeoTIFF
        scn.save_dataset("b4", writer="geotiff", enhance=False, fill_value=0,
                         filename=os.path.join(self.base_dir, self.fnames[0]))
        scn.save_dataset("b11", writer="geotiff", enhance=False, fill_value=0,
                         filename=os.path.join(self.base_dir, self.fnames[1]))
        scn.save_dataset("sza", writer="geotiff", enhance=False, fill_value=0,
                         filename=os.path.join(self.base_dir, self.fnames[3]))

        scn.save_dataset("b4", writer="geotiff", enhance=False, fill_value=0,
                         filename=self.bad_fname_plat)
        scn.save_dataset("b4", writer="geotiff", enhance=False, fill_value=0,
                         filename=self.bad_fname_chan)

        # Write the metadata to a file
        with open(os.path.join(self.base_dir, self.fnames[2]), "wb") as f:
            f.write(metadata_text)
        with open(self.bad_fname_plat2, "wb") as f:
            f.write(metadata_text)

        self.scn = scn

    def teardown_module(self):
        """Remove the temporary directory created for a test."""
        try:
            import shutil
            shutil.rmtree(self.base_dir, ignore_errors=True)
        except OSError:
            pass

    def test_basicload(self):
        """Test loading a Landsat Scene."""
        from satpy import Scene
        scn = Scene(reader="oli_tirs_l1_tif", filenames=[self.fnames[0],
                                                         self.fnames[1],
                                                         self.fnames[2]])
        scn.load(["b04", "b11"])

        # Check dataset is loaded correctly
        assert scn["b04"].shape == (100, 100)
        assert scn["b04"].attrs["area"] == self.area_def
        assert scn["b04"].attrs["saturated"]
        assert scn["b11"].shape == (100, 100)
        assert scn["b11"].attrs["area"] == self.area_def
        with pytest.raises(KeyError, match="saturated"):
            assert not scn["b11"].attrs["saturated"]

    def test_ch_startend(self):
        """Test correct retrieval of start/end times."""
        from datetime import datetime, timezone

        from satpy import Scene
        scn = Scene(reader="oli_tirs_l1_tif", filenames=[self.fnames[0], self.fnames[3], self.fnames[2]])
        bnds = scn.available_dataset_names()
        assert bnds == ["b04", "solar_zenith_angle"]

        scn.load(["b04"])
        assert scn.start_time == datetime(2024, 5, 2, 18, 0, 24, tzinfo=timezone.utc)
        assert scn.end_time == datetime(2024, 5, 2, 18, 0, 24, tzinfo=timezone.utc)

    def test_loading_gd(self):
        """Test loading a Landsat Scene with good channel requests."""
        from satpy.readers.oli_tirs_l1_tif import OLITIRSCHReader, OLITIRSMDReader
        good_mda = OLITIRSMDReader(self.fnames[2], self.filename_info, {})
        rdr = OLITIRSCHReader(self.fnames[0], self.filename_info, self.ftype_info, good_mda)

        # Check case with good file data and load request
        rdr.get_dataset({"name": "b04", "calibration": "counts"}, {"standard_name": "test_data", "units": "test_units"})

    def test_loading_badfil(self):
        """Test loading a Landsat Scene with bad channel requests."""
        from satpy.readers.oli_tirs_l1_tif import OLITIRSCHReader, OLITIRSMDReader
        good_mda = OLITIRSMDReader(self.fnames[2], self.filename_info, {})
        rdr = OLITIRSCHReader(self.fnames[0], self.filename_info, self.ftype_info, good_mda)

        ftype = {"standard_name": "test_data", "units": "test_units"}
        # Check case with request to load channel not matching filename
        with pytest.raises(ValueError, match="Requested channel b05 does not match the reader channel b04"):
            rdr.get_dataset({"name": "b05", "calibration": "counts"}, ftype)

    def test_loading_badchan(self):
        """Test loading a Landsat Scene with bad channel requests."""
        from satpy.readers.oli_tirs_l1_tif import OLITIRSCHReader, OLITIRSMDReader
        good_mda = OLITIRSMDReader(self.fnames[2], self.filename_info, {})
        ftype = {"standard_name": "test_data", "units": "test_units"}
        bad_finfo = self.filename_info.copy()
        bad_finfo["data_type"] = "T"

        # Check loading invalid channel for data type
        rdr = OLITIRSCHReader(self.fnames[1], bad_finfo, self.ftype_info, good_mda)
        with pytest.raises(ValueError, match="Requested channel b04 is not available in this granule"):
            rdr.get_dataset({"name": "b04", "calibration": "counts"}, ftype)

        bad_finfo["data_type"] = "O"
        ftype_b11 = self.ftype_info.copy()
        ftype_b11["file_type"] = "granule_b11"
        rdr = OLITIRSCHReader(self.fnames[1], bad_finfo, ftype_b11, good_mda)
        with pytest.raises(ValueError, match="Requested channel b11 is not available in this granule"):
            rdr.get_dataset({"name": "b11", "calibration": "counts"}, ftype)

    def test_badfiles(self):
        """Test loading a Landsat Scene with bad data."""
        from satpy.readers.oli_tirs_l1_tif import OLITIRSCHReader, OLITIRSMDReader
        bad_fname_info = self.filename_info.copy()
        bad_fname_info["platform_type"] = "B"

        ftype = {"standard_name": "test_data", "units": "test_units"}

        # Test that metadata reader initialises with correct filename
        good_mda = OLITIRSMDReader(self.fnames[2], self.filename_info, ftype)

        # Check metadata reader fails if platform type is wrong
        with pytest.raises(ValueError, match="This reader only supports Landsat data"):
            OLITIRSMDReader(self.fnames[2], bad_fname_info, ftype)

        # Test that metadata reader initialises with correct filename
        OLITIRSCHReader(self.fnames[0], self.filename_info, self.ftype_info, good_mda)

        # Check metadata reader fails if platform type is wrong
        with pytest.raises(ValueError, match="This reader only supports Landsat data"):
            OLITIRSCHReader(self.fnames[0], bad_fname_info, self.ftype_info, good_mda)
        bad_ftype_info = self.ftype_info.copy()
        bad_ftype_info["file_type"] = "granule-b05"
        with pytest.raises(ValueError, match="Invalid file type: granule-b05"):
            OLITIRSCHReader(self.fnames[0], self.filename_info, bad_ftype_info, good_mda)

    def test_calibration_counts(self):
        """Test counts calibration mode for the reader."""
        from satpy import Scene

        scn = Scene(reader="oli_tirs_l1_tif", filenames=self.fnames)
        scn.load(["b04", "b11"], calibration="counts")
        np.testing.assert_allclose(scn["b04"].values, self.test_data__1)
        np.testing.assert_allclose(scn["b11"].values, self.test_data__2)
        assert scn["b04"].attrs["units"] == "1"
        assert scn["b11"].attrs["units"] == "1"
        assert scn["b04"].attrs["standard_name"] == "counts"
        assert scn["b11"].attrs["standard_name"] == "counts"

    def test_calibration_radiance(self):
        """Test radiance calibration mode for the reader."""
        from satpy import Scene
        exp_b04 = (self.test_data__1 * 0.0098329 - 49.16426).astype(np.float32)
        exp_b11 = (self.test_data__2 * 0.0003342 + 0.100000).astype(np.float32)

        scn = Scene(reader="oli_tirs_l1_tif", filenames=self.fnames)
        scn.load(["b04", "b11"], calibration="radiance")
        assert scn["b04"].attrs["units"] == "W m-2 um-1 sr-1"
        assert scn["b11"].attrs["units"] == "W m-2 um-1 sr-1"
        assert scn["b04"].attrs["standard_name"] == "toa_outgoing_radiance_per_unit_wavelength"
        assert scn["b11"].attrs["standard_name"] == "toa_outgoing_radiance_per_unit_wavelength"
        np.testing.assert_allclose(scn["b04"].values, exp_b04, rtol=1e-4)
        np.testing.assert_allclose(scn["b11"].values, exp_b11, rtol=1e-4)

    def test_calibration_highlevel(self):
        """Test high level calibration modes for the reader."""
        from satpy import Scene
        exp_b04 = (self.test_data__1 * 2e-05 - 0.1).astype(np.float32) * 100
        exp_b11 = (self.test_data__2 * 0.0003342 + 0.100000)
        exp_b11 = (1201.1442 / np.log((480.8883 / exp_b11) + 1)).astype(np.float32)
        scn = Scene(reader="oli_tirs_l1_tif", filenames=self.fnames)
        scn.load(["b04", "b11"])

        assert scn["b04"].attrs["units"] == "%"
        assert scn["b11"].attrs["units"] == "K"
        assert scn["b04"].attrs["standard_name"] == "toa_bidirectional_reflectance"
        assert scn["b11"].attrs["standard_name"] == "brightness_temperature"
        np.testing.assert_allclose(np.array(scn["b04"].values), np.array(exp_b04), rtol=1e-4)
        np.testing.assert_allclose(scn["b11"].values, exp_b11, rtol=1e-6)

    def test_angles(self):
        """Test calibration modes for the reader."""
        from satpy import Scene

        # Check angles are calculated correctly
        scn = Scene(reader="oli_tirs_l1_tif", filenames=self.fnames)
        scn.load(["solar_zenith_angle"])
        assert scn["solar_zenith_angle"].attrs["units"] == "degrees"
        assert scn["solar_zenith_angle"].attrs["standard_name"] == "solar_zenith_angle"
        np.testing.assert_allclose(scn["solar_zenith_angle"].values * 100,
                                   np.array(self.test_data__3),
                                   atol=0.01,
                                   rtol=1e-3)

    def test_metadata(self):
        """Check that metadata values loaded correctly."""
        from satpy.readers.oli_tirs_l1_tif import OLITIRSMDReader
        mda = OLITIRSMDReader(self.fnames[2], self.filename_info, {})

        cal_test_dict = {"b01": (0.012357, -61.78647, 2e-05, -0.1),
                         "b05": (0.0060172, -30.08607, 2e-05, -0.1),
                         "b10": (0.0003342, 0.1, 774.8853, 1321.0789)}

        assert mda.platform_name == "Landsat-8"
        assert mda.earth_sun_distance() == 1.0079981
        assert mda.band_calibration["b01"] == cal_test_dict["b01"]
        assert mda.band_calibration["b05"] == cal_test_dict["b05"]
        assert mda.band_calibration["b10"] == cal_test_dict["b10"]
        assert not mda.band_saturation["b01"]
        assert mda.band_saturation["b04"]
        assert not mda.band_saturation["b05"]
        with pytest.raises(KeyError):
            mda.band_saturation["b10"]

    def test_area_def(self):
        """Check we can get the area defs properly."""
        from satpy.readers.oli_tirs_l1_tif import OLITIRSMDReader
        mda = OLITIRSMDReader(self.fnames[2], self.filename_info, {})

        standard_area = mda.build_area_def("b01")
        pan_area = mda.build_area_def("b08")

        assert standard_area.area_extent == (619485.0, 2440485.0, 850515.0, 2675715.0)
        assert pan_area.area_extent == (619492.5, 2440492.5, 850507.5, 2675707.5)
