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
from datetime import datetime, timezone

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pyresample.geometry import AreaDefinition

from satpy import Scene

metadata_text = b"""<?xml version="1.0" encoding="UTF-8"?>
<LANDSAT_METADATA_FILE>
  <PRODUCT_CONTENTS>
    <ORIGIN>Image courtesy of the U.S. Geological Survey</ORIGIN>
    <DIGITAL_OBJECT_IDENTIFIER>https://doi.org/10.5066/P918ROHC</DIGITAL_OBJECT_IDENTIFIER>
    <LANDSAT_PRODUCT_ID>LT04_L1TP_143021_19890818_20200916_02_T1</LANDSAT_PRODUCT_ID>
    <PROCESSING_LEVEL>L1TP</PROCESSING_LEVEL>
    <COLLECTION_NUMBER>02</COLLECTION_NUMBER>
    <COLLECTION_CATEGORY>T1</COLLECTION_CATEGORY>
    <OUTPUT_FORMAT>GEOTIFF</OUTPUT_FORMAT>
    <FILE_NAME_BAND_1>LT04_L1TP_143021_19890818_20200916_02_T1_B1.TIF</FILE_NAME_BAND_1>
    <FILE_NAME_BAND_2>LT04_L1TP_143021_19890818_20200916_02_T1_B2.TIF</FILE_NAME_BAND_2>
    <FILE_NAME_BAND_3>LT04_L1TP_143021_19890818_20200916_02_T1_B3.TIF</FILE_NAME_BAND_3>
    <FILE_NAME_BAND_4>LT04_L1TP_143021_19890818_20200916_02_T1_B4.TIF</FILE_NAME_BAND_4>
    <FILE_NAME_BAND_5>LT04_L1TP_143021_19890818_20200916_02_T1_B5.TIF</FILE_NAME_BAND_5>
    <FILE_NAME_BAND_6>LT04_L1TP_143021_19890818_20200916_02_T1_B6.TIF</FILE_NAME_BAND_6>
    <FILE_NAME_BAND_7>LT04_L1TP_143021_19890818_20200916_02_T1_B7.TIF</FILE_NAME_BAND_7>
    <FILE_NAME_QUALITY_L1_PIXEL>LT04_L1TP_143021_19890818_20200916_02_T1_QA_PIXEL.TIF</FILE_NAME_QUALITY_L1_PIXEL>
    <FILE_NAME_QUALITY_L1_RADIOMETRIC_SATURATION>LT04_L1TP_143021_19890818_20200916_02_T1_QA_RADSAT.TIF</FILE_NAME_QUALITY_L1_RADIOMETRIC_SATURATION>
    <FILE_NAME_GROUND_CONTROL_POINT>LT04_L1TP_143021_19890818_20200916_02_T1_GCP.txt</FILE_NAME_GROUND_CONTROL_POINT>
    <FILE_NAME_ANGLE_COEFFICIENT>LT04_L1TP_143021_19890818_20200916_02_T1_ANG.txt</FILE_NAME_ANGLE_COEFFICIENT>
    <FILE_NAME_ANGLE_SENSOR_AZIMUTH_BAND_4>LT04_L1TP_143021_19890818_20200916_02_T1_VAA.TIF</FILE_NAME_ANGLE_SENSOR_AZIMUTH_BAND_4>
    <FILE_NAME_ANGLE_SENSOR_ZENITH_BAND_4>LT04_L1TP_143021_19890818_20200916_02_T1_VZA.TIF</FILE_NAME_ANGLE_SENSOR_ZENITH_BAND_4>
    <FILE_NAME_ANGLE_SOLAR_AZIMUTH_BAND_4>LT04_L1TP_143021_19890818_20200916_02_T1_SAA.TIF</FILE_NAME_ANGLE_SOLAR_AZIMUTH_BAND_4>
    <FILE_NAME_ANGLE_SOLAR_ZENITH_BAND_4>LT04_L1TP_143021_19890818_20200916_02_T1_SZA.TIF</FILE_NAME_ANGLE_SOLAR_ZENITH_BAND_4>
    <FILE_NAME_METADATA_ODL>LT04_L1TP_143021_19890818_20200916_02_T1_MTL.txt</FILE_NAME_METADATA_ODL>
    <FILE_NAME_METADATA_XML>LT04_L1TP_143021_19890818_20200916_02_T1_MTL.xml</FILE_NAME_METADATA_XML>
    <FILE_NAME_VERIFY_REPORT>LT04_L1TP_143021_19890818_20200916_02_T1_VER.txt</FILE_NAME_VERIFY_REPORT>
    <FILE_NAME_VERIFY_BROWSE>LT04_L1TP_143021_19890818_20200916_02_T1_VER.jpg</FILE_NAME_VERIFY_BROWSE>
    <DATA_TYPE_BAND_1>UINT8</DATA_TYPE_BAND_1>
    <DATA_TYPE_BAND_2>UINT8</DATA_TYPE_BAND_2>
    <DATA_TYPE_BAND_3>UINT8</DATA_TYPE_BAND_3>
    <DATA_TYPE_BAND_4>UINT8</DATA_TYPE_BAND_4>
    <DATA_TYPE_BAND_5>UINT8</DATA_TYPE_BAND_5>
    <DATA_TYPE_BAND_6>UINT8</DATA_TYPE_BAND_6>
    <DATA_TYPE_BAND_7>UINT8</DATA_TYPE_BAND_7>
    <DATA_TYPE_QUALITY_L1_PIXEL>UINT16</DATA_TYPE_QUALITY_L1_PIXEL>
    <DATA_TYPE_QUALITY_L1_RADIOMETRIC_SATURATION>UINT16</DATA_TYPE_QUALITY_L1_RADIOMETRIC_SATURATION>
    <DATA_TYPE_ANGLE_SENSOR_AZIMUTH_BAND_4>INT16</DATA_TYPE_ANGLE_SENSOR_AZIMUTH_BAND_4>
    <DATA_TYPE_ANGLE_SENSOR_ZENITH_BAND_4>INT16</DATA_TYPE_ANGLE_SENSOR_ZENITH_BAND_4>
    <DATA_TYPE_ANGLE_SOLAR_AZIMUTH_BAND_4>INT16</DATA_TYPE_ANGLE_SOLAR_AZIMUTH_BAND_4>
    <DATA_TYPE_ANGLE_SOLAR_ZENITH_BAND_4>INT16</DATA_TYPE_ANGLE_SOLAR_ZENITH_BAND_4>
  </PRODUCT_CONTENTS>
  <IMAGE_ATTRIBUTES>
    <SPACECRAFT_ID>LANDSAT_4</SPACECRAFT_ID>
    <SENSOR_ID>TM</SENSOR_ID>
    <WRS_TYPE>2</WRS_TYPE>
    <WRS_PATH>143</WRS_PATH>
    <WRS_ROW>021</WRS_ROW>
    <DATE_ACQUIRED>1989-08-18</DATE_ACQUIRED>
    <SCENE_CENTER_TIME>04:26:11.9550880Z</SCENE_CENTER_TIME>
    <STATION_ID>XXX</STATION_ID>
    <CLOUD_COVER>10.00</CLOUD_COVER>
    <CLOUD_COVER_LAND>10.00</CLOUD_COVER_LAND>
    <IMAGE_QUALITY>9</IMAGE_QUALITY>
    <SATURATION_BAND_1>N</SATURATION_BAND_1>
    <SATURATION_BAND_2>Y</SATURATION_BAND_2>
    <SATURATION_BAND_3>Y</SATURATION_BAND_3>
    <SATURATION_BAND_4>Y</SATURATION_BAND_4>
    <SATURATION_BAND_5>N</SATURATION_BAND_5>
    <SATURATION_BAND_6>N</SATURATION_BAND_6>
    <SATURATION_BAND_7>Y</SATURATION_BAND_7>
    <SUN_AZIMUTH>149.14166856</SUN_AZIMUTH>
    <SUN_ELEVATION>43.85182285</SUN_ELEVATION>
    <EARTH_SUN_DISTANCE>1.0122057</EARTH_SUN_DISTANCE>
    <SENSOR_MODE>SAM</SENSOR_MODE>
    <SENSOR_MODE_SLC>ON</SENSOR_MODE_SLC>
    <SENSOR_ANOMALIES>NONE</SENSOR_ANOMALIES>
  </IMAGE_ATTRIBUTES>
  <PROJECTION_ATTRIBUTES>
    <MAP_PROJECTION>UTM</MAP_PROJECTION>
    <DATUM>WGS84</DATUM>
    <ELLIPSOID>WGS84</ELLIPSOID>
    <UTM_ZONE>46</UTM_ZONE>
    <GRID_CELL_SIZE_REFLECTIVE>30.00</GRID_CELL_SIZE_REFLECTIVE>
    <GRID_CELL_SIZE_THERMAL>30.00</GRID_CELL_SIZE_THERMAL>
    <REFLECTIVE_LINES>100</REFLECTIVE_LINES>
    <REFLECTIVE_SAMPLES>100</REFLECTIVE_SAMPLES>
    <THERMAL_LINES>100</THERMAL_LINES>
    <THERMAL_SAMPLES>100</THERMAL_SAMPLES>
    <ORIENTATION>NORTH_UP</ORIENTATION>
    <CORNER_UL_LAT_PRODUCT>56.91211</CORNER_UL_LAT_PRODUCT>
    <CORNER_UL_LON_PRODUCT>90.07952</CORNER_UL_LON_PRODUCT>
    <CORNER_UR_LAT_PRODUCT>56.94129</CORNER_UR_LAT_PRODUCT>
    <CORNER_UR_LON_PRODUCT>94.11108</CORNER_UR_LON_PRODUCT>
    <CORNER_LL_LAT_PRODUCT>54.88218</CORNER_LL_LAT_PRODUCT>
    <CORNER_LL_LON_PRODUCT>90.22826</CORNER_LL_LON_PRODUCT>
    <CORNER_LR_LAT_PRODUCT>54.90923</CORNER_LR_LAT_PRODUCT>
    <CORNER_LR_LON_PRODUCT>94.05441</CORNER_LR_LON_PRODUCT>
    <CORNER_UL_PROJECTION_X_PRODUCT>322200.000</CORNER_UL_PROJECTION_X_PRODUCT>
    <CORNER_UL_PROJECTION_Y_PRODUCT>6311400.000</CORNER_UL_PROJECTION_Y_PRODUCT>
    <CORNER_UR_PROJECTION_X_PRODUCT>567600.000</CORNER_UR_PROJECTION_X_PRODUCT>
    <CORNER_UR_PROJECTION_Y_PRODUCT>6311400.000</CORNER_UR_PROJECTION_Y_PRODUCT>
    <CORNER_LL_PROJECTION_X_PRODUCT>322200.000</CORNER_LL_PROJECTION_X_PRODUCT>
    <CORNER_LL_PROJECTION_Y_PRODUCT>6085200.000</CORNER_LL_PROJECTION_Y_PRODUCT>
    <CORNER_LR_PROJECTION_X_PRODUCT>567600.000</CORNER_LR_PROJECTION_X_PRODUCT>
    <CORNER_LR_PROJECTION_Y_PRODUCT>6085200.000</CORNER_LR_PROJECTION_Y_PRODUCT>
  </PROJECTION_ATTRIBUTES>
  <LEVEL1_PROCESSING_RECORD>
    <ORIGIN>Image courtesy of the U.S. Geological Survey</ORIGIN>
    <DIGITAL_OBJECT_IDENTIFIER>https://doi.org/10.5066/P918ROHC</DIGITAL_OBJECT_IDENTIFIER>
    <REQUEST_ID>L2</REQUEST_ID>
    <LANDSAT_SCENE_ID>LT41430211989230XXX02</LANDSAT_SCENE_ID>
    <LANDSAT_PRODUCT_ID>LT04_L1TP_143021_19890818_20200916_02_T1</LANDSAT_PRODUCT_ID>
    <PROCESSING_LEVEL>L1TP</PROCESSING_LEVEL>
    <COLLECTION_CATEGORY>T1</COLLECTION_CATEGORY>
    <OUTPUT_FORMAT>GEOTIFF</OUTPUT_FORMAT>
    <DATE_PRODUCT_GENERATED>2020-09-16T12:08:23Z</DATE_PRODUCT_GENERATED>
    <PROCESSING_SOFTWARE_VERSION>LPGS_15.3.1c</PROCESSING_SOFTWARE_VERSION>
    <FILE_NAME_BAND_1>LT04_L1TP_143021_19890818_20200916_02_T1_B1.TIF</FILE_NAME_BAND_1>
    <FILE_NAME_BAND_2>LT04_L1TP_143021_19890818_20200916_02_T1_B2.TIF</FILE_NAME_BAND_2>
    <FILE_NAME_BAND_3>LT04_L1TP_143021_19890818_20200916_02_T1_B3.TIF</FILE_NAME_BAND_3>
    <FILE_NAME_BAND_4>LT04_L1TP_143021_19890818_20200916_02_T1_B4.TIF</FILE_NAME_BAND_4>
    <FILE_NAME_BAND_5>LT04_L1TP_143021_19890818_20200916_02_T1_B5.TIF</FILE_NAME_BAND_5>
    <FILE_NAME_BAND_6>LT04_L1TP_143021_19890818_20200916_02_T1_B6.TIF</FILE_NAME_BAND_6>
    <FILE_NAME_BAND_7>LT04_L1TP_143021_19890818_20200916_02_T1_B7.TIF</FILE_NAME_BAND_7>
    <FILE_NAME_QUALITY_L1_PIXEL>LT04_L1TP_143021_19890818_20200916_02_T1_QA_PIXEL.TIF</FILE_NAME_QUALITY_L1_PIXEL>
    <FILE_NAME_QUALITY_L1_RADIOMETRIC_SATURATION>LT04_L1TP_143021_19890818_20200916_02_T1_QA_RADSAT.TIF</FILE_NAME_QUALITY_L1_RADIOMETRIC_SATURATION>
    <FILE_NAME_GROUND_CONTROL_POINT>LT04_L1TP_143021_19890818_20200916_02_T1_GCP.txt</FILE_NAME_GROUND_CONTROL_POINT>
    <FILE_NAME_ANGLE_COEFFICIENT>LT04_L1TP_143021_19890818_20200916_02_T1_ANG.txt</FILE_NAME_ANGLE_COEFFICIENT>
    <FILE_NAME_ANGLE_SENSOR_AZIMUTH_BAND_4>LT04_L1TP_143021_19890818_20200916_02_T1_VAA.TIF</FILE_NAME_ANGLE_SENSOR_AZIMUTH_BAND_4>
    <FILE_NAME_ANGLE_SENSOR_ZENITH_BAND_4>LT04_L1TP_143021_19890818_20200916_02_T1_VZA.TIF</FILE_NAME_ANGLE_SENSOR_ZENITH_BAND_4>
    <FILE_NAME_ANGLE_SOLAR_AZIMUTH_BAND_4>LT04_L1TP_143021_19890818_20200916_02_T1_SAA.TIF</FILE_NAME_ANGLE_SOLAR_AZIMUTH_BAND_4>
    <FILE_NAME_ANGLE_SOLAR_ZENITH_BAND_4>LT04_L1TP_143021_19890818_20200916_02_T1_SZA.TIF</FILE_NAME_ANGLE_SOLAR_ZENITH_BAND_4>
    <FILE_NAME_METADATA_ODL>LT04_L1TP_143021_19890818_20200916_02_T1_MTL.txt</FILE_NAME_METADATA_ODL>
    <FILE_NAME_METADATA_XML>LT04_L1TP_143021_19890818_20200916_02_T1_MTL.xml</FILE_NAME_METADATA_XML>
    <FILE_NAME_CPF>LT04CPF_19890701_19890930_02.01</FILE_NAME_CPF>
    <FILE_NAME_VERIFY_REPORT>LT04_L1TP_143021_19890818_20200916_02_T1_VER.txt</FILE_NAME_VERIFY_REPORT>
    <FILE_NAME_VERIFY_BROWSE>LT04_L1TP_143021_19890818_20200916_02_T1_VER.jpg</FILE_NAME_VERIFY_BROWSE>
    <DATA_SOURCE_ELEVATION>GLS2000</DATA_SOURCE_ELEVATION>
    <GROUND_CONTROL_POINTS_VERSION>5</GROUND_CONTROL_POINTS_VERSION>
    <GROUND_CONTROL_POINTS_MODEL>918</GROUND_CONTROL_POINTS_MODEL>
    <GEOMETRIC_RMSE_MODEL>4.609</GEOMETRIC_RMSE_MODEL>
    <GEOMETRIC_RMSE_MODEL_Y>3.369</GEOMETRIC_RMSE_MODEL_Y>
    <GEOMETRIC_RMSE_MODEL_X>3.145</GEOMETRIC_RMSE_MODEL_X>
    <GROUND_CONTROL_POINTS_VERIFY>2236</GROUND_CONTROL_POINTS_VERIFY>
    <GEOMETRIC_RMSE_VERIFY>0.233</GEOMETRIC_RMSE_VERIFY>
    <GEOMETRIC_RMSE_VERIFY_QUAD_UL>0.195</GEOMETRIC_RMSE_VERIFY_QUAD_UL>
    <GEOMETRIC_RMSE_VERIFY_QUAD_UR>0.211</GEOMETRIC_RMSE_VERIFY_QUAD_UR>
    <GEOMETRIC_RMSE_VERIFY_QUAD_LL>0.296</GEOMETRIC_RMSE_VERIFY_QUAD_LL>
    <GEOMETRIC_RMSE_VERIFY_QUAD_LR>0.247</GEOMETRIC_RMSE_VERIFY_QUAD_LR>
    <EPHEMERIS_TYPE>DEFINITIVE</EPHEMERIS_TYPE>
  </LEVEL1_PROCESSING_RECORD>
  <LEVEL1_MIN_MAX_RADIANCE>
    <RADIANCE_MAXIMUM_BAND_1>171.000</RADIANCE_MAXIMUM_BAND_1>
    <RADIANCE_MINIMUM_BAND_1>-1.520</RADIANCE_MINIMUM_BAND_1>
    <RADIANCE_MAXIMUM_BAND_2>336.000</RADIANCE_MAXIMUM_BAND_2>
    <RADIANCE_MINIMUM_BAND_2>-2.840</RADIANCE_MINIMUM_BAND_2>
    <RADIANCE_MAXIMUM_BAND_3>254.000</RADIANCE_MAXIMUM_BAND_3>
    <RADIANCE_MINIMUM_BAND_3>-1.170</RADIANCE_MINIMUM_BAND_3>
    <RADIANCE_MAXIMUM_BAND_4>221.000</RADIANCE_MAXIMUM_BAND_4>
    <RADIANCE_MINIMUM_BAND_4>-1.510</RADIANCE_MINIMUM_BAND_4>
    <RADIANCE_MAXIMUM_BAND_5>31.400</RADIANCE_MAXIMUM_BAND_5>
    <RADIANCE_MINIMUM_BAND_5>-0.370</RADIANCE_MINIMUM_BAND_5>
    <RADIANCE_MAXIMUM_BAND_6>15.303</RADIANCE_MAXIMUM_BAND_6>
    <RADIANCE_MINIMUM_BAND_6>1.238</RADIANCE_MINIMUM_BAND_6>
    <RADIANCE_MAXIMUM_BAND_7>16.600</RADIANCE_MAXIMUM_BAND_7>
    <RADIANCE_MINIMUM_BAND_7>-0.150</RADIANCE_MINIMUM_BAND_7>
  </LEVEL1_MIN_MAX_RADIANCE>
  <LEVEL1_MIN_MAX_REFLECTANCE>
    <REFLECTANCE_MAXIMUM_BAND_1>0.283277</REFLECTANCE_MAXIMUM_BAND_1>
    <REFLECTANCE_MINIMUM_BAND_1>-0.002518</REFLECTANCE_MINIMUM_BAND_1>
    <REFLECTANCE_MAXIMUM_BAND_2>0.615188</REFLECTANCE_MAXIMUM_BAND_2>
    <REFLECTANCE_MINIMUM_BAND_2>-0.005200</REFLECTANCE_MINIMUM_BAND_2>
    <REFLECTANCE_MAXIMUM_BAND_3>0.550547</REFLECTANCE_MAXIMUM_BAND_3>
    <REFLECTANCE_MINIMUM_BAND_3>-0.002536</REFLECTANCE_MINIMUM_BAND_3>
    <REFLECTANCE_MAXIMUM_BAND_4>0.688620</REFLECTANCE_MAXIMUM_BAND_4>
    <REFLECTANCE_MINIMUM_BAND_4>-0.004705</REFLECTANCE_MINIMUM_BAND_4>
    <REFLECTANCE_MAXIMUM_BAND_5>0.455881</REFLECTANCE_MAXIMUM_BAND_5>
    <REFLECTANCE_MINIMUM_BAND_5>-0.005372</REFLECTANCE_MINIMUM_BAND_5>
    <REFLECTANCE_MAXIMUM_BAND_7>0.641894</REFLECTANCE_MAXIMUM_BAND_7>
    <REFLECTANCE_MINIMUM_BAND_7>-0.005800</REFLECTANCE_MINIMUM_BAND_7>
  </LEVEL1_MIN_MAX_REFLECTANCE>
  <LEVEL1_MIN_MAX_PIXEL_VALUE>
    <QUANTIZE_CAL_MAX_BAND_1>255</QUANTIZE_CAL_MAX_BAND_1>
    <QUANTIZE_CAL_MIN_BAND_1>1</QUANTIZE_CAL_MIN_BAND_1>
    <QUANTIZE_CAL_MAX_BAND_2>255</QUANTIZE_CAL_MAX_BAND_2>
    <QUANTIZE_CAL_MIN_BAND_2>1</QUANTIZE_CAL_MIN_BAND_2>
    <QUANTIZE_CAL_MAX_BAND_3>255</QUANTIZE_CAL_MAX_BAND_3>
    <QUANTIZE_CAL_MIN_BAND_3>1</QUANTIZE_CAL_MIN_BAND_3>
    <QUANTIZE_CAL_MAX_BAND_4>255</QUANTIZE_CAL_MAX_BAND_4>
    <QUANTIZE_CAL_MIN_BAND_4>1</QUANTIZE_CAL_MIN_BAND_4>
    <QUANTIZE_CAL_MAX_BAND_5>255</QUANTIZE_CAL_MAX_BAND_5>
    <QUANTIZE_CAL_MIN_BAND_5>1</QUANTIZE_CAL_MIN_BAND_5>
    <QUANTIZE_CAL_MAX_BAND_6>255</QUANTIZE_CAL_MAX_BAND_6>
    <QUANTIZE_CAL_MIN_BAND_6>1</QUANTIZE_CAL_MIN_BAND_6>
    <QUANTIZE_CAL_MAX_BAND_7>255</QUANTIZE_CAL_MAX_BAND_7>
    <QUANTIZE_CAL_MIN_BAND_7>1</QUANTIZE_CAL_MIN_BAND_7>
  </LEVEL1_MIN_MAX_PIXEL_VALUE>
  <LEVEL1_RADIOMETRIC_RESCALING>
    <RADIANCE_MULT_BAND_1>6.7921E-01</RADIANCE_MULT_BAND_1>
    <RADIANCE_MULT_BAND_2>1.3340E+00</RADIANCE_MULT_BAND_2>
    <RADIANCE_MULT_BAND_3>1.0046E+00</RADIANCE_MULT_BAND_3>
    <RADIANCE_MULT_BAND_4>8.7602E-01</RADIANCE_MULT_BAND_4>
    <RADIANCE_MULT_BAND_5>1.2508E-01</RADIANCE_MULT_BAND_5>
    <RADIANCE_MULT_BAND_6>5.5375E-02</RADIANCE_MULT_BAND_6>
    <RADIANCE_MULT_BAND_7>6.5945E-02</RADIANCE_MULT_BAND_7>
    <RADIANCE_ADD_BAND_1>-2.19921</RADIANCE_ADD_BAND_1>
    <RADIANCE_ADD_BAND_2>-4.17402</RADIANCE_ADD_BAND_2>
    <RADIANCE_ADD_BAND_3>-2.17461</RADIANCE_ADD_BAND_3>
    <RADIANCE_ADD_BAND_4>-2.38602</RADIANCE_ADD_BAND_4>
    <RADIANCE_ADD_BAND_5>-0.49508</RADIANCE_ADD_BAND_5>
    <RADIANCE_ADD_BAND_6>1.18243</RADIANCE_ADD_BAND_6>
    <RADIANCE_ADD_BAND_7>-0.21594</RADIANCE_ADD_BAND_7>
    <REFLECTANCE_MULT_BAND_1>1.1252E-03</REFLECTANCE_MULT_BAND_1>
    <REFLECTANCE_MULT_BAND_2>2.4425E-03</REFLECTANCE_MULT_BAND_2>
    <REFLECTANCE_MULT_BAND_3>2.1775E-03</REFLECTANCE_MULT_BAND_3>
    <REFLECTANCE_MULT_BAND_4>2.7296E-03</REFLECTANCE_MULT_BAND_4>
    <REFLECTANCE_MULT_BAND_5>1.8160E-03</REFLECTANCE_MULT_BAND_5>
    <REFLECTANCE_MULT_BAND_7>2.5500E-03</REFLECTANCE_MULT_BAND_7>
    <REFLECTANCE_ADD_BAND_1>-0.003643</REFLECTANCE_ADD_BAND_1>
    <REFLECTANCE_ADD_BAND_2>-0.007642</REFLECTANCE_ADD_BAND_2>
    <REFLECTANCE_ADD_BAND_3>-0.004713</REFLECTANCE_ADD_BAND_3>
    <REFLECTANCE_ADD_BAND_4>-0.007435</REFLECTANCE_ADD_BAND_4>
    <REFLECTANCE_ADD_BAND_5>-0.007188</REFLECTANCE_ADD_BAND_5>
    <REFLECTANCE_ADD_BAND_7>-0.008350</REFLECTANCE_ADD_BAND_7>
  </LEVEL1_RADIOMETRIC_RESCALING>
  <LEVEL1_THERMAL_CONSTANTS>
    <K1_CONSTANT_BAND_6>671.62</K1_CONSTANT_BAND_6>
    <K2_CONSTANT_BAND_6>1284.30</K2_CONSTANT_BAND_6>
  </LEVEL1_THERMAL_CONSTANTS>
  <LEVEL1_PROJECTION_PARAMETERS>
    <MAP_PROJECTION>UTM</MAP_PROJECTION>
    <DATUM>WGS84</DATUM>
    <ELLIPSOID>WGS84</ELLIPSOID>
    <UTM_ZONE>46</UTM_ZONE>
    <GRID_CELL_SIZE_REFLECTIVE>30.00</GRID_CELL_SIZE_REFLECTIVE>
    <GRID_CELL_SIZE_THERMAL>30.00</GRID_CELL_SIZE_THERMAL>
    <ORIENTATION>NORTH_UP</ORIENTATION>
    <RESAMPLING_OPTION>CUBIC_CONVOLUTION</RESAMPLING_OPTION>
    <MAP_PROJECTION_L0RA>NA</MAP_PROJECTION_L0RA>
  </LEVEL1_PROJECTION_PARAMETERS>
  <PRODUCT_PARAMETERS>
    <DATA_TYPE_L0RP>TMR_L0RP</DATA_TYPE_L0RP>
    <CORRECTION_GAIN_BAND_1>CPF</CORRECTION_GAIN_BAND_1>
    <CORRECTION_GAIN_BAND_2>CPF</CORRECTION_GAIN_BAND_2>
    <CORRECTION_GAIN_BAND_3>CPF</CORRECTION_GAIN_BAND_3>
    <CORRECTION_GAIN_BAND_4>CPF</CORRECTION_GAIN_BAND_4>
    <CORRECTION_GAIN_BAND_5>CPF</CORRECTION_GAIN_BAND_5>
    <CORRECTION_GAIN_BAND_6>INTERNAL_CALIBRATION</CORRECTION_GAIN_BAND_6>
    <CORRECTION_GAIN_BAND_7>CPF</CORRECTION_GAIN_BAND_7>
    <CORRECTION_BIAS_BAND_1>INTERNAL_CALIBRATION</CORRECTION_BIAS_BAND_1>
    <CORRECTION_BIAS_BAND_2>INTERNAL_CALIBRATION</CORRECTION_BIAS_BAND_2>
    <CORRECTION_BIAS_BAND_3>INTERNAL_CALIBRATION</CORRECTION_BIAS_BAND_3>
    <CORRECTION_BIAS_BAND_4>INTERNAL_CALIBRATION</CORRECTION_BIAS_BAND_4>
    <CORRECTION_BIAS_BAND_5>INTERNAL_CALIBRATION</CORRECTION_BIAS_BAND_5>
    <CORRECTION_BIAS_BAND_6>INTERNAL_CALIBRATION</CORRECTION_BIAS_BAND_6>
    <CORRECTION_BIAS_BAND_7>INTERNAL_CALIBRATION</CORRECTION_BIAS_BAND_7>
  </PRODUCT_PARAMETERS>
</LANDSAT_METADATA_FILE>
"""


x_size = 100
y_size = 100
date = datetime(1989, 8, 18, tzinfo=timezone.utc)


@pytest.fixture(scope="session")
def area():
    """Get the landsat 1 area def."""
    pcs_id = "WGS84 / UTM zone 46N"
    proj4_dict = {"proj": "utm", "zone": 46, "datum": "WGS84", "units": "m", "no_defs": None, "type": "crs"}
    area_extent = (322185.0, 6085185.0, 567615.0, 6311415.0)
    return AreaDefinition("geotiff_area", pcs_id, pcs_id,
                          proj4_dict, x_size, y_size,
                          area_extent)


@pytest.fixture(scope="session")
def b4_data():
    """Get the data for the b4 channel."""
    return da.random.randint(12000, 16000,
                             size=(y_size, x_size),
                             chunks=(50, 50)).astype(np.uint16)


@pytest.fixture(scope="session")
def b6_data():
    """Get the data for the b6 channel."""
    return da.random.randint(8000, 14000,
                             size=(y_size, x_size),
                             chunks=(50, 50)).astype(np.uint16)


@pytest.fixture(scope="session")
def sza_data():
    """Get the data for the sza."""
    return da.random.randint(1, 10000,
                             size=(y_size, x_size),
                             chunks=(50, 50)).astype(np.uint16)


def create_tif_file(data, name, area, filename):
    """Create a tif file."""
    data_array = xr.DataArray(data,
                              dims=("y", "x"),
                              attrs={"name": name,
                                     "start_time": date})
    scn = Scene()
    scn["band_data"] = data_array
    scn["band_data"].attrs["area"] = area
    scn.save_dataset("band_data", writer="geotiff", enhance=False, fill_value=0,
                     filename=os.fspath(filename))


@pytest.fixture(scope="session")
def files_path(tmp_path_factory):
    """Create the path for l1 files."""
    return tmp_path_factory.mktemp("tm_l1_files")


@pytest.fixture(scope="session")
def b4_file(files_path, b4_data, area):
    """Create the file for the b4 channel."""
    data = b4_data
    filename = files_path / "LT04_L1TP_143021_19890818_20200916_02_T1_B4.TIF"
    name = "B4"
    create_tif_file(data, name, area, filename)
    return os.fspath(filename)

@pytest.fixture(scope="session")
def b6_file(files_path, b6_data, area):
    """Create the file for the b6 channel."""
    data = b6_data
    filename = files_path / "LT04_L1TP_143021_19890818_20200916_02_T1_B6.TIF"
    name = "B6"
    create_tif_file(data, name, area, filename)
    return os.fspath(filename)

@pytest.fixture(scope="session")
def sza_file(files_path, sza_data, area):
    """Create the file for the sza."""
    data = sza_data
    filename = files_path / "LT04_L1TP_143021_19890818_20200916_02_T1_SZA.TIF"
    name = "sza"
    create_tif_file(data, name, area, filename)
    return os.fspath(filename)


@pytest.fixture(scope="session")
def mda_file(files_path):
    """Create the metadata xml file."""
    filename = files_path / "LT04_L1TP_143021_19890818_20200916_02_T1_MTL.xml"
    with open(filename, "wb") as f:
        f.write(metadata_text)
    return os.fspath(filename)


@pytest.fixture(scope="session")
def all_files(b4_file, b6_file, mda_file, sza_file):
    """Return all the files."""
    return b4_file, b6_file, mda_file, sza_file


@pytest.fixture(scope="session")
def all_fs_files(b4_file, b6_file, mda_file, sza_file):
    """Return all the files as FSFile objects."""
    from fsspec.implementations.local import LocalFileSystem

    from satpy.readers.core.remote import FSFile

    fs = LocalFileSystem()
    b4_file, b6_file, mda_file, sza_file = (
        FSFile(os.path.abspath(file), fs=fs)
        for file in [b4_file, b6_file, mda_file, sza_file]
    )
    return b4_file, b6_file, mda_file, sza_file


class TestTML1:
    """Test generic image reader."""

    def setup_method(self, tmp_path):
        """Set up the filename and filetype info dicts.."""
        self.filename_info = dict(observation_date=datetime(1989, 8, 18),
                                  platform_type="L",
                                  process_level_correction="L1TP",
                                  spacecraft_id="04",
                                  data_type="T",
                                  collection_id="02")
        self.ftype_info = {"file_type": "granule_B4"}

    def test_basicload(self, area, b4_file, b6_file, mda_file):
        """Test loading a Landsat Scene."""
        scn = Scene(reader="tm_l1_tif", filenames=[b4_file,
                                                   b6_file,
                                                   mda_file])
        scn.load(["B4", "B6"])

        # Check dataset is loaded correctly
        assert scn["B4"].shape == (100, 100)
        assert scn["B4"].attrs["area"] == area
        assert scn["B4"].attrs["saturated"]
        assert scn["B6"].shape == (100, 100)
        assert scn["B6"].attrs["area"] == area
        assert not scn["B6"].attrs["saturated"]

    def test_ch_startend(self, b4_file, sza_file, mda_file):
        """Test correct retrieval of start/end times."""
        scn = Scene(reader="tm_l1_tif", filenames=[b4_file,
                                                   sza_file,
                                                   mda_file])
        bnds = scn.available_dataset_names()
        assert bnds == ["B4", "solar_zenith_angle"]

        scn.load(["B4"])
        assert scn.start_time == datetime(1989, 8, 18, 4, 26, 11, tzinfo=timezone.utc)
        assert scn.end_time == datetime(1989, 8, 18, 4, 26, 11, tzinfo=timezone.utc)

    def test_loading_gd(self, mda_file, b4_file):
        """Test loading a Landsat Scene with good channel requests."""
        from satpy.readers.landsat_base import TMCHReader, TMMDReader
        good_mda = TMMDReader(mda_file, self.filename_info, {})
        rdr = TMCHReader(b4_file, self.filename_info, self.ftype_info, good_mda)

        # Check case with good file data and load request
        rdr.get_dataset({"name": "B4", "calibration": "counts"}, {"standard_name": "test_data", "units": "test_units"})

    def test_loading_badfil(self, mda_file, b4_file):
        """Test loading a Landsat Scene with bad channel requests."""
        from satpy.readers.landsat_base import TMCHReader, TMMDReader
        good_mda = TMMDReader(mda_file, self.filename_info, {})
        rdr = TMCHReader(b4_file, self.filename_info, self.ftype_info, good_mda)

        ftype = {"standard_name": "test_data", "units": "test_units"}
        # Check case with request to load channel not matching filename
        with pytest.raises(ValueError, match="Requested channel B5 does not match the reader channel B4"):
            rdr.get_dataset({"name": "B5", "calibration": "counts"}, ftype)

    def test_badfiles(self, mda_file, b4_file):
        """Test loading a Landsat Scene with bad data."""
        from satpy.readers.landsat_base import TMCHReader, TMMDReader
        bad_fname_info = self.filename_info.copy()
        bad_fname_info["platform_type"] = "B"

        ftype = {"standard_name": "test_data", "units": "test_units"}

        # Test that metadata reader initialises with correct filename
        good_mda = TMMDReader(mda_file, self.filename_info, ftype)

        # Check metadata reader fails if platform type is wrong
        with pytest.raises(ValueError, match="This reader only supports Landsat data"):
            TMMDReader(mda_file, bad_fname_info, ftype)

        # Test that metadata reader initialises with correct filename
        TMCHReader(b4_file, self.filename_info, self.ftype_info, good_mda)

        # Check metadata reader fails if platform type is wrong
        with pytest.raises(ValueError, match="This reader only supports Landsat data"):
            TMCHReader(b4_file, bad_fname_info, self.ftype_info, good_mda)
        bad_ftype_info = self.ftype_info.copy()
        bad_ftype_info["file_type"] = "granule-b05"
        with pytest.raises(ValueError, match="Invalid file type: granule-b05"):
            TMCHReader(b4_file, self.filename_info, bad_ftype_info, good_mda)

    def test_calibration_counts(self, all_files, b4_data, b6_data):
        """Test counts calibration mode for the reader."""
        from satpy import Scene

        scn = Scene(reader="tm_l1_tif", filenames=all_files)
        scn.load(["B4", "B6"], calibration="counts")
        np.testing.assert_allclose(scn["B4"].values, b4_data)
        np.testing.assert_allclose(scn["B6"].values, b6_data)
        assert scn["B4"].attrs["units"] == "1"
        assert scn["B6"].attrs["units"] == "1"
        assert scn["B4"].attrs["standard_name"] == "counts"
        assert scn["B6"].attrs["standard_name"] == "counts"

    def test_calibration_radiance(self, all_files, b4_data, b6_data):
        """Test radiance calibration mode for the reader."""
        from satpy import Scene
        exp_b04 = (b4_data * 8.7602e-01 - 2.38602).astype(np.float32)
        exp_b6 = (b6_data * 5.5375e-02 + 1.18243).astype(np.float32)

        scn = Scene(reader="tm_l1_tif", filenames=all_files)
        scn.load(["B4", "B6"], calibration="radiance")
        assert scn["B4"].attrs["units"] == "W m-2 um-1 sr-1"
        assert scn["B6"].attrs["units"] == "W m-2 um-1 sr-1"
        assert scn["B4"].attrs["standard_name"] == "toa_outgoing_radiance_per_unit_wavelength"
        assert scn["B6"].attrs["standard_name"] == "toa_outgoing_radiance_per_unit_wavelength"
        np.testing.assert_allclose(scn["B4"].values, exp_b04, rtol=1e-4)
        np.testing.assert_allclose(scn["B6"].values, exp_b6, rtol=1e-4)

    def test_calibration_highlevel(self, all_files, b4_data, b6_data):
        """Test high level calibration modes for the reader."""
        from satpy import Scene
        exp_b04 = (b4_data * 2.7296e-03 - 0.007435).astype(np.float32) * 100
        exp_b6 = (b6_data * 5.5375e-02 + 1.18243).astype(np.float32)
        exp_b6 = (1284.30 / np.log((671.62 / exp_b6) + 1)).astype(np.float32)
        scn = Scene(reader="tm_l1_tif", filenames=all_files)
        scn.load(["B4", "B6"])

        assert scn["B4"].attrs["units"] == "%"
        assert scn["B6"].attrs["units"] == "K"
        assert scn["B4"].attrs["standard_name"] == "toa_bidirectional_reflectance"
        assert scn["B6"].attrs["standard_name"] == "brightness_temperature"
        np.testing.assert_allclose(np.array(scn["B4"].values), np.array(exp_b04), rtol=1e-4)
        np.testing.assert_allclose(scn["B6"].values, exp_b6, rtol=1e-6)

    def test_angles(self, all_files, sza_data):
        """Test calibration modes for the reader."""
        from satpy import Scene

        # Check angles are calculated correctly
        scn = Scene(reader="tm_l1_tif", filenames=all_files)
        scn.load(["solar_zenith_angle"])
        assert scn["solar_zenith_angle"].attrs["units"] == "degrees"
        assert scn["solar_zenith_angle"].attrs["standard_name"] == "solar_zenith_angle"
        np.testing.assert_allclose(scn["solar_zenith_angle"].values * 100,
                                   np.array(sza_data),
                                   atol=0.01,
                                   rtol=1e-3)

    def test_metadata(self, mda_file):
        """Check that metadata values loaded correctly."""
        from satpy.readers.landsat_base import TMMDReader
        mda = TMMDReader(mda_file, self.filename_info, {})

        cal_test_dict = {"B1": (6.7921e-01, -2.19921, 1.1252e-03, -0.003643),
                         "B5": (1.2508e-01, -0.49508, 1.8160e-03, -0.007188),
                         "B6": (5.5375e-02, 1.18243, 671.62, 1284.30)}

        assert mda.platform_name == "Landsat-4"
        assert mda.earth_sun_distance() == 1.0122057
        assert mda.band_calibration["B1"] == cal_test_dict["B1"]
        assert mda.band_calibration["B5"] == cal_test_dict["B5"]
        assert mda.band_calibration["B6"] == cal_test_dict["B6"]
        assert not mda.band_saturation["B1"]
        assert mda.band_saturation["B4"]
        assert not mda.band_saturation["B5"]
        assert not mda.band_saturation["B6"]

    def test_area_def(self, mda_file):
        """Check we can get the area defs properly."""
        from satpy.readers.landsat_base import TMMDReader
        mda = TMMDReader(mda_file, self.filename_info, {})

        standard_area = mda.build_area_def("B1")

        assert standard_area.area_extent == (322185.0, 6085185.0, 567615.0, 6311415.0)

    def test_basicload_remote(self, area, all_fs_files):
        """Test loading a Landsat Scene from a fsspec filesystem."""
        scn = Scene(reader="tm_l1_tif", filenames=all_fs_files)
        scn.load(["B4", "B6"])

        # Check dataset is loaded correctly
        assert scn["B4"].shape == (100, 100)
        assert scn["B4"].attrs["area"] == area
        assert scn["B4"].attrs["saturated"]
        assert scn["B6"].shape == (100, 100)
        assert scn["B6"].attrs["area"] == area
        assert not scn["B6"].attrs["saturated"]
