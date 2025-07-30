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
    <DIGITAL_OBJECT_IDENTIFIER>https://doi.org/10.5066/P9OGBGM6</DIGITAL_OBJECT_IDENTIFIER>
    <LANDSAT_PRODUCT_ID>LC09_L2SP_029030_20240616_20240617_02_T1</LANDSAT_PRODUCT_ID>
    <PROCESSING_LEVEL>L2SP</PROCESSING_LEVEL>
    <COLLECTION_NUMBER>02</COLLECTION_NUMBER>
    <COLLECTION_CATEGORY>T1</COLLECTION_CATEGORY>
    <OUTPUT_FORMAT>GEOTIFF</OUTPUT_FORMAT>
    <FILE_NAME_BAND_1>LC09_L2SP_029030_20240616_20240617_02_T1_SR_B1.TIF</FILE_NAME_BAND_1>
    <FILE_NAME_BAND_2>LC09_L2SP_029030_20240616_20240617_02_T1_SR_B2.TIF</FILE_NAME_BAND_2>
    <FILE_NAME_BAND_3>LC09_L2SP_029030_20240616_20240617_02_T1_SR_B3.TIF</FILE_NAME_BAND_3>
    <FILE_NAME_BAND_4>LC09_L2SP_029030_20240616_20240617_02_T1_SR_B4.TIF</FILE_NAME_BAND_4>
    <FILE_NAME_BAND_5>LC09_L2SP_029030_20240616_20240617_02_T1_SR_B5.TIF</FILE_NAME_BAND_5>
    <FILE_NAME_BAND_6>LC09_L2SP_029030_20240616_20240617_02_T1_SR_B6.TIF</FILE_NAME_BAND_6>
    <FILE_NAME_BAND_7>LC09_L2SP_029030_20240616_20240617_02_T1_SR_B7.TIF</FILE_NAME_BAND_7>
    <FILE_NAME_BAND_ST_B10>LC09_L2SP_029030_20240616_20240617_02_T1_ST_B10.TIF</FILE_NAME_BAND_ST_B10>
    <FILE_NAME_THERMAL_RADIANCE>LC09_L2SP_029030_20240616_20240617_02_T1_ST_TRAD.TIF</FILE_NAME_THERMAL_RADIANCE>
    <FILE_NAME_UPWELL_RADIANCE>LC09_L2SP_029030_20240616_20240617_02_T1_ST_URAD.TIF</FILE_NAME_UPWELL_RADIANCE>
    <FILE_NAME_DOWNWELL_RADIANCE>LC09_L2SP_029030_20240616_20240617_02_T1_ST_DRAD.TIF</FILE_NAME_DOWNWELL_RADIANCE>
    <FILE_NAME_ATMOSPHERIC_TRANSMITTANCE>LC09_L2SP_029030_20240616_20240617_02_T1_ST_ATRAN.TIF</FILE_NAME_ATMOSPHERIC_TRANSMITTANCE>
    <FILE_NAME_EMISSIVITY>LC09_L2SP_029030_20240616_20240617_02_T1_ST_EMIS.TIF</FILE_NAME_EMISSIVITY>
    <FILE_NAME_EMISSIVITY_STDEV>LC09_L2SP_029030_20240616_20240617_02_T1_ST_EMSD.TIF</FILE_NAME_EMISSIVITY_STDEV>
    <FILE_NAME_CLOUD_DISTANCE>LC09_L2SP_029030_20240616_20240617_02_T1_ST_CDIST.TIF</FILE_NAME_CLOUD_DISTANCE>
    <FILE_NAME_QUALITY_L2_AEROSOL>LC09_L2SP_029030_20240616_20240617_02_T1_SR_QA_AEROSOL.TIF</FILE_NAME_QUALITY_L2_AEROSOL>
    <FILE_NAME_QUALITY_L2_SURFACE_TEMPERATURE>LC09_L2SP_029030_20240616_20240617_02_T1_ST_QA.TIF</FILE_NAME_QUALITY_L2_SURFACE_TEMPERATURE>
    <FILE_NAME_QUALITY_L1_PIXEL>LC09_L2SP_029030_20240616_20240617_02_T1_QA_PIXEL.TIF</FILE_NAME_QUALITY_L1_PIXEL>
    <FILE_NAME_QUALITY_L1_RADIOMETRIC_SATURATION>LC09_L2SP_029030_20240616_20240617_02_T1_QA_RADSAT.TIF</FILE_NAME_QUALITY_L1_RADIOMETRIC_SATURATION>
    <FILE_NAME_ANGLE_COEFFICIENT>LC09_L2SP_029030_20240616_20240617_02_T1_ANG.txt</FILE_NAME_ANGLE_COEFFICIENT>
    <FILE_NAME_METADATA_ODL>LC09_L2SP_029030_20240616_20240617_02_T1_MTL.txt</FILE_NAME_METADATA_ODL>
    <FILE_NAME_METADATA_XML>LC09_L2SP_029030_20240616_20240617_02_T1_MTL.xml</FILE_NAME_METADATA_XML>
    <DATA_TYPE_BAND_1>UINT16</DATA_TYPE_BAND_1>
    <DATA_TYPE_BAND_2>UINT16</DATA_TYPE_BAND_2>
    <DATA_TYPE_BAND_3>UINT16</DATA_TYPE_BAND_3>
    <DATA_TYPE_BAND_4>UINT16</DATA_TYPE_BAND_4>
    <DATA_TYPE_BAND_5>UINT16</DATA_TYPE_BAND_5>
    <DATA_TYPE_BAND_6>UINT16</DATA_TYPE_BAND_6>
    <DATA_TYPE_BAND_7>UINT16</DATA_TYPE_BAND_7>
    <DATA_TYPE_BAND_ST_B10>UINT16</DATA_TYPE_BAND_ST_B10>
    <DATA_TYPE_THERMAL_RADIANCE>INT16</DATA_TYPE_THERMAL_RADIANCE>
    <DATA_TYPE_UPWELL_RADIANCE>INT16</DATA_TYPE_UPWELL_RADIANCE>
    <DATA_TYPE_DOWNWELL_RADIANCE>INT16</DATA_TYPE_DOWNWELL_RADIANCE>
    <DATA_TYPE_ATMOSPHERIC_TRANSMITTANCE>INT16</DATA_TYPE_ATMOSPHERIC_TRANSMITTANCE>
    <DATA_TYPE_EMISSIVITY>INT16</DATA_TYPE_EMISSIVITY>
    <DATA_TYPE_EMISSIVITY_STDEV>INT16</DATA_TYPE_EMISSIVITY_STDEV>
    <DATA_TYPE_CLOUD_DISTANCE>INT16</DATA_TYPE_CLOUD_DISTANCE>
    <DATA_TYPE_QUALITY_L2_AEROSOL>UINT8</DATA_TYPE_QUALITY_L2_AEROSOL>
    <DATA_TYPE_QUALITY_L2_SURFACE_TEMPERATURE>INT16</DATA_TYPE_QUALITY_L2_SURFACE_TEMPERATURE>
    <DATA_TYPE_QUALITY_L1_PIXEL>UINT16</DATA_TYPE_QUALITY_L1_PIXEL>
    <DATA_TYPE_QUALITY_L1_RADIOMETRIC_SATURATION>UINT16</DATA_TYPE_QUALITY_L1_RADIOMETRIC_SATURATION>
  </PRODUCT_CONTENTS>
  <IMAGE_ATTRIBUTES>
    <SPACECRAFT_ID>LANDSAT_9</SPACECRAFT_ID>
    <SENSOR_ID>OLI_TIRS</SENSOR_ID>
    <WRS_TYPE>2</WRS_TYPE>
    <WRS_PATH>29</WRS_PATH>
    <WRS_ROW>30</WRS_ROW>
    <NADIR_OFFNADIR>NADIR</NADIR_OFFNADIR>
    <TARGET_WRS_PATH>29</TARGET_WRS_PATH>
    <TARGET_WRS_ROW>30</TARGET_WRS_ROW>
    <DATE_ACQUIRED>2024-06-16</DATE_ACQUIRED>
    <SCENE_CENTER_TIME>17:10:58.5278200Z</SCENE_CENTER_TIME>
    <STATION_ID>LGN</STATION_ID>
    <CLOUD_COVER>29.27</CLOUD_COVER>
    <CLOUD_COVER_LAND>29.27</CLOUD_COVER_LAND>
    <IMAGE_QUALITY_OLI>9</IMAGE_QUALITY_OLI>
    <IMAGE_QUALITY_TIRS>9</IMAGE_QUALITY_TIRS>
    <SATURATION_BAND_1>N</SATURATION_BAND_1>
    <SATURATION_BAND_2>N</SATURATION_BAND_2>
    <SATURATION_BAND_3>N</SATURATION_BAND_3>
    <SATURATION_BAND_4>Y</SATURATION_BAND_4>
    <SATURATION_BAND_5>N</SATURATION_BAND_5>
    <SATURATION_BAND_6>Y</SATURATION_BAND_6>
    <SATURATION_BAND_7>Y</SATURATION_BAND_7>
    <SATURATION_BAND_8>N</SATURATION_BAND_8>
    <SATURATION_BAND_9>N</SATURATION_BAND_9>
    <ROLL_ANGLE>0.000</ROLL_ANGLE>
    <SUN_AZIMUTH>134.43500878</SUN_AZIMUTH>
    <SUN_ELEVATION>64.41443455</SUN_ELEVATION>
    <EARTH_SUN_DISTANCE>1.0158933</EARTH_SUN_DISTANCE>
  </IMAGE_ATTRIBUTES>
  <PROJECTION_ATTRIBUTES>
    <MAP_PROJECTION>UTM</MAP_PROJECTION>
    <DATUM>WGS84</DATUM>
    <ELLIPSOID>WGS84</ELLIPSOID>
    <UTM_ZONE>14</UTM_ZONE>
    <GRID_CELL_SIZE_REFLECTIVE>30.00</GRID_CELL_SIZE_REFLECTIVE>
    <GRID_CELL_SIZE_THERMAL>30.00</GRID_CELL_SIZE_THERMAL>
    <REFLECTIVE_LINES>100</REFLECTIVE_LINES>
    <REFLECTIVE_SAMPLES>100</REFLECTIVE_SAMPLES>
    <THERMAL_LINES>100</THERMAL_LINES>
    <THERMAL_SAMPLES>100</THERMAL_SAMPLES>
    <ORIENTATION>NORTH_UP</ORIENTATION>
    <CORNER_UL_LAT_PRODUCT>44.24610</CORNER_UL_LAT_PRODUCT>
    <CORNER_UL_LON_PRODUCT>-98.56289</CORNER_UL_LON_PRODUCT>
    <CORNER_UR_LAT_PRODUCT>44.19877</CORNER_UR_LAT_PRODUCT>
    <CORNER_UR_LON_PRODUCT>-95.68366</CORNER_UR_LON_PRODUCT>
    <CORNER_LL_LAT_PRODUCT>42.14174</CORNER_LL_LAT_PRODUCT>
    <CORNER_LL_LON_PRODUCT>-98.57765</CORNER_LL_LON_PRODUCT>
    <CORNER_LR_LAT_PRODUCT>42.09775</CORNER_LR_LAT_PRODUCT>
    <CORNER_LR_LON_PRODUCT>-95.79546</CORNER_LR_LON_PRODUCT>
    <CORNER_UL_PROJECTION_X_PRODUCT>534900.000</CORNER_UL_PROJECTION_X_PRODUCT>
    <CORNER_UL_PROJECTION_Y_PRODUCT>4899300.000</CORNER_UL_PROJECTION_Y_PRODUCT>
    <CORNER_UR_PROJECTION_X_PRODUCT>765000.000</CORNER_UR_PROJECTION_X_PRODUCT>
    <CORNER_UR_PROJECTION_Y_PRODUCT>4899300.000</CORNER_UR_PROJECTION_Y_PRODUCT>
    <CORNER_LL_PROJECTION_X_PRODUCT>534900.000</CORNER_LL_PROJECTION_X_PRODUCT>
    <CORNER_LL_PROJECTION_Y_PRODUCT>4665600.000</CORNER_LL_PROJECTION_Y_PRODUCT>
    <CORNER_LR_PROJECTION_X_PRODUCT>765000.000</CORNER_LR_PROJECTION_X_PRODUCT>
    <CORNER_LR_PROJECTION_Y_PRODUCT>4665600.000</CORNER_LR_PROJECTION_Y_PRODUCT>
  </PROJECTION_ATTRIBUTES>
  <LEVEL2_PROCESSING_RECORD>
    <ORIGIN>Image courtesy of the U.S. Geological Survey</ORIGIN>
    <DIGITAL_OBJECT_IDENTIFIER>https://doi.org/10.5066/P9OGBGM6</DIGITAL_OBJECT_IDENTIFIER>
    <REQUEST_ID>1898780_00011</REQUEST_ID>
    <LANDSAT_PRODUCT_ID>LC09_L2SP_029030_20240616_20240617_02_T1</LANDSAT_PRODUCT_ID>
    <PROCESSING_LEVEL>L2SP</PROCESSING_LEVEL>
    <OUTPUT_FORMAT>GEOTIFF</OUTPUT_FORMAT>
    <DATE_PRODUCT_GENERATED>2024-06-17T11:12:45Z</DATE_PRODUCT_GENERATED>
    <PROCESSING_SOFTWARE_VERSION>LPGS_16.4.0</PROCESSING_SOFTWARE_VERSION>
    <ALGORITHM_SOURCE_SURFACE_REFLECTANCE>LaSRC_1.6.0</ALGORITHM_SOURCE_SURFACE_REFLECTANCE>
    <DATA_SOURCE_OZONE>VIIRS</DATA_SOURCE_OZONE>
    <DATA_SOURCE_PRESSURE>Calculated</DATA_SOURCE_PRESSURE>
    <DATA_SOURCE_WATER_VAPOR>VIIRS</DATA_SOURCE_WATER_VAPOR>
    <DATA_SOURCE_AIR_TEMPERATURE>VIIRS</DATA_SOURCE_AIR_TEMPERATURE>
    <ALGORITHM_SOURCE_SURFACE_TEMPERATURE>st_1.5.0</ALGORITHM_SOURCE_SURFACE_TEMPERATURE>
    <DATA_SOURCE_REANALYSIS>GEOS-5 IT</DATA_SOURCE_REANALYSIS>
  </LEVEL2_PROCESSING_RECORD>
  <LEVEL2_SURFACE_REFLECTANCE_PARAMETERS>
    <REFLECTANCE_MAXIMUM_BAND_1>1.602213</REFLECTANCE_MAXIMUM_BAND_1>
    <REFLECTANCE_MINIMUM_BAND_1>-0.199972</REFLECTANCE_MINIMUM_BAND_1>
    <REFLECTANCE_MAXIMUM_BAND_2>1.602213</REFLECTANCE_MAXIMUM_BAND_2>
    <REFLECTANCE_MINIMUM_BAND_2>-0.199972</REFLECTANCE_MINIMUM_BAND_2>
    <REFLECTANCE_MAXIMUM_BAND_3>1.602213</REFLECTANCE_MAXIMUM_BAND_3>
    <REFLECTANCE_MINIMUM_BAND_3>-0.199972</REFLECTANCE_MINIMUM_BAND_3>
    <REFLECTANCE_MAXIMUM_BAND_4>1.602213</REFLECTANCE_MAXIMUM_BAND_4>
    <REFLECTANCE_MINIMUM_BAND_4>-0.199972</REFLECTANCE_MINIMUM_BAND_4>
    <REFLECTANCE_MAXIMUM_BAND_5>1.602213</REFLECTANCE_MAXIMUM_BAND_5>
    <REFLECTANCE_MINIMUM_BAND_5>-0.199972</REFLECTANCE_MINIMUM_BAND_5>
    <REFLECTANCE_MAXIMUM_BAND_6>1.602213</REFLECTANCE_MAXIMUM_BAND_6>
    <REFLECTANCE_MINIMUM_BAND_6>-0.199972</REFLECTANCE_MINIMUM_BAND_6>
    <REFLECTANCE_MAXIMUM_BAND_7>1.602213</REFLECTANCE_MAXIMUM_BAND_7>
    <REFLECTANCE_MINIMUM_BAND_7>-0.199972</REFLECTANCE_MINIMUM_BAND_7>
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
    <REFLECTANCE_MULT_BAND_1>2.75e-05</REFLECTANCE_MULT_BAND_1>
    <REFLECTANCE_MULT_BAND_2>2.75e-05</REFLECTANCE_MULT_BAND_2>
    <REFLECTANCE_MULT_BAND_3>2.75e-05</REFLECTANCE_MULT_BAND_3>
    <REFLECTANCE_MULT_BAND_4>2.75e-05</REFLECTANCE_MULT_BAND_4>
    <REFLECTANCE_MULT_BAND_5>2.75e-05</REFLECTANCE_MULT_BAND_5>
    <REFLECTANCE_MULT_BAND_6>2.75e-05</REFLECTANCE_MULT_BAND_6>
    <REFLECTANCE_MULT_BAND_7>2.75e-05</REFLECTANCE_MULT_BAND_7>
    <REFLECTANCE_ADD_BAND_1>-0.2</REFLECTANCE_ADD_BAND_1>
    <REFLECTANCE_ADD_BAND_2>-0.2</REFLECTANCE_ADD_BAND_2>
    <REFLECTANCE_ADD_BAND_3>-0.2</REFLECTANCE_ADD_BAND_3>
    <REFLECTANCE_ADD_BAND_4>-0.2</REFLECTANCE_ADD_BAND_4>
    <REFLECTANCE_ADD_BAND_5>-0.2</REFLECTANCE_ADD_BAND_5>
    <REFLECTANCE_ADD_BAND_6>-0.2</REFLECTANCE_ADD_BAND_6>
    <REFLECTANCE_ADD_BAND_7>-0.2</REFLECTANCE_ADD_BAND_7>
  </LEVEL2_SURFACE_REFLECTANCE_PARAMETERS>
  <LEVEL2_SURFACE_TEMPERATURE_PARAMETERS>
    <TEMPERATURE_MAXIMUM_BAND_ST_B10>372.999941</TEMPERATURE_MAXIMUM_BAND_ST_B10>
    <TEMPERATURE_MINIMUM_BAND_ST_B10>149.003418</TEMPERATURE_MINIMUM_BAND_ST_B10>
    <QUANTIZE_CAL_MAXIMUM_BAND_ST_B10>65535</QUANTIZE_CAL_MAXIMUM_BAND_ST_B10>
    <QUANTIZE_CAL_MINIMUM_BAND_ST_B10>1</QUANTIZE_CAL_MINIMUM_BAND_ST_B10>
    <TEMPERATURE_MULT_BAND_ST_B10>0.00341802</TEMPERATURE_MULT_BAND_ST_B10>
    <TEMPERATURE_ADD_BAND_ST_B10>149.0</TEMPERATURE_ADD_BAND_ST_B10>
  </LEVEL2_SURFACE_TEMPERATURE_PARAMETERS>
  <LEVEL1_PROCESSING_RECORD>
    <ORIGIN>Image courtesy of the U.S. Geological Survey</ORIGIN>
    <DIGITAL_OBJECT_IDENTIFIER>https://doi.org/10.5066/P975CC9B</DIGITAL_OBJECT_IDENTIFIER>
    <REQUEST_ID>1898675_00011</REQUEST_ID>
    <LANDSAT_SCENE_ID>LC90290302024168LGN00</LANDSAT_SCENE_ID>
    <LANDSAT_PRODUCT_ID>LC09_L1TP_029030_20240616_20240616_02_T1</LANDSAT_PRODUCT_ID>
    <PROCESSING_LEVEL>L1TP</PROCESSING_LEVEL>
    <COLLECTION_CATEGORY>T1</COLLECTION_CATEGORY>
    <OUTPUT_FORMAT>GEOTIFF</OUTPUT_FORMAT>
    <DATE_PRODUCT_GENERATED>2024-06-16T22:54:36Z</DATE_PRODUCT_GENERATED>
    <PROCESSING_SOFTWARE_VERSION>LPGS_16.4.0</PROCESSING_SOFTWARE_VERSION>
    <FILE_NAME_BAND_1>LC09_L1TP_029030_20240616_20240616_02_T1_B1.TIF</FILE_NAME_BAND_1>
    <FILE_NAME_BAND_2>LC09_L1TP_029030_20240616_20240616_02_T1_B2.TIF</FILE_NAME_BAND_2>
    <FILE_NAME_BAND_3>LC09_L1TP_029030_20240616_20240616_02_T1_B3.TIF</FILE_NAME_BAND_3>
    <FILE_NAME_BAND_4>LC09_L1TP_029030_20240616_20240616_02_T1_B4.TIF</FILE_NAME_BAND_4>
    <FILE_NAME_BAND_5>LC09_L1TP_029030_20240616_20240616_02_T1_B5.TIF</FILE_NAME_BAND_5>
    <FILE_NAME_BAND_6>LC09_L1TP_029030_20240616_20240616_02_T1_B6.TIF</FILE_NAME_BAND_6>
    <FILE_NAME_BAND_7>LC09_L1TP_029030_20240616_20240616_02_T1_B7.TIF</FILE_NAME_BAND_7>
    <FILE_NAME_BAND_8>LC09_L1TP_029030_20240616_20240616_02_T1_B8.TIF</FILE_NAME_BAND_8>
    <FILE_NAME_BAND_9>LC09_L1TP_029030_20240616_20240616_02_T1_B9.TIF</FILE_NAME_BAND_9>
    <FILE_NAME_BAND_10>LC09_L1TP_029030_20240616_20240616_02_T1_B10.TIF</FILE_NAME_BAND_10>
    <FILE_NAME_BAND_11>LC09_L1TP_029030_20240616_20240616_02_T1_B11.TIF</FILE_NAME_BAND_11>
    <FILE_NAME_QUALITY_L1_PIXEL>LC09_L1TP_029030_20240616_20240616_02_T1_QA_PIXEL.TIF</FILE_NAME_QUALITY_L1_PIXEL>
    <FILE_NAME_QUALITY_L1_RADIOMETRIC_SATURATION>LC09_L1TP_029030_20240616_20240616_02_T1_QA_RADSAT.TIF</FILE_NAME_QUALITY_L1_RADIOMETRIC_SATURATION>
    <FILE_NAME_ANGLE_COEFFICIENT>LC09_L1TP_029030_20240616_20240616_02_T1_ANG.txt</FILE_NAME_ANGLE_COEFFICIENT>
    <FILE_NAME_ANGLE_SENSOR_AZIMUTH_BAND_4>LC09_L1TP_029030_20240616_20240616_02_T1_VAA.TIF</FILE_NAME_ANGLE_SENSOR_AZIMUTH_BAND_4>
    <FILE_NAME_ANGLE_SENSOR_ZENITH_BAND_4>LC09_L1TP_029030_20240616_20240616_02_T1_VZA.TIF</FILE_NAME_ANGLE_SENSOR_ZENITH_BAND_4>
    <FILE_NAME_ANGLE_SOLAR_AZIMUTH_BAND_4>LC09_L1TP_029030_20240616_20240616_02_T1_SAA.TIF</FILE_NAME_ANGLE_SOLAR_AZIMUTH_BAND_4>
    <FILE_NAME_ANGLE_SOLAR_ZENITH_BAND_4>LC09_L1TP_029030_20240616_20240616_02_T1_SZA.TIF</FILE_NAME_ANGLE_SOLAR_ZENITH_BAND_4>
    <FILE_NAME_METADATA_ODL>LC09_L1TP_029030_20240616_20240616_02_T1_MTL.txt</FILE_NAME_METADATA_ODL>
    <FILE_NAME_METADATA_XML>LC09_L1TP_029030_20240616_20240616_02_T1_MTL.xml</FILE_NAME_METADATA_XML>
    <FILE_NAME_CPF>LC09CPF_20240401_20240630_02.02</FILE_NAME_CPF>
    <FILE_NAME_BPF_OLI>LO9BPF20240616165050_20240616171843.02</FILE_NAME_BPF_OLI>
    <FILE_NAME_BPF_TIRS>LT9BPF20240616164559_20240616182452.01</FILE_NAME_BPF_TIRS>
    <FILE_NAME_RLUT>LC09RLUT_20230701_20531231_02_10.h5</FILE_NAME_RLUT>
    <DATA_SOURCE_ELEVATION>GLS2000</DATA_SOURCE_ELEVATION>
    <GROUND_CONTROL_POINTS_VERSION>5</GROUND_CONTROL_POINTS_VERSION>
    <GROUND_CONTROL_POINTS_MODEL>160</GROUND_CONTROL_POINTS_MODEL>
    <GEOMETRIC_RMSE_MODEL>7.390</GEOMETRIC_RMSE_MODEL>
    <GEOMETRIC_RMSE_MODEL_Y>5.361</GEOMETRIC_RMSE_MODEL_Y>
    <GEOMETRIC_RMSE_MODEL_X>5.086</GEOMETRIC_RMSE_MODEL_X>
  </LEVEL1_PROCESSING_RECORD>
  <LEVEL1_MIN_MAX_RADIANCE>
    <RADIANCE_MAXIMUM_BAND_1>734.06683</RADIANCE_MAXIMUM_BAND_1>
    <RADIANCE_MINIMUM_BAND_1>-60.61948</RADIANCE_MINIMUM_BAND_1>
    <RADIANCE_MAXIMUM_BAND_2>753.94604</RADIANCE_MAXIMUM_BAND_2>
    <RADIANCE_MINIMUM_BAND_2>-62.26111</RADIANCE_MINIMUM_BAND_2>
    <RADIANCE_MAXIMUM_BAND_3>695.55688</RADIANCE_MAXIMUM_BAND_3>
    <RADIANCE_MINIMUM_BAND_3>-57.43931</RADIANCE_MINIMUM_BAND_3>
    <RADIANCE_MAXIMUM_BAND_4>586.00403</RADIANCE_MAXIMUM_BAND_4>
    <RADIANCE_MINIMUM_BAND_4>-48.39240</RADIANCE_MINIMUM_BAND_4>
    <RADIANCE_MAXIMUM_BAND_5>359.16302</RADIANCE_MAXIMUM_BAND_5>
    <RADIANCE_MINIMUM_BAND_5>-29.65980</RADIANCE_MINIMUM_BAND_5>
    <RADIANCE_MAXIMUM_BAND_6>89.27634</RADIANCE_MAXIMUM_BAND_6>
    <RADIANCE_MINIMUM_BAND_6>-7.37247</RADIANCE_MINIMUM_BAND_6>
    <RADIANCE_MAXIMUM_BAND_7>30.08380</RADIANCE_MAXIMUM_BAND_7>
    <RADIANCE_MINIMUM_BAND_7>-2.48433</RADIANCE_MINIMUM_BAND_7>
    <RADIANCE_MAXIMUM_BAND_8>661.33466</RADIANCE_MAXIMUM_BAND_8>
    <RADIANCE_MINIMUM_BAND_8>-54.61323</RADIANCE_MINIMUM_BAND_8>
    <RADIANCE_MAXIMUM_BAND_9>140.38083</RADIANCE_MAXIMUM_BAND_9>
    <RADIANCE_MINIMUM_BAND_9>-11.59269</RADIANCE_MINIMUM_BAND_9>
    <RADIANCE_MAXIMUM_BAND_10>25.00330</RADIANCE_MAXIMUM_BAND_10>
    <RADIANCE_MINIMUM_BAND_10>0.10038</RADIANCE_MINIMUM_BAND_10>
    <RADIANCE_MAXIMUM_BAND_11>22.97172</RADIANCE_MAXIMUM_BAND_11>
    <RADIANCE_MINIMUM_BAND_11>0.10035</RADIANCE_MINIMUM_BAND_11>
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
    <RADIANCE_MULT_BAND_1>1.2126E-02</RADIANCE_MULT_BAND_1>
    <RADIANCE_MULT_BAND_2>1.2455E-02</RADIANCE_MULT_BAND_2>
    <RADIANCE_MULT_BAND_3>1.1490E-02</RADIANCE_MULT_BAND_3>
    <RADIANCE_MULT_BAND_4>9.6804E-03</RADIANCE_MULT_BAND_4>
    <RADIANCE_MULT_BAND_5>5.9331E-03</RADIANCE_MULT_BAND_5>
    <RADIANCE_MULT_BAND_6>1.4748E-03</RADIANCE_MULT_BAND_6>
    <RADIANCE_MULT_BAND_7>4.9697E-04</RADIANCE_MULT_BAND_7>
    <RADIANCE_MULT_BAND_8>1.0925E-02</RADIANCE_MULT_BAND_8>
    <RADIANCE_MULT_BAND_9>2.3190E-03</RADIANCE_MULT_BAND_9>
    <RADIANCE_MULT_BAND_10>3.8000E-04</RADIANCE_MULT_BAND_10>
    <RADIANCE_MULT_BAND_11>3.4900E-04</RADIANCE_MULT_BAND_11>
    <RADIANCE_ADD_BAND_1>-60.63160</RADIANCE_ADD_BAND_1>
    <RADIANCE_ADD_BAND_2>-62.27356</RADIANCE_ADD_BAND_2>
    <RADIANCE_ADD_BAND_3>-57.45080</RADIANCE_ADD_BAND_3>
    <RADIANCE_ADD_BAND_4>-48.40208</RADIANCE_ADD_BAND_4>
    <RADIANCE_ADD_BAND_5>-29.66573</RADIANCE_ADD_BAND_5>
    <RADIANCE_ADD_BAND_6>-7.37394</RADIANCE_ADD_BAND_6>
    <RADIANCE_ADD_BAND_7>-2.48483</RADIANCE_ADD_BAND_7>
    <RADIANCE_ADD_BAND_8>-54.62416</RADIANCE_ADD_BAND_8>
    <RADIANCE_ADD_BAND_9>-11.59501</RADIANCE_ADD_BAND_9>
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
    <K1_CONSTANT_BAND_10>799.0284</K1_CONSTANT_BAND_10>
    <K2_CONSTANT_BAND_10>1329.2405</K2_CONSTANT_BAND_10>
    <K1_CONSTANT_BAND_11>475.6581</K1_CONSTANT_BAND_11>
    <K2_CONSTANT_BAND_11>1198.3494</K2_CONSTANT_BAND_11>
  </LEVEL1_THERMAL_CONSTANTS>
  <LEVEL1_PROJECTION_PARAMETERS>
    <MAP_PROJECTION>UTM</MAP_PROJECTION>
    <DATUM>WGS84</DATUM>
    <ELLIPSOID>WGS84</ELLIPSOID>
    <UTM_ZONE>14</UTM_ZONE>
    <GRID_CELL_SIZE_PANCHROMATIC>15.00</GRID_CELL_SIZE_PANCHROMATIC>
    <GRID_CELL_SIZE_REFLECTIVE>30.00</GRID_CELL_SIZE_REFLECTIVE>
    <GRID_CELL_SIZE_THERMAL>30.00</GRID_CELL_SIZE_THERMAL>
    <ORIENTATION>NORTH_UP</ORIENTATION>
    <RESAMPLING_OPTION>CUBIC_CONVOLUTION</RESAMPLING_OPTION>
  </LEVEL1_PROJECTION_PARAMETERS>
</LANDSAT_METADATA_FILE>
"""


x_size = 100
y_size = 100
date = datetime(2024, 6, 16, tzinfo=timezone.utc)


@pytest.fixture(scope="session")
def area():
    """Get the landsat 1 area def."""
    pcs_id = "WGS84 / UTM zone 14N"
    proj4_dict = {"proj": "utm", "zone": 14, "datum": "WGS84", "units": "m", "no_defs": None, "type": "crs"}
    area_extent = (534885.0, 4665585.0, 765015.0, 4899315.0)
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
def b10_data():
    """Get the data for the b11 channel."""
    return da.random.randint(8000, 14000,
                             size=(y_size, x_size),
                             chunks=(50, 50)).astype(np.uint16)


@pytest.fixture(scope="session")
def rad_data():
    """Get the data for the radiance channel."""
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
    return tmp_path_factory.mktemp("oli_tirs_l2_files")


@pytest.fixture(scope="session")
def b4_file(files_path, b4_data, area):
    """Create the file for the b4 channel."""
    data = b4_data
    filename = files_path / "LC09_L2SP_029030_20240616_20240617_02_T1_SR_B4.TIF"
    name = "B4"
    create_tif_file(data, name, area, filename)
    return os.fspath(filename)

@pytest.fixture(scope="session")
def b10_file(files_path, b10_data, area):
    """Create the file for the b11 channel."""
    data = b10_data
    filename = files_path / "LC09_L2SP_029030_20240616_20240617_02_T1_ST_B10.TIF"
    name = "B10"
    create_tif_file(data, name, area, filename)
    return os.fspath(filename)

@pytest.fixture(scope="session")
def rad_file(files_path, rad_data, area):
    """Create the file for the sza."""
    data = rad_data
    filename = files_path / "LC09_L2SP_029030_20240616_20240617_02_T1_ST_TRAD.TIF"
    name = "TRAD"
    create_tif_file(data, name, area, filename)
    return os.fspath(filename)


@pytest.fixture(scope="session")
def mda_file(files_path):
    """Create the metadata xml file."""
    filename = files_path / "LC09_L2SP_029030_20240616_20240617_02_T1_MTL.xml"
    with open(filename, "wb") as f:
        f.write(metadata_text)
    return os.fspath(filename)


@pytest.fixture(scope="session")
def all_files(b4_file, b10_file, mda_file, rad_file):
    """Return all the files."""
    return b4_file, b10_file, mda_file, rad_file


@pytest.fixture(scope="session")
def all_fsspec_files(b4_file, b10_file, mda_file, rad_file):
    """Return all the files as FSFile objects."""
    from fsspec.implementations.local import LocalFileSystem

    from satpy.readers.core.remote import FSFile

    fs = LocalFileSystem()
    b4_file, b10_file, mda_file, rad_file = (
        FSFile(os.path.abspath(file), fs=fs)
        for file in [b4_file, b10_file, mda_file, rad_file]
    )
    return b4_file, b10_file, mda_file, rad_file


class TestOLITIRSL2:
    """Test generic image reader."""

    def setup_method(self, tmp_path):
        """Set up the filename and filetype info dicts.."""
        self.filename_info = dict(observation_date=datetime(2024, 5, 3),
                                  platform_type="L",
                                  process_level_correction="L2SP",
                                  spacecraft_id="09",
                                  data_type="C",
                                  collection_id="02")
        self.ftype_info = {"file_type": "granule_B4"}

    def test_basicload(self, area, b4_file, b10_file, mda_file):
        """Test loading a Landsat Scene."""
        scn = Scene(reader="oli_tirs_l2_tif", filenames=[b4_file,
                                                         b10_file,
                                                         mda_file])
        scn.load(["B4", "B10"])

        # Check dataset is loaded correctly
        assert scn["B4"].shape == (100, 100)
        assert scn["B4"].attrs["area"] == area
        assert scn["B4"].attrs["saturated"]
        assert scn["B10"].shape == (100, 100)
        assert scn["B10"].attrs["area"] == area
        with pytest.raises(KeyError, match="saturated"):
            assert not scn["B10"].attrs["saturated"]

    def test_ch_startend(self, b4_file, mda_file):
        """Test correct retrieval of start/end times."""
        scn = Scene(reader="oli_tirs_l2_tif", filenames=[b4_file, mda_file])
        bnds = scn.available_dataset_names()
        assert bnds == ["B4"]

        scn.load(["B4"])
        assert scn.start_time == datetime(2024, 6, 16, 17, 10, 58, tzinfo=timezone.utc)
        assert scn.end_time == datetime(2024, 6, 16, 17, 10, 58, tzinfo=timezone.utc)

    def test_loading_gd(self, mda_file, b4_file):
        """Test loading a Landsat Scene with good channel requests."""
        from satpy.readers.oli_tirs_l1_tif import OLITIRSL2CHReader, OLITIRSL2MDReader
        good_mda = OLITIRSL2MDReader(mda_file, self.filename_info, {})
        rdr = OLITIRSL2CHReader(b4_file, self.filename_info, self.ftype_info, good_mda)

        # Check case with good file data and load request
        rdr.get_dataset({"name": "B4", "calibration": "counts"}, {"standard_name": "test_data", "units": "test_units"})

    def test_loading_badfil(self, mda_file, b4_file):
        """Test loading a Landsat Scene with bad channel requests."""
        from satpy.readers.oli_tirs_l1_tif import OLITIRSL2CHReader, OLITIRSL2MDReader
        good_mda = OLITIRSL2MDReader(mda_file, self.filename_info, {})
        rdr = OLITIRSL2CHReader(b4_file, self.filename_info, self.ftype_info, good_mda)

        ftype = {"standard_name": "test_data", "units": "test_units"}
        # Check case with request to load channel not matching filename
        with pytest.raises(ValueError, match="Requested channel B5 does not match the reader channel B4"):
            rdr.get_dataset({"name": "B5", "calibration": "counts"}, ftype)

    def test_loading_badchan(self, mda_file, b10_file):
        """Test loading a Landsat Scene with bad channel requests."""
        from satpy.readers.oli_tirs_l1_tif import OLITIRSL2CHReader, OLITIRSL2MDReader
        good_mda = OLITIRSL2MDReader(mda_file, self.filename_info, {})
        ftype = {"standard_name": "test_data", "units": "test_units"}
        bad_finfo = self.filename_info.copy()
        bad_finfo["data_type"] = "T"

        # Check loading invalid channel for data type
        rdr = OLITIRSL2CHReader(b10_file, bad_finfo, self.ftype_info, good_mda)
        with pytest.raises(ValueError, match="Requested channel B4 is not available in this granule"):
            rdr.get_dataset({"name": "B4", "calibration": "counts"}, ftype)

        bad_finfo["data_type"] = "O"
        ftype_b10 = self.ftype_info.copy()
        ftype_b10["file_type"] = "granule_B10"
        rdr = OLITIRSL2CHReader(b10_file, bad_finfo, ftype_b10, good_mda)
        with pytest.raises(ValueError, match="Requested channel B10 is not available in this granule"):
            rdr.get_dataset({"name": "B10", "calibration": "counts"}, ftype)

    def test_badfiles(self, mda_file, b4_file):
        """Test loading a Landsat Scene with bad data."""
        from satpy.readers.oli_tirs_l1_tif import OLITIRSL2CHReader, OLITIRSL2MDReader
        bad_fname_info = self.filename_info.copy()
        bad_fname_info["platform_type"] = "B"

        ftype = {"standard_name": "test_data", "units": "test_units"}

        # Test that metadata reader initialises with correct filename
        good_mda = OLITIRSL2MDReader(mda_file, self.filename_info, ftype)

        # Check metadata reader fails if platform type is wrong
        with pytest.raises(ValueError, match="This reader only supports Landsat data"):
            OLITIRSL2MDReader(mda_file, bad_fname_info, ftype)

        # Test that metadata reader initialises with correct filename
        OLITIRSL2CHReader(b4_file, self.filename_info, self.ftype_info, good_mda)

        # Check metadata reader fails if platform type is wrong
        with pytest.raises(ValueError, match="This reader only supports Landsat data"):
            OLITIRSL2CHReader(b4_file, bad_fname_info, self.ftype_info, good_mda)
        bad_ftype_info = self.ftype_info.copy()
        bad_ftype_info["file_type"] = "granule-b05"
        with pytest.raises(ValueError, match="Invalid file type: granule-b05"):
            OLITIRSL2CHReader(b4_file, self.filename_info, bad_ftype_info, good_mda)

    def test_calibration_counts(self, all_files, b4_data, b10_data, rad_data):
        """Test counts calibration mode for the reader."""
        from satpy import Scene

        scn = Scene(reader="oli_tirs_l2_tif", filenames=all_files)
        scn.load(["B4", "B10", "TRAD"], calibration="counts")
        np.testing.assert_allclose(scn["B4"].values, b4_data)
        np.testing.assert_allclose(scn["B10"].values, b10_data)
        np.testing.assert_allclose(scn["TRAD"].values, rad_data)
        assert scn["B4"].attrs["units"] == "1"
        assert scn["B10"].attrs["units"] == "1"
        assert scn["TRAD"].attrs["units"] == "1"
        assert scn["B4"].attrs["standard_name"] == "counts"
        assert scn["B10"].attrs["standard_name"] == "counts"
        assert scn["TRAD"].attrs["standard_name"] == "counts"

    def test_calibration_highlevel(self, all_files, b4_data, b10_data, rad_data):
        """Test high level calibration modes for the reader."""
        from satpy import Scene
        exp_b04 = (b4_data * 2.75e-05 - 0.2).astype(np.float32) * 100
        exp_b10 = (b10_data * 0.00341802 + 149.0).astype(np.float32)
        exp_rad = (rad_data * 0.001).astype(np.float32)
        scn = Scene(reader="oli_tirs_l2_tif", filenames=all_files)
        scn.load(["B4", "B10", "TRAD"])

        assert scn["B4"].attrs["units"] == "%"
        assert scn["B10"].attrs["units"] == "K"
        assert scn["TRAD"].attrs["units"] == "W m-2 um-1 sr-1"
        assert scn["B4"].attrs["standard_name"] == "surface_bidirectional_reflectance"
        assert scn["B10"].attrs["standard_name"] == "brightness_temperature"
        assert scn["TRAD"].attrs["standard_name"] == "toa_outgoing_radiance_per_unit_wavelength"
        np.testing.assert_allclose(np.array(scn["B4"].values), np.array(exp_b04), rtol=1e-4)
        np.testing.assert_allclose(scn["B10"].values, exp_b10, rtol=1e-6)
        np.testing.assert_allclose(scn["TRAD"].values, exp_rad, rtol=1e-6)

    def test_metadata(self, mda_file):
        """Check that metadata values loaded correctly."""
        from satpy.readers.oli_tirs_l1_tif import OLITIRSL2MDReader
        mda = OLITIRSL2MDReader(mda_file, self.filename_info, {})

        cal_test_dict = {"B1": (2.75e-05, -0.2),
                         "B5": (2.75e-05, -0.2),
                         "B10": (0.00341802, 149.0)}

        assert mda.platform_name == "Landsat-9"
        assert mda.earth_sun_distance() == 1.0158933
        assert mda.band_calibration["B1"] == cal_test_dict["B1"]
        assert mda.band_calibration["B5"] == cal_test_dict["B5"]
        assert mda.band_calibration["B10"] == cal_test_dict["B10"]
        assert not mda.band_saturation["B1"]
        assert mda.band_saturation["B4"]
        assert not mda.band_saturation["B5"]
        with pytest.raises(KeyError):
            mda.band_saturation["B10"]

    def test_area_def(self, mda_file):
        """Check we can get the area defs properly."""
        from satpy.readers.oli_tirs_l1_tif import OLITIRSL2MDReader
        mda = OLITIRSL2MDReader(mda_file, self.filename_info, {})

        standard_area = mda.build_area_def("B1")

        assert standard_area.area_extent == (534885.0, 4665585.0, 765015.0, 4899315.0)

    def test_basicload_remote(self, area, all_fsspec_files):
        """Test loading a Landsat Scene from a fsspec filesystem."""
        scn = Scene(reader="oli_tirs_l2_tif", filenames=all_fsspec_files)
        scn.load(["B4", "B10"])

        # Check dataset is loaded correctly
        assert scn["B4"].shape == (100, 100)
        assert scn["B4"].attrs["area"] == area
        assert scn["B4"].attrs["saturated"]
        assert scn["B10"].shape == (100, 100)
        assert scn["B10"].attrs["area"] == area
        with pytest.raises(KeyError, match="saturated"):
            assert not scn["B10"].attrs["saturated"]
