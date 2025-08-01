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
    <DIGITAL_OBJECT_IDENTIFIER>https://doi.org/10.5066/P9C7I13B</DIGITAL_OBJECT_IDENTIFIER>
    <LANDSAT_PRODUCT_ID>LE07_L2SP_028030_20230817_20230912_02_T1</LANDSAT_PRODUCT_ID>
    <PROCESSING_LEVEL>L2SP</PROCESSING_LEVEL>
    <COLLECTION_NUMBER>02</COLLECTION_NUMBER>
    <COLLECTION_CATEGORY>T1</COLLECTION_CATEGORY>
    <OUTPUT_FORMAT>GEOTIFF</OUTPUT_FORMAT>
    <FILE_NAME_BAND_1>LE07_L2SP_028030_20230817_20230912_02_T1_SR_B1.TIF</FILE_NAME_BAND_1>
    <FILE_NAME_BAND_2>LE07_L2SP_028030_20230817_20230912_02_T1_SR_B2.TIF</FILE_NAME_BAND_2>
    <FILE_NAME_BAND_3>LE07_L2SP_028030_20230817_20230912_02_T1_SR_B3.TIF</FILE_NAME_BAND_3>
    <FILE_NAME_BAND_4>LE07_L2SP_028030_20230817_20230912_02_T1_SR_B4.TIF</FILE_NAME_BAND_4>
    <FILE_NAME_BAND_5>LE07_L2SP_028030_20230817_20230912_02_T1_SR_B5.TIF</FILE_NAME_BAND_5>
    <FILE_NAME_BAND_ST_B6>LE07_L2SP_028030_20230817_20230912_02_T1_ST_B6.TIF</FILE_NAME_BAND_ST_B6>
    <FILE_NAME_BAND_7>LE07_L2SP_028030_20230817_20230912_02_T1_SR_B7.TIF</FILE_NAME_BAND_7>
    <FILE_NAME_THERMAL_RADIANCE>LE07_L2SP_028030_20230817_20230912_02_T1_ST_TRAD.TIF</FILE_NAME_THERMAL_RADIANCE>
    <FILE_NAME_UPWELL_RADIANCE>LE07_L2SP_028030_20230817_20230912_02_T1_ST_URAD.TIF</FILE_NAME_UPWELL_RADIANCE>
    <FILE_NAME_DOWNWELL_RADIANCE>LE07_L2SP_028030_20230817_20230912_02_T1_ST_DRAD.TIF</FILE_NAME_DOWNWELL_RADIANCE>
    <FILE_NAME_ATMOSPHERIC_TRANSMITTANCE>LE07_L2SP_028030_20230817_20230912_02_T1_ST_ATRAN.TIF</FILE_NAME_ATMOSPHERIC_TRANSMITTANCE>
    <FILE_NAME_EMISSIVITY>LE07_L2SP_028030_20230817_20230912_02_T1_ST_EMIS.TIF</FILE_NAME_EMISSIVITY>
    <FILE_NAME_EMISSIVITY_STDEV>LE07_L2SP_028030_20230817_20230912_02_T1_ST_EMSD.TIF</FILE_NAME_EMISSIVITY_STDEV>
    <FILE_NAME_CLOUD_DISTANCE>LE07_L2SP_028030_20230817_20230912_02_T1_ST_CDIST.TIF</FILE_NAME_CLOUD_DISTANCE>
    <FILE_NAME_ATMOSPHERIC_OPACITY>LE07_L2SP_028030_20230817_20230912_02_T1_SR_ATMOS_OPACITY.TIF</FILE_NAME_ATMOSPHERIC_OPACITY>
    <FILE_NAME_QUALITY_L2_SURFACE_REFLECTANCE_CLOUD>LE07_L2SP_028030_20230817_20230912_02_T1_SR_CLOUD_QA.TIF</FILE_NAME_QUALITY_L2_SURFACE_REFLECTANCE_CLOUD>
    <FILE_NAME_QUALITY_L2_SURFACE_TEMPERATURE>LE07_L2SP_028030_20230817_20230912_02_T1_ST_QA.TIF</FILE_NAME_QUALITY_L2_SURFACE_TEMPERATURE>
    <FILE_NAME_QUALITY_L1_PIXEL>LE07_L2SP_028030_20230817_20230912_02_T1_QA_PIXEL.TIF</FILE_NAME_QUALITY_L1_PIXEL>
    <FILE_NAME_QUALITY_L1_RADIOMETRIC_SATURATION>LE07_L2SP_028030_20230817_20230912_02_T1_QA_RADSAT.TIF</FILE_NAME_QUALITY_L1_RADIOMETRIC_SATURATION>
    <FILE_NAME_ANGLE_COEFFICIENT>LE07_L2SP_028030_20230817_20230912_02_T1_ANG.txt</FILE_NAME_ANGLE_COEFFICIENT>
    <FILE_NAME_METADATA_ODL>LE07_L2SP_028030_20230817_20230912_02_T1_MTL.txt</FILE_NAME_METADATA_ODL>
    <FILE_NAME_METADATA_XML>LE07_L2SP_028030_20230817_20230912_02_T1_MTL.xml</FILE_NAME_METADATA_XML>
    <DATA_TYPE_BAND_1>UINT16</DATA_TYPE_BAND_1>
    <DATA_TYPE_BAND_2>UINT16</DATA_TYPE_BAND_2>
    <DATA_TYPE_BAND_3>UINT16</DATA_TYPE_BAND_3>
    <DATA_TYPE_BAND_4>UINT16</DATA_TYPE_BAND_4>
    <DATA_TYPE_BAND_5>UINT16</DATA_TYPE_BAND_5>
    <DATA_TYPE_BAND_ST_B6>UINT16</DATA_TYPE_BAND_ST_B6>
    <DATA_TYPE_BAND_7>UINT16</DATA_TYPE_BAND_7>
    <DATA_TYPE_THERMAL_RADIANCE>INT16</DATA_TYPE_THERMAL_RADIANCE>
    <DATA_TYPE_UPWELL_RADIANCE>INT16</DATA_TYPE_UPWELL_RADIANCE>
    <DATA_TYPE_DOWNWELL_RADIANCE>INT16</DATA_TYPE_DOWNWELL_RADIANCE>
    <DATA_TYPE_ATMOSPHERIC_TRANSMITTANCE>INT16</DATA_TYPE_ATMOSPHERIC_TRANSMITTANCE>
    <DATA_TYPE_EMISSIVITY>INT16</DATA_TYPE_EMISSIVITY>
    <DATA_TYPE_EMISSIVITY_STDEV>INT16</DATA_TYPE_EMISSIVITY_STDEV>
    <DATA_TYPE_CLOUD_DISTANCE>INT16</DATA_TYPE_CLOUD_DISTANCE>
    <DATA_TYPE_ATMOSPHERIC_OPACITY>INT16</DATA_TYPE_ATMOSPHERIC_OPACITY>
    <DATA_TYPE_QUALITY_L2_SURFACE_REFLECTANCE_CLOUD>UINT8</DATA_TYPE_QUALITY_L2_SURFACE_REFLECTANCE_CLOUD>
    <DATA_TYPE_QUALITY_L2_SURFACE_TEMPERATURE>INT16</DATA_TYPE_QUALITY_L2_SURFACE_TEMPERATURE>
    <DATA_TYPE_QUALITY_L1_PIXEL>UINT16</DATA_TYPE_QUALITY_L1_PIXEL>
    <DATA_TYPE_QUALITY_L1_RADIOMETRIC_SATURATION>UINT16</DATA_TYPE_QUALITY_L1_RADIOMETRIC_SATURATION>
  </PRODUCT_CONTENTS>
  <IMAGE_ATTRIBUTES>
    <SPACECRAFT_ID>LANDSAT_7</SPACECRAFT_ID>
    <SENSOR_ID>ETM</SENSOR_ID>
    <WRS_TYPE>2</WRS_TYPE>
    <WRS_PATH>028</WRS_PATH>
    <WRS_ROW>030</WRS_ROW>
    <DATE_ACQUIRED>2023-08-17</DATE_ACQUIRED>
    <SCENE_CENTER_TIME>14:54:20.0400998Z</SCENE_CENTER_TIME>
    <STATION_ID>ASN</STATION_ID>
    <CLOUD_COVER>0.00</CLOUD_COVER>
    <CLOUD_COVER_LAND>0.00</CLOUD_COVER_LAND>
    <IMAGE_QUALITY>9</IMAGE_QUALITY>
    <SATURATION_BAND_1>N</SATURATION_BAND_1>
    <SATURATION_BAND_2>Y</SATURATION_BAND_2>
    <SATURATION_BAND_3>Y</SATURATION_BAND_3>
    <SATURATION_BAND_4>Y</SATURATION_BAND_4>
    <SATURATION_BAND_5>N</SATURATION_BAND_5>
    <SATURATION_BAND_6_VCID_1>N</SATURATION_BAND_6_VCID_1>
    <SATURATION_BAND_6_VCID_2>Y</SATURATION_BAND_6_VCID_2>
    <SATURATION_BAND_7>Y</SATURATION_BAND_7>
    <SATURATION_BAND_8>N</SATURATION_BAND_8>
    <SUN_AZIMUTH>105.82706424</SUN_AZIMUTH>
    <SUN_ELEVATION>35.09902700</SUN_ELEVATION>
    <EARTH_SUN_DISTANCE>1.0124651</EARTH_SUN_DISTANCE>
    <SENSOR_MODE>BUMPER</SENSOR_MODE>
    <SENSOR_MODE_SLC>OFF</SENSOR_MODE_SLC>
    <SENSOR_ANOMALIES>NONE</SENSOR_ANOMALIES>
  </IMAGE_ATTRIBUTES>
  <PROJECTION_ATTRIBUTES>
    <MAP_PROJECTION>UTM</MAP_PROJECTION>
    <DATUM>WGS84</DATUM>
    <ELLIPSOID>WGS84</ELLIPSOID>
    <UTM_ZONE>15</UTM_ZONE>
    <GRID_CELL_SIZE_REFLECTIVE>30.00</GRID_CELL_SIZE_REFLECTIVE>
    <GRID_CELL_SIZE_THERMAL>30.00</GRID_CELL_SIZE_THERMAL>
    <REFLECTIVE_LINES>100</REFLECTIVE_LINES>
    <REFLECTIVE_SAMPLES>100</REFLECTIVE_SAMPLES>
    <THERMAL_LINES>100</THERMAL_LINES>
    <THERMAL_SAMPLES>100</THERMAL_SAMPLES>
    <ORIENTATION>NORTH_UP</ORIENTATION>
    <CORNER_UL_LAT_PRODUCT>44.17250</CORNER_UL_LAT_PRODUCT>
    <CORNER_UL_LON_PRODUCT>-98.18108</CORNER_UL_LON_PRODUCT>
    <CORNER_UR_LAT_PRODUCT>44.27187</CORNER_UR_LAT_PRODUCT>
    <CORNER_UR_LON_PRODUCT>-95.04367</CORNER_UR_LON_PRODUCT>
    <CORNER_LL_LAT_PRODUCT>42.17058</CORNER_LL_LAT_PRODUCT>
    <CORNER_LL_LON_PRODUCT>-98.01413</CORNER_LL_LON_PRODUCT>
    <CORNER_LR_LAT_PRODUCT>42.26326</CORNER_LR_LAT_PRODUCT>
    <CORNER_LR_LON_PRODUCT>-94.97752</CORNER_LR_LON_PRODUCT>
    <CORNER_UL_PROJECTION_X_PRODUCT>85800.000</CORNER_UL_PROJECTION_X_PRODUCT>
    <CORNER_UL_PROJECTION_Y_PRODUCT>4904100.000</CORNER_UL_PROJECTION_Y_PRODUCT>
    <CORNER_UR_PROJECTION_X_PRODUCT>336900.000</CORNER_UR_PROJECTION_X_PRODUCT>
    <CORNER_UR_PROJECTION_Y_PRODUCT>4904100.000</CORNER_UR_PROJECTION_Y_PRODUCT>
    <CORNER_LL_PROJECTION_X_PRODUCT>85800.000</CORNER_LL_PROJECTION_X_PRODUCT>
    <CORNER_LL_PROJECTION_Y_PRODUCT>4680900.000</CORNER_LL_PROJECTION_Y_PRODUCT>
    <CORNER_LR_PROJECTION_X_PRODUCT>336900.000</CORNER_LR_PROJECTION_X_PRODUCT>
    <CORNER_LR_PROJECTION_Y_PRODUCT>4680900.000</CORNER_LR_PROJECTION_Y_PRODUCT>
  </PROJECTION_ATTRIBUTES>
  <LEVEL2_PROCESSING_RECORD>
    <ORIGIN>Image courtesy of the U.S. Geological Survey</ORIGIN>
    <DIGITAL_OBJECT_IDENTIFIER>https://doi.org/10.5066/P9C7I13B</DIGITAL_OBJECT_IDENTIFIER>
    <REQUEST_ID>1786896_00133</REQUEST_ID>
    <LANDSAT_PRODUCT_ID>LE07_L2SP_028030_20230817_20230912_02_T1</LANDSAT_PRODUCT_ID>
    <PROCESSING_LEVEL>L2SP</PROCESSING_LEVEL>
    <OUTPUT_FORMAT>GEOTIFF</OUTPUT_FORMAT>
    <DATE_PRODUCT_GENERATED>2023-09-12T21:14:59Z</DATE_PRODUCT_GENERATED>
    <PROCESSING_SOFTWARE_VERSION>LPGS_16.3.0</PROCESSING_SOFTWARE_VERSION>
    <ALGORITHM_SOURCE_SURFACE_REFLECTANCE>LEDAPS_3.4.0</ALGORITHM_SOURCE_SURFACE_REFLECTANCE>
    <DATA_SOURCE_OZONE>TOMS</DATA_SOURCE_OZONE>
    <DATA_SOURCE_PRESSURE>NCEP</DATA_SOURCE_PRESSURE>
    <DATA_SOURCE_WATER_VAPOR>NCEP</DATA_SOURCE_WATER_VAPOR>
    <DATA_SOURCE_AIR_TEMPERATURE>NCEP</DATA_SOURCE_AIR_TEMPERATURE>
    <ALGORITHM_SOURCE_SURFACE_TEMPERATURE>st_1.5.0</ALGORITHM_SOURCE_SURFACE_TEMPERATURE>
    <DATA_SOURCE_REANALYSIS>GEOS-5 FP-IT</DATA_SOURCE_REANALYSIS>
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
    <QUANTIZE_CAL_MAX_BAND_7>65535</QUANTIZE_CAL_MAX_BAND_7>
    <QUANTIZE_CAL_MIN_BAND_7>1</QUANTIZE_CAL_MIN_BAND_7>
    <REFLECTANCE_MULT_BAND_1>2.75e-05</REFLECTANCE_MULT_BAND_1>
    <REFLECTANCE_MULT_BAND_2>2.75e-05</REFLECTANCE_MULT_BAND_2>
    <REFLECTANCE_MULT_BAND_3>2.75e-05</REFLECTANCE_MULT_BAND_3>
    <REFLECTANCE_MULT_BAND_4>2.75e-05</REFLECTANCE_MULT_BAND_4>
    <REFLECTANCE_MULT_BAND_5>2.75e-05</REFLECTANCE_MULT_BAND_5>
    <REFLECTANCE_MULT_BAND_7>2.75e-05</REFLECTANCE_MULT_BAND_7>
    <REFLECTANCE_ADD_BAND_1>-0.2</REFLECTANCE_ADD_BAND_1>
    <REFLECTANCE_ADD_BAND_2>-0.2</REFLECTANCE_ADD_BAND_2>
    <REFLECTANCE_ADD_BAND_3>-0.2</REFLECTANCE_ADD_BAND_3>
    <REFLECTANCE_ADD_BAND_4>-0.2</REFLECTANCE_ADD_BAND_4>
    <REFLECTANCE_ADD_BAND_5>-0.2</REFLECTANCE_ADD_BAND_5>
    <REFLECTANCE_ADD_BAND_7>-0.2</REFLECTANCE_ADD_BAND_7>
  </LEVEL2_SURFACE_REFLECTANCE_PARAMETERS>
  <LEVEL2_SURFACE_TEMPERATURE_PARAMETERS>
    <TEMPERATURE_MAXIMUM_BAND_ST_B6>372.999941</TEMPERATURE_MAXIMUM_BAND_ST_B6>
    <TEMPERATURE_MINIMUM_BAND_ST_B6>149.003418</TEMPERATURE_MINIMUM_BAND_ST_B6>
    <QUANTIZE_CAL_MAXIMUM_BAND_ST_B6>65535</QUANTIZE_CAL_MAXIMUM_BAND_ST_B6>
    <QUANTIZE_CAL_MINIMUM_BAND_ST_B6>1</QUANTIZE_CAL_MINIMUM_BAND_ST_B6>
    <TEMPERATURE_MULT_BAND_ST_B6>0.00341802</TEMPERATURE_MULT_BAND_ST_B6>
    <TEMPERATURE_ADD_BAND_ST_B6>149.0</TEMPERATURE_ADD_BAND_ST_B6>
  </LEVEL2_SURFACE_TEMPERATURE_PARAMETERS>
  <LEVEL1_PROCESSING_RECORD>
    <ORIGIN>Image courtesy of the U.S. Geological Survey</ORIGIN>
    <DIGITAL_OBJECT_IDENTIFIER>https://doi.org/10.5066/P9TU80IG</DIGITAL_OBJECT_IDENTIFIER>
    <REQUEST_ID>1786896_00133</REQUEST_ID>
    <LANDSAT_SCENE_ID>LE70280302023229ASN00</LANDSAT_SCENE_ID>
    <LANDSAT_PRODUCT_ID>LE07_L1TP_028030_20230817_20230912_02_T1</LANDSAT_PRODUCT_ID>
    <PROCESSING_LEVEL>L1TP</PROCESSING_LEVEL>
    <COLLECTION_CATEGORY>T1</COLLECTION_CATEGORY>
    <OUTPUT_FORMAT>GEOTIFF</OUTPUT_FORMAT>
    <DATE_PRODUCT_GENERATED>2023-09-12T21:03:18Z</DATE_PRODUCT_GENERATED>
    <PROCESSING_SOFTWARE_VERSION>LPGS_16.3.0</PROCESSING_SOFTWARE_VERSION>
    <FILE_NAME_BAND_1>LE07_L1TP_028030_20230817_20230912_02_T1_B1.TIF</FILE_NAME_BAND_1>
    <FILE_NAME_BAND_2>LE07_L1TP_028030_20230817_20230912_02_T1_B2.TIF</FILE_NAME_BAND_2>
    <FILE_NAME_BAND_3>LE07_L1TP_028030_20230817_20230912_02_T1_B3.TIF</FILE_NAME_BAND_3>
    <FILE_NAME_BAND_4>LE07_L1TP_028030_20230817_20230912_02_T1_B4.TIF</FILE_NAME_BAND_4>
    <FILE_NAME_BAND_5>LE07_L1TP_028030_20230817_20230912_02_T1_B5.TIF</FILE_NAME_BAND_5>
    <FILE_NAME_BAND_6_VCID_1>LE07_L1TP_028030_20230817_20230912_02_T1_B6_VCID_1.TIF</FILE_NAME_BAND_6_VCID_1>
    <FILE_NAME_BAND_6_VCID_2>LE07_L1TP_028030_20230817_20230912_02_T1_B6_VCID_2.TIF</FILE_NAME_BAND_6_VCID_2>
    <FILE_NAME_BAND_7>LE07_L1TP_028030_20230817_20230912_02_T1_B7.TIF</FILE_NAME_BAND_7>
    <FILE_NAME_BAND_8>LE07_L1TP_028030_20230817_20230912_02_T1_B8.TIF</FILE_NAME_BAND_8>
    <FILE_NAME_QUALITY_L1_PIXEL>LE07_L1TP_028030_20230817_20230912_02_T1_QA_PIXEL.TIF</FILE_NAME_QUALITY_L1_PIXEL>
    <FILE_NAME_QUALITY_L1_RADIOMETRIC_SATURATION>LE07_L1TP_028030_20230817_20230912_02_T1_QA_RADSAT.TIF</FILE_NAME_QUALITY_L1_RADIOMETRIC_SATURATION>
    <FILE_NAME_GROUND_CONTROL_POINT>LE07_L1TP_028030_20230817_20230912_02_T1_GCP.txt</FILE_NAME_GROUND_CONTROL_POINT>
    <FILE_NAME_ANGLE_COEFFICIENT>LE07_L1TP_028030_20230817_20230912_02_T1_ANG.txt</FILE_NAME_ANGLE_COEFFICIENT>
    <FILE_NAME_ANGLE_SENSOR_AZIMUTH_BAND_4>LE07_L1TP_028030_20230817_20230912_02_T1_VAA.TIF</FILE_NAME_ANGLE_SENSOR_AZIMUTH_BAND_4>
    <FILE_NAME_ANGLE_SENSOR_ZENITH_BAND_4>LE07_L1TP_028030_20230817_20230912_02_T1_VZA.TIF</FILE_NAME_ANGLE_SENSOR_ZENITH_BAND_4>
    <FILE_NAME_ANGLE_SOLAR_AZIMUTH_BAND_4>LE07_L1TP_028030_20230817_20230912_02_T1_SAA.TIF</FILE_NAME_ANGLE_SOLAR_AZIMUTH_BAND_4>
    <FILE_NAME_ANGLE_SOLAR_ZENITH_BAND_4>LE07_L1TP_028030_20230817_20230912_02_T1_SZA.TIF</FILE_NAME_ANGLE_SOLAR_ZENITH_BAND_4>
    <FILE_NAME_METADATA_ODL>LE07_L1TP_028030_20230817_20230912_02_T1_MTL.txt</FILE_NAME_METADATA_ODL>
    <FILE_NAME_METADATA_XML>LE07_L1TP_028030_20230817_20230912_02_T1_MTL.xml</FILE_NAME_METADATA_XML>
    <FILE_NAME_CPF>LE07CPF_20230701_20230930_02.05</FILE_NAME_CPF>
    <DATA_SOURCE_ELEVATION>GLS2000</DATA_SOURCE_ELEVATION>
    <GROUND_CONTROL_POINTS_VERSION>5</GROUND_CONTROL_POINTS_VERSION>
    <GROUND_CONTROL_POINTS_MODEL>435</GROUND_CONTROL_POINTS_MODEL>
    <GEOMETRIC_RMSE_MODEL>5.332</GEOMETRIC_RMSE_MODEL>
    <GEOMETRIC_RMSE_MODEL_Y>4.119</GEOMETRIC_RMSE_MODEL_Y>
    <GEOMETRIC_RMSE_MODEL_X>3.386</GEOMETRIC_RMSE_MODEL_X>
    <EPHEMERIS_TYPE>DEFINITIVE</EPHEMERIS_TYPE>
  </LEVEL1_PROCESSING_RECORD>
  <LEVEL1_MIN_MAX_RADIANCE>
    <RADIANCE_MAXIMUM_BAND_1>191.600</RADIANCE_MAXIMUM_BAND_1>
    <RADIANCE_MINIMUM_BAND_1>-6.200</RADIANCE_MINIMUM_BAND_1>
    <RADIANCE_MAXIMUM_BAND_2>196.500</RADIANCE_MAXIMUM_BAND_2>
    <RADIANCE_MINIMUM_BAND_2>-6.400</RADIANCE_MINIMUM_BAND_2>
    <RADIANCE_MAXIMUM_BAND_3>152.900</RADIANCE_MAXIMUM_BAND_3>
    <RADIANCE_MINIMUM_BAND_3>-5.000</RADIANCE_MINIMUM_BAND_3>
    <RADIANCE_MAXIMUM_BAND_4>241.100</RADIANCE_MAXIMUM_BAND_4>
    <RADIANCE_MINIMUM_BAND_4>-5.100</RADIANCE_MINIMUM_BAND_4>
    <RADIANCE_MAXIMUM_BAND_5>31.060</RADIANCE_MAXIMUM_BAND_5>
    <RADIANCE_MINIMUM_BAND_5>-1.000</RADIANCE_MINIMUM_BAND_5>
    <RADIANCE_MAXIMUM_BAND_6_VCID_1>17.040</RADIANCE_MAXIMUM_BAND_6_VCID_1>
    <RADIANCE_MINIMUM_BAND_6_VCID_1>0.000</RADIANCE_MINIMUM_BAND_6_VCID_1>
    <RADIANCE_MAXIMUM_BAND_6_VCID_2>12.650</RADIANCE_MAXIMUM_BAND_6_VCID_2>
    <RADIANCE_MINIMUM_BAND_6_VCID_2>3.200</RADIANCE_MINIMUM_BAND_6_VCID_2>
    <RADIANCE_MAXIMUM_BAND_7>16.540</RADIANCE_MAXIMUM_BAND_7>
    <RADIANCE_MINIMUM_BAND_7>-0.350</RADIANCE_MINIMUM_BAND_7>
    <RADIANCE_MAXIMUM_BAND_8>243.100</RADIANCE_MAXIMUM_BAND_8>
    <RADIANCE_MINIMUM_BAND_8>-4.700</RADIANCE_MINIMUM_BAND_8>
  </LEVEL1_MIN_MAX_RADIANCE>
  <LEVEL1_MIN_MAX_REFLECTANCE>
    <REFLECTANCE_MAXIMUM_BAND_1>0.303059</REFLECTANCE_MAXIMUM_BAND_1>
    <REFLECTANCE_MINIMUM_BAND_1>-0.009807</REFLECTANCE_MINIMUM_BAND_1>
    <REFLECTANCE_MAXIMUM_BAND_2>0.340953</REFLECTANCE_MAXIMUM_BAND_2>
    <REFLECTANCE_MINIMUM_BAND_2>-0.011105</REFLECTANCE_MINIMUM_BAND_2>
    <REFLECTANCE_MAXIMUM_BAND_3>0.322885</REFLECTANCE_MAXIMUM_BAND_3>
    <REFLECTANCE_MINIMUM_BAND_3>-0.010559</REFLECTANCE_MINIMUM_BAND_3>
    <REFLECTANCE_MAXIMUM_BAND_4>0.724966</REFLECTANCE_MAXIMUM_BAND_4>
    <REFLECTANCE_MINIMUM_BAND_4>-0.015335</REFLECTANCE_MINIMUM_BAND_4>
    <REFLECTANCE_MAXIMUM_BAND_5>0.451379</REFLECTANCE_MAXIMUM_BAND_5>
    <REFLECTANCE_MINIMUM_BAND_5>-0.014532</REFLECTANCE_MINIMUM_BAND_5>
    <REFLECTANCE_MAXIMUM_BAND_7>0.654688</REFLECTANCE_MAXIMUM_BAND_7>
    <REFLECTANCE_MINIMUM_BAND_7>-0.013854</REFLECTANCE_MINIMUM_BAND_7>
    <REFLECTANCE_MAXIMUM_BAND_8>0.593540</REFLECTANCE_MAXIMUM_BAND_8>
    <REFLECTANCE_MINIMUM_BAND_8>-0.011475</REFLECTANCE_MINIMUM_BAND_8>
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
    <QUANTIZE_CAL_MAX_BAND_6_VCID_1>255</QUANTIZE_CAL_MAX_BAND_6_VCID_1>
    <QUANTIZE_CAL_MIN_BAND_6_VCID_1>1</QUANTIZE_CAL_MIN_BAND_6_VCID_1>
    <QUANTIZE_CAL_MAX_BAND_6_VCID_2>255</QUANTIZE_CAL_MAX_BAND_6_VCID_2>
    <QUANTIZE_CAL_MIN_BAND_6_VCID_2>1</QUANTIZE_CAL_MIN_BAND_6_VCID_2>
    <QUANTIZE_CAL_MAX_BAND_7>255</QUANTIZE_CAL_MAX_BAND_7>
    <QUANTIZE_CAL_MIN_BAND_7>1</QUANTIZE_CAL_MIN_BAND_7>
    <QUANTIZE_CAL_MAX_BAND_8>255</QUANTIZE_CAL_MAX_BAND_8>
    <QUANTIZE_CAL_MIN_BAND_8>1</QUANTIZE_CAL_MIN_BAND_8>
  </LEVEL1_MIN_MAX_PIXEL_VALUE>
  <LEVEL1_RADIOMETRIC_RESCALING>
    <RADIANCE_MULT_BAND_1>7.7874E-01</RADIANCE_MULT_BAND_1>
    <RADIANCE_MULT_BAND_2>7.9882E-01</RADIANCE_MULT_BAND_2>
    <RADIANCE_MULT_BAND_3>6.2165E-01</RADIANCE_MULT_BAND_3>
    <RADIANCE_MULT_BAND_4>9.6929E-01</RADIANCE_MULT_BAND_4>
    <RADIANCE_MULT_BAND_5>1.2622E-01</RADIANCE_MULT_BAND_5>
    <RADIANCE_MULT_BAND_6_VCID_1>6.7087E-02</RADIANCE_MULT_BAND_6_VCID_1>
    <RADIANCE_MULT_BAND_6_VCID_2>3.7205E-02</RADIANCE_MULT_BAND_6_VCID_2>
    <RADIANCE_MULT_BAND_7>6.6496E-02</RADIANCE_MULT_BAND_7>
    <RADIANCE_MULT_BAND_8>9.7559E-01</RADIANCE_MULT_BAND_8>
    <RADIANCE_ADD_BAND_1>-6.97874</RADIANCE_ADD_BAND_1>
    <RADIANCE_ADD_BAND_2>-7.19882</RADIANCE_ADD_BAND_2>
    <RADIANCE_ADD_BAND_3>-5.62165</RADIANCE_ADD_BAND_3>
    <RADIANCE_ADD_BAND_4>-6.06929</RADIANCE_ADD_BAND_4>
    <RADIANCE_ADD_BAND_5>-1.12622</RADIANCE_ADD_BAND_5>
    <RADIANCE_ADD_BAND_6_VCID_1>-0.06709</RADIANCE_ADD_BAND_6_VCID_1>
    <RADIANCE_ADD_BAND_6_VCID_2>3.16280</RADIANCE_ADD_BAND_6_VCID_2>
    <RADIANCE_ADD_BAND_7>-0.41650</RADIANCE_ADD_BAND_7>
    <RADIANCE_ADD_BAND_8>-5.67559</RADIANCE_ADD_BAND_8>
    <REFLECTANCE_MULT_BAND_1>1.2318E-03</REFLECTANCE_MULT_BAND_1>
    <REFLECTANCE_MULT_BAND_2>1.3861E-03</REFLECTANCE_MULT_BAND_2>
    <REFLECTANCE_MULT_BAND_3>1.3128E-03</REFLECTANCE_MULT_BAND_3>
    <REFLECTANCE_MULT_BAND_4>2.9146E-03</REFLECTANCE_MULT_BAND_4>
    <REFLECTANCE_MULT_BAND_5>1.8343E-03</REFLECTANCE_MULT_BAND_5>
    <REFLECTANCE_MULT_BAND_7>2.6321E-03</REFLECTANCE_MULT_BAND_7>
    <REFLECTANCE_MULT_BAND_8>2.3820E-03</REFLECTANCE_MULT_BAND_8>
    <REFLECTANCE_ADD_BAND_1>-0.011038</REFLECTANCE_ADD_BAND_1>
    <REFLECTANCE_ADD_BAND_2>-0.012491</REFLECTANCE_ADD_BAND_2>
    <REFLECTANCE_ADD_BAND_3>-0.011871</REFLECTANCE_ADD_BAND_3>
    <REFLECTANCE_ADD_BAND_4>-0.018250</REFLECTANCE_ADD_BAND_4>
    <REFLECTANCE_ADD_BAND_5>-0.016367</REFLECTANCE_ADD_BAND_5>
    <REFLECTANCE_ADD_BAND_7>-0.016486</REFLECTANCE_ADD_BAND_7>
    <REFLECTANCE_ADD_BAND_8>-0.013857</REFLECTANCE_ADD_BAND_8>
  </LEVEL1_RADIOMETRIC_RESCALING>
  <LEVEL1_THERMAL_CONSTANTS>
    <K1_CONSTANT_BAND_6_VCID_1>666.09</K1_CONSTANT_BAND_6_VCID_1>
    <K2_CONSTANT_BAND_6_VCID_1>1282.71</K2_CONSTANT_BAND_6_VCID_1>
    <K1_CONSTANT_BAND_6_VCID_2>666.09</K1_CONSTANT_BAND_6_VCID_2>
    <K2_CONSTANT_BAND_6_VCID_2>1282.71</K2_CONSTANT_BAND_6_VCID_2>
  </LEVEL1_THERMAL_CONSTANTS>
  <LEVEL1_PROJECTION_PARAMETERS>
    <MAP_PROJECTION>UTM</MAP_PROJECTION>
    <DATUM>WGS84</DATUM>
    <ELLIPSOID>WGS84</ELLIPSOID>
    <UTM_ZONE>15</UTM_ZONE>
    <GRID_CELL_SIZE_PANCHROMATIC>15.00</GRID_CELL_SIZE_PANCHROMATIC>
    <GRID_CELL_SIZE_REFLECTIVE>30.00</GRID_CELL_SIZE_REFLECTIVE>
    <GRID_CELL_SIZE_THERMAL>30.00</GRID_CELL_SIZE_THERMAL>
    <ORIENTATION>NORTH_UP</ORIENTATION>
    <RESAMPLING_OPTION>CUBIC_CONVOLUTION</RESAMPLING_OPTION>
    <SCAN_GAP_INTERPOLATION>2.0</SCAN_GAP_INTERPOLATION>
  </LEVEL1_PROJECTION_PARAMETERS>
  <PRODUCT_PARAMETERS>
    <CORRECTION_GAIN_BAND_1>CPF</CORRECTION_GAIN_BAND_1>
    <CORRECTION_GAIN_BAND_2>CPF</CORRECTION_GAIN_BAND_2>
    <CORRECTION_GAIN_BAND_3>CPF</CORRECTION_GAIN_BAND_3>
    <CORRECTION_GAIN_BAND_4>CPF</CORRECTION_GAIN_BAND_4>
    <CORRECTION_GAIN_BAND_5>CPF</CORRECTION_GAIN_BAND_5>
    <CORRECTION_GAIN_BAND_6_VCID_1>CPF</CORRECTION_GAIN_BAND_6_VCID_1>
    <CORRECTION_GAIN_BAND_6_VCID_2>CPF</CORRECTION_GAIN_BAND_6_VCID_2>
    <CORRECTION_GAIN_BAND_7>CPF</CORRECTION_GAIN_BAND_7>
    <CORRECTION_GAIN_BAND_8>CPF</CORRECTION_GAIN_BAND_8>
    <CORRECTION_BIAS_BAND_1>INTERNAL_CALIBRATION</CORRECTION_BIAS_BAND_1>
    <CORRECTION_BIAS_BAND_2>INTERNAL_CALIBRATION</CORRECTION_BIAS_BAND_2>
    <CORRECTION_BIAS_BAND_3>INTERNAL_CALIBRATION</CORRECTION_BIAS_BAND_3>
    <CORRECTION_BIAS_BAND_4>INTERNAL_CALIBRATION</CORRECTION_BIAS_BAND_4>
    <CORRECTION_BIAS_BAND_5>INTERNAL_CALIBRATION</CORRECTION_BIAS_BAND_5>
    <CORRECTION_BIAS_BAND_6_VCID_1>INTERNAL_CALIBRATION</CORRECTION_BIAS_BAND_6_VCID_1>
    <CORRECTION_BIAS_BAND_6_VCID_2>INTERNAL_CALIBRATION</CORRECTION_BIAS_BAND_6_VCID_2>
    <CORRECTION_BIAS_BAND_7>INTERNAL_CALIBRATION</CORRECTION_BIAS_BAND_7>
    <CORRECTION_BIAS_BAND_8>INTERNAL_CALIBRATION</CORRECTION_BIAS_BAND_8>
    <GAIN_BAND_1>H</GAIN_BAND_1>
    <GAIN_BAND_2>H</GAIN_BAND_2>
    <GAIN_BAND_3>H</GAIN_BAND_3>
    <GAIN_BAND_4>L</GAIN_BAND_4>
    <GAIN_BAND_5>H</GAIN_BAND_5>
    <GAIN_BAND_6_VCID_1>L</GAIN_BAND_6_VCID_1>
    <GAIN_BAND_6_VCID_2>H</GAIN_BAND_6_VCID_2>
    <GAIN_BAND_7>L</GAIN_BAND_7>
    <GAIN_BAND_8>L</GAIN_BAND_8>
    <GAIN_CHANGE_BAND_1>HH</GAIN_CHANGE_BAND_1>
    <GAIN_CHANGE_BAND_2>HH</GAIN_CHANGE_BAND_2>
    <GAIN_CHANGE_BAND_3>HH</GAIN_CHANGE_BAND_3>
    <GAIN_CHANGE_BAND_4>LL</GAIN_CHANGE_BAND_4>
    <GAIN_CHANGE_BAND_5>HH</GAIN_CHANGE_BAND_5>
    <GAIN_CHANGE_BAND_6_VCID_1>LL</GAIN_CHANGE_BAND_6_VCID_1>
    <GAIN_CHANGE_BAND_6_VCID_2>HH</GAIN_CHANGE_BAND_6_VCID_2>
    <GAIN_CHANGE_BAND_7>LL</GAIN_CHANGE_BAND_7>
    <GAIN_CHANGE_BAND_8>LL</GAIN_CHANGE_BAND_8>
    <GAIN_CHANGE_SCAN_BAND_1>0</GAIN_CHANGE_SCAN_BAND_1>
    <GAIN_CHANGE_SCAN_BAND_2>0</GAIN_CHANGE_SCAN_BAND_2>
    <GAIN_CHANGE_SCAN_BAND_3>0</GAIN_CHANGE_SCAN_BAND_3>
    <GAIN_CHANGE_SCAN_BAND_4>0</GAIN_CHANGE_SCAN_BAND_4>
    <GAIN_CHANGE_SCAN_BAND_5>0</GAIN_CHANGE_SCAN_BAND_5>
    <GAIN_CHANGE_SCAN_BAND_6_VCID_1>0</GAIN_CHANGE_SCAN_BAND_6_VCID_1>
    <GAIN_CHANGE_SCAN_BAND_6_VCID_2>0</GAIN_CHANGE_SCAN_BAND_6_VCID_2>
    <GAIN_CHANGE_SCAN_BAND_7>0</GAIN_CHANGE_SCAN_BAND_7>
    <GAIN_CHANGE_SCAN_BAND_8>0</GAIN_CHANGE_SCAN_BAND_8>
  </PRODUCT_PARAMETERS>
</LANDSAT_METADATA_FILE>
"""


x_size = 100
y_size = 100
date = datetime(2023, 8, 17, tzinfo=timezone.utc)


@pytest.fixture(scope="session")
def area():
    """Get the landsat 1 area def."""
    pcs_id = "WGS84 / UTM zone 15N"
    proj4_dict = {"proj": "utm", "zone": 15, "datum": "WGS84", "units": "m", "no_defs": None, "type": "crs"}
    area_extent = (85785.0, 4680885.0, 336915.0, 4904115.0)
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
    return tmp_path_factory.mktemp("etm_l2_files")


@pytest.fixture(scope="session")
def b4_file(files_path, b4_data, area):
    """Create the file for the b4 channel."""
    data = b4_data
    filename = files_path / "LE07_L2SP_028030_20230817_20230912_02_T1_SR_B4.TIF"
    name = "B4"
    create_tif_file(data, name, area, filename)
    return os.fspath(filename)

@pytest.fixture(scope="session")
def b6_file(files_path, b6_data, area):
    """Create the file for the b6 channel."""
    data = b6_data
    filename = files_path / "LE07_L2SP_028030_20230817_20230912_02_T1_ST_B6.TIF"
    name = "B6"
    create_tif_file(data, name, area, filename)
    return os.fspath(filename)

@pytest.fixture(scope="session")
def rad_file(files_path, rad_data, area):
    """Create the file for the sza."""
    data = rad_data
    filename = files_path / "LE07_L2SP_028030_20230817_20230912_02_T1_ST_TRAD.TIF"
    name = "TRAD"
    create_tif_file(data, name, area, filename)
    return os.fspath(filename)


@pytest.fixture(scope="session")
def mda_file(files_path):
    """Create the metadata xml file."""
    filename = files_path / "LE07_L2SP_028030_20230817_20230912_02_T1_MTL.xml"
    with open(filename, "wb") as f:
        f.write(metadata_text)
    return os.fspath(filename)


@pytest.fixture(scope="session")
def all_files(b4_file, b6_file, mda_file, rad_file):
    """Return all the files."""
    return b4_file, b6_file, mda_file, rad_file


@pytest.fixture(scope="session")
def all_fsspec_files(b4_file, b6_file, mda_file, rad_file):
    """Return all the files as FSFile objects."""
    from fsspec.implementations.local import LocalFileSystem

    from satpy.readers.core.remote import FSFile

    fs = LocalFileSystem()
    b4_file, b6_file, mda_file, rad_file = (
        FSFile(os.path.abspath(file), fs=fs)
        for file in [b4_file, b6_file, mda_file, rad_file]
    )
    return b4_file, b6_file, mda_file, rad_file


class TestETML2:
    """Test generic image reader."""

    def setup_method(self, tmp_path):
        """Set up the filename and filetype info dicts.."""
        self.filename_info = dict(observation_date=datetime(2023, 8, 17),
                                  platform_type="L",
                                  process_level_correction="L2SP",
                                  spacecraft_id="07",
                                  data_type="E",
                                  collection_id="02")
        self.ftype_info = {"file_type": "granule_B4"}

    def test_basicload(self, area, b4_file, b6_file, mda_file):
        """Test loading a Landsat Scene."""
        scn = Scene(reader="etm_l2_tif", filenames=[b4_file,
                                                    b6_file,
                                                    mda_file])
        scn.load(["B4", "B6"])

        # Check dataset is loaded correctly
        assert scn["B4"].shape == (100, 100)
        assert scn["B4"].attrs["area"] == area
        assert scn["B4"].attrs["saturated"]
        assert scn["B6"].shape == (100, 100)
        assert scn["B6"].attrs["area"] == area
        with pytest.raises(KeyError, match="saturated"):
            assert not scn["B6"].attrs["saturated"]

    def test_ch_startend(self, b4_file, mda_file):
        """Test correct retrieval of start/end times."""
        scn = Scene(reader="etm_l2_tif", filenames=[b4_file, mda_file])
        bnds = scn.available_dataset_names()
        assert bnds == ["B4"]

        scn.load(["B4"])
        assert scn.start_time == datetime(2023, 8, 17, 14, 54, 20, tzinfo=timezone.utc)
        assert scn.end_time == datetime(2023, 8, 17, 14, 54, 20, tzinfo=timezone.utc)

    def test_loading_gd(self, mda_file, b4_file):
        """Test loading a Landsat Scene with good channel requests."""
        from satpy.readers.landsat_base import ETML2CHReader, ETML2MDReader
        good_mda = ETML2MDReader(mda_file, self.filename_info, {})
        rdr = ETML2CHReader(b4_file, self.filename_info, self.ftype_info, good_mda)

        # Check case with good file data and load request
        rdr.get_dataset({"name": "B4", "calibration": "counts"}, {"standard_name": "test_data", "units": "test_units"})

    def test_loading_badfil(self, mda_file, b4_file):
        """Test loading a Landsat Scene with bad channel requests."""
        from satpy.readers.landsat_base import ETML2CHReader, ETML2MDReader
        good_mda = ETML2MDReader(mda_file, self.filename_info, {})
        rdr = ETML2CHReader(b4_file, self.filename_info, self.ftype_info, good_mda)

        ftype = {"standard_name": "test_data", "units": "test_units"}
        # Check case with request to load channel not matching filename
        with pytest.raises(ValueError, match="Requested channel B5 does not match the reader channel B4"):
            rdr.get_dataset({"name": "B5", "calibration": "counts"}, ftype)

    def test_badfiles(self, mda_file, b4_file):
        """Test loading a Landsat Scene with bad data."""
        from satpy.readers.landsat_base import ETML2CHReader, ETML2MDReader
        bad_fname_info = self.filename_info.copy()
        bad_fname_info["platform_type"] = "B"

        ftype = {"standard_name": "test_data", "units": "test_units"}

        # Test that metadata reader initialises with correct filename
        good_mda = ETML2MDReader(mda_file, self.filename_info, ftype)

        # Check metadata reader fails if platform type is wrong
        with pytest.raises(ValueError, match="This reader only supports Landsat data"):
            ETML2MDReader(mda_file, bad_fname_info, ftype)

        # Test that metadata reader initialises with correct filename
        ETML2CHReader(b4_file, self.filename_info, self.ftype_info, good_mda)

        # Check metadata reader fails if platform type is wrong
        with pytest.raises(ValueError, match="This reader only supports Landsat data"):
            ETML2CHReader(b4_file, bad_fname_info, self.ftype_info, good_mda)
        bad_ftype_info = self.ftype_info.copy()
        bad_ftype_info["file_type"] = "granule-b05"
        with pytest.raises(ValueError, match="Invalid file type: granule-b05"):
            ETML2CHReader(b4_file, self.filename_info, bad_ftype_info, good_mda)

    def test_calibration_counts(self, all_files, b4_data, b6_data, rad_data):
        """Test counts calibration mode for the reader."""
        from satpy import Scene

        scn = Scene(reader="etm_l2_tif", filenames=all_files)
        scn.load(["B4", "B6", "TRAD"], calibration="counts")
        np.testing.assert_allclose(scn["B4"].values, b4_data)
        np.testing.assert_allclose(scn["B6"].values, b6_data)
        np.testing.assert_allclose(scn["TRAD"].values, rad_data)
        assert scn["B4"].attrs["units"] == "1"
        assert scn["B6"].attrs["units"] == "1"
        assert scn["TRAD"].attrs["units"] == "1"
        assert scn["B4"].attrs["standard_name"] == "counts"
        assert scn["B6"].attrs["standard_name"] == "counts"
        assert scn["TRAD"].attrs["standard_name"] == "counts"

    def test_calibration_highlevel(self, all_files, b4_data, b6_data, rad_data):
        """Test high level calibration modes for the reader."""
        from satpy import Scene
        exp_b04 = (b4_data * 2.75e-05 - 0.2).astype(np.float32) * 100
        exp_b6 = (b6_data * 0.00341802 + 149.0).astype(np.float32)
        exp_rad = (rad_data * 0.001).astype(np.float32)
        scn = Scene(reader="etm_l2_tif", filenames=all_files)
        scn.load(["B4", "B6", "TRAD"])

        assert scn["B4"].attrs["units"] == "%"
        assert scn["B6"].attrs["units"] == "K"
        assert scn["TRAD"].attrs["units"] == "W m-2 um-1 sr-1"
        assert scn["B4"].attrs["standard_name"] == "surface_bidirectional_reflectance"
        assert scn["B6"].attrs["standard_name"] == "brightness_temperature"
        assert scn["TRAD"].attrs["standard_name"] == "toa_outgoing_radiance_per_unit_wavelength"
        np.testing.assert_allclose(np.array(scn["B4"].values), np.array(exp_b04), rtol=1e-4)
        np.testing.assert_allclose(scn["B6"].values, exp_b6, rtol=1e-6)
        np.testing.assert_allclose(scn["TRAD"].values, exp_rad, rtol=1e-6)

    def test_metadata(self, mda_file):
        """Check that metadata values loaded correctly."""
        from satpy.readers.landsat_base import ETML2MDReader
        mda = ETML2MDReader(mda_file, self.filename_info, {})

        cal_test_dict = {"B1": (2.75e-05, -0.2),
                         "B5": (2.75e-05, -0.2),
                         "B6": (0.00341802, 149.0)}

        assert mda.platform_name == "Landsat-7"
        assert mda.earth_sun_distance() == 1.0124651
        assert mda.band_calibration["B1"] == cal_test_dict["B1"]
        assert mda.band_calibration["B5"] == cal_test_dict["B5"]
        assert mda.band_calibration["B6"] == cal_test_dict["B6"]
        assert not mda.band_saturation["B1"]
        assert mda.band_saturation["B4"]
        assert not mda.band_saturation["B5"]
        with pytest.raises(KeyError):
            mda.band_saturation["B6"]

    def test_area_def(self, mda_file):
        """Check we can get the area defs properly."""
        from satpy.readers.landsat_base import ETML2MDReader
        mda = ETML2MDReader(mda_file, self.filename_info, {})

        standard_area = mda.build_area_def("B1")

        assert standard_area.area_extent == (85785.0, 4680885.0, 336915.0, 4904115.0)

    def test_basicload_remote(self, area, all_fsspec_files):
        """Test loading a Landsat Scene from a fsspec filesystem."""
        scn = Scene(reader="etm_l2_tif", filenames=all_fsspec_files)
        scn.load(["B4", "B6"])

        # Check dataset is loaded correctly
        assert scn["B4"].shape == (100, 100)
        assert scn["B4"].attrs["area"] == area
        assert scn["B4"].attrs["saturated"]
        assert scn["B6"].shape == (100, 100)
        assert scn["B6"].attrs["area"] == area
        with pytest.raises(KeyError, match="saturated"):
            assert not scn["B6"].attrs["saturated"]
