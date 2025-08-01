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

l1_metadata_text = b"""<?xml version="1.0" encoding="UTF-8"?>
<LANDSAT_METADATA_FILE>
  <PRODUCT_CONTENTS>
    <ORIGIN>Image courtesy of the U.S. Geological Survey</ORIGIN>
    <DIGITAL_OBJECT_IDENTIFIER>https://doi.org/10.5066/P9AF14YV</DIGITAL_OBJECT_IDENTIFIER>
    <LANDSAT_PRODUCT_ID>LM01_L1TP_032030_19720729_20200909_02_T2</LANDSAT_PRODUCT_ID>
    <PROCESSING_LEVEL>L1TP</PROCESSING_LEVEL>
    <COLLECTION_NUMBER>02</COLLECTION_NUMBER>
    <COLLECTION_CATEGORY>T2</COLLECTION_CATEGORY>
    <OUTPUT_FORMAT>GEOTIFF</OUTPUT_FORMAT>
    <FILE_NAME_BAND_4>LM01_L1TP_032030_19720729_20200909_02_T2_B4.TIF</FILE_NAME_BAND_4>
    <FILE_NAME_BAND_5>LM01_L1TP_032030_19720729_20200909_02_T2_B5.TIF</FILE_NAME_BAND_5>
    <FILE_NAME_BAND_6>LM01_L1TP_032030_19720729_20200909_02_T2_B6.TIF</FILE_NAME_BAND_6>
    <FILE_NAME_BAND_7>LM01_L1TP_032030_19720729_20200909_02_T2_B7.TIF</FILE_NAME_BAND_7>
    <FILE_NAME_QUALITY_L1_PIXEL>LM01_L1TP_032030_19720729_20200909_02_T2_QA_PIXEL.TIF</FILE_NAME_QUALITY_L1_PIXEL>
    <FILE_NAME_QUALITY_L1_RADIOMETRIC_SATURATION>LM01_L1TP_032030_19720729_20200909_02_T2_QA_RADSAT.TIF</FILE_NAME_QUALITY_L1_RADIOMETRIC_SATURATION>
    <FILE_NAME_GROUND_CONTROL_POINT>LM01_L1TP_032030_19720729_20200909_02_T2_GCP.txt</FILE_NAME_GROUND_CONTROL_POINT>
    <FILE_NAME_METADATA_ODL>LM01_L1TP_032030_19720729_20200909_02_T2_MTL.txt</FILE_NAME_METADATA_ODL>
    <FILE_NAME_METADATA_XML>LM01_L1TP_032030_19720729_20200909_02_T2_MTL.xml</FILE_NAME_METADATA_XML>
    <FILE_NAME_VERIFY_REPORT>LM01_L1TP_032030_19720729_20200909_02_T2_VER.txt</FILE_NAME_VERIFY_REPORT>
    <FILE_NAME_VERIFY_BROWSE>LM01_L1TP_032030_19720729_20200909_02_T2_VER.jpg</FILE_NAME_VERIFY_BROWSE>
    <DATA_TYPE_BAND_4>UINT8</DATA_TYPE_BAND_4>
    <DATA_TYPE_BAND_5>UINT8</DATA_TYPE_BAND_5>
    <DATA_TYPE_BAND_6>UINT8</DATA_TYPE_BAND_6>
    <DATA_TYPE_BAND_7>UINT8</DATA_TYPE_BAND_7>
    <DATA_TYPE_QUALITY_L1_PIXEL>UINT16</DATA_TYPE_QUALITY_L1_PIXEL>
    <DATA_TYPE_QUALITY_L1_RADIOMETRIC_SATURATION>UINT16</DATA_TYPE_QUALITY_L1_RADIOMETRIC_SATURATION>
    <PRESENT_BAND_4>Y</PRESENT_BAND_4>
    <PRESENT_BAND_5>Y</PRESENT_BAND_5>
    <PRESENT_BAND_6>Y</PRESENT_BAND_6>
    <PRESENT_BAND_7>Y</PRESENT_BAND_7>
  </PRODUCT_CONTENTS>
  <IMAGE_ATTRIBUTES>
    <SPACECRAFT_ID>LANDSAT_1</SPACECRAFT_ID>
    <SENSOR_ID>MSS</SENSOR_ID>
    <WRS_TYPE>1</WRS_TYPE>
    <WRS_PATH>032</WRS_PATH>
    <WRS_ROW>030</WRS_ROW>
    <DATE_ACQUIRED>1972-07-29</DATE_ACQUIRED>
    <SCENE_CENTER_TIME>16:49:31.7330000Z</SCENE_CENTER_TIME>
    <STATION_ID>AAA</STATION_ID>
    <CLOUD_COVER>0.00</CLOUD_COVER>
    <CLOUD_COVER_LAND>0.00</CLOUD_COVER_LAND>
    <IMAGE_QUALITY>-1</IMAGE_QUALITY>
    <SATURATION_BAND_4>N</SATURATION_BAND_4>
    <SATURATION_BAND_5>Y</SATURATION_BAND_5>
    <SATURATION_BAND_6>Y</SATURATION_BAND_6>
    <SATURATION_BAND_7>N</SATURATION_BAND_7>
    <SUN_AZIMUTH>128.00842031</SUN_AZIMUTH>
    <SUN_ELEVATION>56.29977419</SUN_ELEVATION>
    <EARTH_SUN_DISTANCE>1.0152109</EARTH_SUN_DISTANCE>
  </IMAGE_ATTRIBUTES>
  <PROJECTION_ATTRIBUTES>
    <MAP_PROJECTION>UTM</MAP_PROJECTION>
    <DATUM>WGS84</DATUM>
    <ELLIPSOID>WGS84</ELLIPSOID>
    <UTM_ZONE>14</UTM_ZONE>
    <GRID_CELL_SIZE_REFLECTIVE>60.00</GRID_CELL_SIZE_REFLECTIVE>
    <REFLECTIVE_LINES>100</REFLECTIVE_LINES>
    <REFLECTIVE_SAMPLES>100</REFLECTIVE_SAMPLES>
    <ORIENTATION>NORTH_UP</ORIENTATION>
    <CORNER_UL_LAT_PRODUCT>44.14255</CORNER_UL_LAT_PRODUCT>
    <CORNER_UL_LON_PRODUCT>-99.72341</CORNER_UL_LON_PRODUCT>
    <CORNER_UR_LAT_PRODUCT>44.12425</CORNER_UR_LAT_PRODUCT>
    <CORNER_UR_LON_PRODUCT>-96.83145</CORNER_UR_LON_PRODUCT>
    <CORNER_LL_LAT_PRODUCT>42.15821</CORNER_LL_LAT_PRODUCT>
    <CORNER_LL_LON_PRODUCT>-99.70038</CORNER_LL_LON_PRODUCT>
    <CORNER_LR_LAT_PRODUCT>42.14114</CORNER_LR_LAT_PRODUCT>
    <CORNER_LR_LON_PRODUCT>-96.90044</CORNER_LR_LON_PRODUCT>
    <CORNER_UL_PROJECTION_X_PRODUCT>442140.000</CORNER_UL_PROJECTION_X_PRODUCT>
    <CORNER_UL_PROJECTION_Y_PRODUCT>4887960.000</CORNER_UL_PROJECTION_Y_PRODUCT>
    <CORNER_UR_PROJECTION_X_PRODUCT>673500.000</CORNER_UR_PROJECTION_X_PRODUCT>
    <CORNER_UR_PROJECTION_Y_PRODUCT>4887960.000</CORNER_UR_PROJECTION_Y_PRODUCT>
    <CORNER_LL_PROJECTION_X_PRODUCT>442140.000</CORNER_LL_PROJECTION_X_PRODUCT>
    <CORNER_LL_PROJECTION_Y_PRODUCT>4667580.000</CORNER_LL_PROJECTION_Y_PRODUCT>
    <CORNER_LR_PROJECTION_X_PRODUCT>673500.000</CORNER_LR_PROJECTION_X_PRODUCT>
    <CORNER_LR_PROJECTION_Y_PRODUCT>4667580.000</CORNER_LR_PROJECTION_Y_PRODUCT>
  </PROJECTION_ATTRIBUTES>
  <LEVEL1_PROCESSING_RECORD>
    <ORIGIN>Image courtesy of the U.S. Geological Survey</ORIGIN>
    <DIGITAL_OBJECT_IDENTIFIER>https://doi.org/10.5066/P9AF14YV</DIGITAL_OBJECT_IDENTIFIER>
    <REQUEST_ID>L2</REQUEST_ID>
    <LANDSAT_SCENE_ID>LM10320301972211AAA04</LANDSAT_SCENE_ID>
    <LANDSAT_PRODUCT_ID>LM01_L1TP_032030_19720729_20200909_02_T2</LANDSAT_PRODUCT_ID>
    <PROCESSING_LEVEL>L1TP</PROCESSING_LEVEL>
    <COLLECTION_CATEGORY>T2</COLLECTION_CATEGORY>
    <OUTPUT_FORMAT>GEOTIFF</OUTPUT_FORMAT>
    <DATE_PRODUCT_GENERATED>2020-09-09T16:01:31Z</DATE_PRODUCT_GENERATED>
    <PROCESSING_SOFTWARE_VERSION>LPGS_15.3.1c</PROCESSING_SOFTWARE_VERSION>
    <FILE_NAME_BAND_4>LM01_L1TP_032030_19720729_20200909_02_T2_B4.TIF</FILE_NAME_BAND_4>
    <FILE_NAME_BAND_5>LM01_L1TP_032030_19720729_20200909_02_T2_B5.TIF</FILE_NAME_BAND_5>
    <FILE_NAME_BAND_6>LM01_L1TP_032030_19720729_20200909_02_T2_B6.TIF</FILE_NAME_BAND_6>
    <FILE_NAME_BAND_7>LM01_L1TP_032030_19720729_20200909_02_T2_B7.TIF</FILE_NAME_BAND_7>
    <FILE_NAME_QUALITY_L1_PIXEL>LM01_L1TP_032030_19720729_20200909_02_T2_QA_PIXEL.TIF</FILE_NAME_QUALITY_L1_PIXEL>
    <FILE_NAME_QUALITY_L1_RADIOMETRIC_SATURATION>LM01_L1TP_032030_19720729_20200909_02_T2_QA_RADSAT.TIF</FILE_NAME_QUALITY_L1_RADIOMETRIC_SATURATION>
    <FILE_NAME_GROUND_CONTROL_POINT>LM01_L1TP_032030_19720729_20200909_02_T2_GCP.txt</FILE_NAME_GROUND_CONTROL_POINT>
    <FILE_NAME_METADATA_ODL>LM01_L1TP_032030_19720729_20200909_02_T2_MTL.txt</FILE_NAME_METADATA_ODL>
    <FILE_NAME_METADATA_XML>LM01_L1TP_032030_19720729_20200909_02_T2_MTL.xml</FILE_NAME_METADATA_XML>
    <FILE_NAME_CPF>LM01CPF_19720723_19780107_02.01</FILE_NAME_CPF>
    <FILE_NAME_VERIFY_REPORT>LM01_L1TP_032030_19720729_20200909_02_T2_VER.txt</FILE_NAME_VERIFY_REPORT>
    <FILE_NAME_VERIFY_BROWSE>LM01_L1TP_032030_19720729_20200909_02_T2_VER.jpg</FILE_NAME_VERIFY_BROWSE>
    <DATA_SOURCE_ELEVATION>GLS2000</DATA_SOURCE_ELEVATION>
    <GROUND_CONTROL_POINTS_VERSION>5</GROUND_CONTROL_POINTS_VERSION>
    <GROUND_CONTROL_POINTS_MODEL>26</GROUND_CONTROL_POINTS_MODEL>
    <GEOMETRIC_RMSE_MODEL>80.835</GEOMETRIC_RMSE_MODEL>
    <GEOMETRIC_RMSE_MODEL_Y>37.450</GEOMETRIC_RMSE_MODEL_Y>
    <GEOMETRIC_RMSE_MODEL_X>71.636</GEOMETRIC_RMSE_MODEL_X>
    <GROUND_CONTROL_POINTS_VERIFY>53</GROUND_CONTROL_POINTS_VERIFY>
    <GEOMETRIC_RMSE_VERIFY>0.944</GEOMETRIC_RMSE_VERIFY>
    <GEOMETRIC_RMSE_VERIFY_QUAD_UL>0.770</GEOMETRIC_RMSE_VERIFY_QUAD_UL>
    <GEOMETRIC_RMSE_VERIFY_QUAD_UR>1.495</GEOMETRIC_RMSE_VERIFY_QUAD_UR>
    <GEOMETRIC_RMSE_VERIFY_QUAD_LL>0.842</GEOMETRIC_RMSE_VERIFY_QUAD_LL>
    <GEOMETRIC_RMSE_VERIFY_QUAD_LR>1.049</GEOMETRIC_RMSE_VERIFY_QUAD_LR>
    <EPHEMERIS_TYPE>PREDICTIVE</EPHEMERIS_TYPE>
  </LEVEL1_PROCESSING_RECORD>
  <LEVEL1_MIN_MAX_RADIANCE>
    <RADIANCE_MAXIMUM_BAND_4>225.200</RADIANCE_MAXIMUM_BAND_4>
    <RADIANCE_MINIMUM_BAND_4>-17.600</RADIANCE_MINIMUM_BAND_4>
    <RADIANCE_MAXIMUM_BAND_5>164.600</RADIANCE_MAXIMUM_BAND_5>
    <RADIANCE_MINIMUM_BAND_5>-0.100</RADIANCE_MINIMUM_BAND_5>
    <RADIANCE_MAXIMUM_BAND_6>165.600</RADIANCE_MAXIMUM_BAND_6>
    <RADIANCE_MINIMUM_BAND_6>-0.100</RADIANCE_MINIMUM_BAND_6>
    <RADIANCE_MAXIMUM_BAND_7>154.600</RADIANCE_MAXIMUM_BAND_7>
    <RADIANCE_MINIMUM_BAND_7>0.000</RADIANCE_MINIMUM_BAND_7>
  </LEVEL1_MIN_MAX_RADIANCE>
  <LEVEL1_MIN_MAX_REFLECTANCE>
    <REFLECTANCE_MAXIMUM_BAND_4>0.407132</REFLECTANCE_MAXIMUM_BAND_4>
    <REFLECTANCE_MINIMUM_BAND_4>-0.031818</REFLECTANCE_MINIMUM_BAND_4>
    <REFLECTANCE_MAXIMUM_BAND_5>0.346752</REFLECTANCE_MAXIMUM_BAND_5>
    <REFLECTANCE_MINIMUM_BAND_5>-0.000211</REFLECTANCE_MINIMUM_BAND_5>
    <REFLECTANCE_MAXIMUM_BAND_6>0.420875</REFLECTANCE_MAXIMUM_BAND_6>
    <REFLECTANCE_MINIMUM_BAND_6>-0.000254</REFLECTANCE_MINIMUM_BAND_6>
    <REFLECTANCE_MAXIMUM_BAND_7>0.591490</REFLECTANCE_MAXIMUM_BAND_7>
    <REFLECTANCE_MINIMUM_BAND_7>0.000000</REFLECTANCE_MINIMUM_BAND_7>
  </LEVEL1_MIN_MAX_REFLECTANCE>
  <LEVEL1_MIN_MAX_PIXEL_VALUE>
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
    <RADIANCE_MULT_BAND_4>9.5591E-01</RADIANCE_MULT_BAND_4>
    <RADIANCE_MULT_BAND_5>6.4843E-01</RADIANCE_MULT_BAND_5>
    <RADIANCE_MULT_BAND_6>6.5236E-01</RADIANCE_MULT_BAND_6>
    <RADIANCE_MULT_BAND_7>6.0866E-01</RADIANCE_MULT_BAND_7>
    <RADIANCE_ADD_BAND_4>-18.55591</RADIANCE_ADD_BAND_4>
    <RADIANCE_ADD_BAND_5>-0.74843</RADIANCE_ADD_BAND_5>
    <RADIANCE_ADD_BAND_6>-0.75236</RADIANCE_ADD_BAND_6>
    <RADIANCE_ADD_BAND_7>-0.60866</RADIANCE_ADD_BAND_7>
    <REFLECTANCE_MULT_BAND_4>1.7282E-03</REFLECTANCE_MULT_BAND_4>
    <REFLECTANCE_MULT_BAND_5>1.3660E-03</REFLECTANCE_MULT_BAND_5>
    <REFLECTANCE_MULT_BAND_6>1.6580E-03</REFLECTANCE_MULT_BAND_6>
    <REFLECTANCE_MULT_BAND_7>2.3287E-03</REFLECTANCE_MULT_BAND_7>
    <REFLECTANCE_ADD_BAND_4>-0.033547</REFLECTANCE_ADD_BAND_4>
    <REFLECTANCE_ADD_BAND_5>-0.001577</REFLECTANCE_ADD_BAND_5>
    <REFLECTANCE_ADD_BAND_6>-0.001912</REFLECTANCE_ADD_BAND_6>
    <REFLECTANCE_ADD_BAND_7>-0.002329</REFLECTANCE_ADD_BAND_7>
  </LEVEL1_RADIOMETRIC_RESCALING>
  <LEVEL1_PROJECTION_PARAMETERS>
    <MAP_PROJECTION>UTM</MAP_PROJECTION>
    <DATUM>WGS84</DATUM>
    <ELLIPSOID>WGS84</ELLIPSOID>
    <UTM_ZONE>14</UTM_ZONE>
    <GRID_CELL_SIZE_REFLECTIVE>60.00</GRID_CELL_SIZE_REFLECTIVE>
    <ORIENTATION>NORTH_UP</ORIENTATION>
    <RESAMPLING_OPTION>CUBIC_CONVOLUTION</RESAMPLING_OPTION>
    <MAP_PROJECTION_L0RA>SOM</MAP_PROJECTION_L0RA>
  </LEVEL1_PROJECTION_PARAMETERS>
  <PRODUCT_PARAMETERS>
    <DATA_TYPE_L0RP>MSSX_L0RP</DATA_TYPE_L0RP>
    <CORRECTION_GAIN_BAND_4>CPF</CORRECTION_GAIN_BAND_4>
    <CORRECTION_GAIN_BAND_5>CPF</CORRECTION_GAIN_BAND_5>
    <CORRECTION_GAIN_BAND_6>CPF</CORRECTION_GAIN_BAND_6>
    <CORRECTION_GAIN_BAND_7>CPF</CORRECTION_GAIN_BAND_7>
    <GAIN_BAND_4>L</GAIN_BAND_4>
    <GAIN_BAND_5>L</GAIN_BAND_5>
    <GAIN_BAND_6>L</GAIN_BAND_6>
    <GAIN_BAND_7>L</GAIN_BAND_7>
  </PRODUCT_PARAMETERS>
</LANDSAT_METADATA_FILE>
"""


l4_metadata_text = b"""<?xml version="1.0" encoding="UTF-8"?>
<LANDSAT_METADATA_FILE>
  <PRODUCT_CONTENTS>
    <ORIGIN>Image courtesy of the U.S. Geological Survey</ORIGIN>
    <DIGITAL_OBJECT_IDENTIFIER>https://doi.org/10.5066/P9AF14YV</DIGITAL_OBJECT_IDENTIFIER>
    <LANDSAT_PRODUCT_ID>LM04_L1TP_029030_19840415_20200903_02_T2</LANDSAT_PRODUCT_ID>
    <PROCESSING_LEVEL>L1TP</PROCESSING_LEVEL>
    <COLLECTION_NUMBER>02</COLLECTION_NUMBER>
    <COLLECTION_CATEGORY>T2</COLLECTION_CATEGORY>
    <OUTPUT_FORMAT>GEOTIFF</OUTPUT_FORMAT>
    <FILE_NAME_BAND_1>LM04_L1TP_029030_19840415_20200903_02_T2_B1.TIF</FILE_NAME_BAND_1>
    <FILE_NAME_BAND_2>LM04_L1TP_029030_19840415_20200903_02_T2_B2.TIF</FILE_NAME_BAND_2>
    <FILE_NAME_BAND_3>LM04_L1TP_029030_19840415_20200903_02_T2_B3.TIF</FILE_NAME_BAND_3>
    <FILE_NAME_BAND_4>LM04_L1TP_029030_19840415_20200903_02_T2_B4.TIF</FILE_NAME_BAND_4>
    <FILE_NAME_QUALITY_L1_PIXEL>LM04_L1TP_029030_19840415_20200903_02_T2_QA_PIXEL.TIF</FILE_NAME_QUALITY_L1_PIXEL>
    <FILE_NAME_QUALITY_L1_RADIOMETRIC_SATURATION>LM04_L1TP_029030_19840415_20200903_02_T2_QA_RADSAT.TIF</FILE_NAME_QUALITY_L1_RADIOMETRIC_SATURATION>
    <FILE_NAME_GROUND_CONTROL_POINT>LM04_L1TP_029030_19840415_20200903_02_T2_GCP.txt</FILE_NAME_GROUND_CONTROL_POINT>
    <FILE_NAME_METADATA_ODL>LM04_L1TP_029030_19840415_20200903_02_T2_MTL.txt</FILE_NAME_METADATA_ODL>
    <FILE_NAME_METADATA_XML>LM04_L1TP_029030_19840415_20200903_02_T2_MTL.xml</FILE_NAME_METADATA_XML>
    <FILE_NAME_VERIFY_REPORT>LM04_L1TP_029030_19840415_20200903_02_T2_VER.txt</FILE_NAME_VERIFY_REPORT>
    <FILE_NAME_VERIFY_BROWSE>LM04_L1TP_029030_19840415_20200903_02_T2_VER.jpg</FILE_NAME_VERIFY_BROWSE>
    <DATA_TYPE_BAND_1>UINT8</DATA_TYPE_BAND_1>
    <DATA_TYPE_BAND_2>UINT8</DATA_TYPE_BAND_2>
    <DATA_TYPE_BAND_3>UINT8</DATA_TYPE_BAND_3>
    <DATA_TYPE_BAND_4>UINT8</DATA_TYPE_BAND_4>
    <DATA_TYPE_QUALITY_L1_PIXEL>UINT16</DATA_TYPE_QUALITY_L1_PIXEL>
    <DATA_TYPE_QUALITY_L1_RADIOMETRIC_SATURATION>UINT16</DATA_TYPE_QUALITY_L1_RADIOMETRIC_SATURATION>
    <PRESENT_BAND_1>Y</PRESENT_BAND_1>
    <PRESENT_BAND_2>Y</PRESENT_BAND_2>
    <PRESENT_BAND_3>Y</PRESENT_BAND_3>
    <PRESENT_BAND_4>Y</PRESENT_BAND_4>
  </PRODUCT_CONTENTS>
  <IMAGE_ATTRIBUTES>
    <SPACECRAFT_ID>LANDSAT_4</SPACECRAFT_ID>
    <SENSOR_ID>MSS</SENSOR_ID>
    <WRS_TYPE>2</WRS_TYPE>
    <WRS_PATH>029</WRS_PATH>
    <WRS_ROW>030</WRS_ROW>
    <DATE_ACQUIRED>1984-04-15</DATE_ACQUIRED>
    <SCENE_CENTER_TIME>16:38:15.9750000Z</SCENE_CENTER_TIME>
    <STATION_ID>AAA</STATION_ID>
    <CLOUD_COVER>2.00</CLOUD_COVER>
    <CLOUD_COVER_LAND>2.00</CLOUD_COVER_LAND>
    <IMAGE_QUALITY>5</IMAGE_QUALITY>
    <SATURATION_BAND_1>N</SATURATION_BAND_1>
    <SATURATION_BAND_2>Y</SATURATION_BAND_2>
    <SATURATION_BAND_3>Y</SATURATION_BAND_3>
    <SATURATION_BAND_4>N</SATURATION_BAND_4>
    <SUN_AZIMUTH>135.96433420</SUN_AZIMUTH>
    <SUN_ELEVATION>49.08760662</SUN_ELEVATION>
    <EARTH_SUN_DISTANCE>1.0035512</EARTH_SUN_DISTANCE>
  </IMAGE_ATTRIBUTES>
  <PROJECTION_ATTRIBUTES>
    <MAP_PROJECTION>UTM</MAP_PROJECTION>
    <DATUM>WGS84</DATUM>
    <ELLIPSOID>WGS84</ELLIPSOID>
    <UTM_ZONE>14</UTM_ZONE>
    <GRID_CELL_SIZE_REFLECTIVE>60.00</GRID_CELL_SIZE_REFLECTIVE>
    <REFLECTIVE_LINES>100</REFLECTIVE_LINES>
    <REFLECTIVE_SAMPLES>100</REFLECTIVE_SAMPLES>
    <ORIENTATION>NORTH_UP</ORIENTATION>
    <CORNER_UL_LAT_PRODUCT>44.14698</CORNER_UL_LAT_PRODUCT>
    <CORNER_UL_LON_PRODUCT>-98.49910</CORNER_UL_LON_PRODUCT>
    <CORNER_UR_LAT_PRODUCT>44.09661</CORNER_UR_LAT_PRODUCT>
    <CORNER_UR_LON_PRODUCT>-95.57170</CORNER_UR_LON_PRODUCT>
    <CORNER_LL_LAT_PRODUCT>42.27118</CORNER_LL_LAT_PRODUCT>
    <CORNER_LL_LON_PRODUCT>-98.51422</CORNER_LL_LON_PRODUCT>
    <CORNER_LR_LAT_PRODUCT>42.22399</CORNER_LR_LAT_PRODUCT>
    <CORNER_LR_LON_PRODUCT>-95.67495</CORNER_LR_LON_PRODUCT>
    <CORNER_UL_PROJECTION_X_PRODUCT>540060.000</CORNER_UL_PROJECTION_X_PRODUCT>
    <CORNER_UL_PROJECTION_Y_PRODUCT>4888320.000</CORNER_UL_PROJECTION_Y_PRODUCT>
    <CORNER_UR_PROJECTION_X_PRODUCT>774420.000</CORNER_UR_PROJECTION_X_PRODUCT>
    <CORNER_UR_PROJECTION_Y_PRODUCT>4888320.000</CORNER_UR_PROJECTION_Y_PRODUCT>
    <CORNER_LL_PROJECTION_X_PRODUCT>540060.000</CORNER_LL_PROJECTION_X_PRODUCT>
    <CORNER_LL_PROJECTION_Y_PRODUCT>4680000.000</CORNER_LL_PROJECTION_Y_PRODUCT>
    <CORNER_LR_PROJECTION_X_PRODUCT>774420.000</CORNER_LR_PROJECTION_X_PRODUCT>
    <CORNER_LR_PROJECTION_Y_PRODUCT>4680000.000</CORNER_LR_PROJECTION_Y_PRODUCT>
  </PROJECTION_ATTRIBUTES>
  <LEVEL1_PROCESSING_RECORD>
    <ORIGIN>Image courtesy of the U.S. Geological Survey</ORIGIN>
    <DIGITAL_OBJECT_IDENTIFIER>https://doi.org/10.5066/P9AF14YV</DIGITAL_OBJECT_IDENTIFIER>
    <REQUEST_ID>L2</REQUEST_ID>
    <LANDSAT_SCENE_ID>LM40290301984106AAA03</LANDSAT_SCENE_ID>
    <LANDSAT_PRODUCT_ID>LM04_L1TP_029030_19840415_20200903_02_T2</LANDSAT_PRODUCT_ID>
    <PROCESSING_LEVEL>L1TP</PROCESSING_LEVEL>
    <COLLECTION_CATEGORY>T2</COLLECTION_CATEGORY>
    <OUTPUT_FORMAT>GEOTIFF</OUTPUT_FORMAT>
    <DATE_PRODUCT_GENERATED>2020-09-03T04:50:06Z</DATE_PRODUCT_GENERATED>
    <PROCESSING_SOFTWARE_VERSION>LPGS_15.3.1c</PROCESSING_SOFTWARE_VERSION>
    <FILE_NAME_BAND_1>LM04_L1TP_029030_19840415_20200903_02_T2_B1.TIF</FILE_NAME_BAND_1>
    <FILE_NAME_BAND_2>LM04_L1TP_029030_19840415_20200903_02_T2_B2.TIF</FILE_NAME_BAND_2>
    <FILE_NAME_BAND_3>LM04_L1TP_029030_19840415_20200903_02_T2_B3.TIF</FILE_NAME_BAND_3>
    <FILE_NAME_BAND_4>LM04_L1TP_029030_19840415_20200903_02_T2_B4.TIF</FILE_NAME_BAND_4>
    <FILE_NAME_QUALITY_L1_PIXEL>LM04_L1TP_029030_19840415_20200903_02_T2_QA_PIXEL.TIF</FILE_NAME_QUALITY_L1_PIXEL>
    <FILE_NAME_QUALITY_L1_RADIOMETRIC_SATURATION>LM04_L1TP_029030_19840415_20200903_02_T2_QA_RADSAT.TIF</FILE_NAME_QUALITY_L1_RADIOMETRIC_SATURATION>
    <FILE_NAME_GROUND_CONTROL_POINT>LM04_L1TP_029030_19840415_20200903_02_T2_GCP.txt</FILE_NAME_GROUND_CONTROL_POINT>
    <FILE_NAME_METADATA_ODL>LM04_L1TP_029030_19840415_20200903_02_T2_MTL.txt</FILE_NAME_METADATA_ODL>
    <FILE_NAME_METADATA_XML>LM04_L1TP_029030_19840415_20200903_02_T2_MTL.xml</FILE_NAME_METADATA_XML>
    <FILE_NAME_CPF>LM04CPF_19830401_19931231_02.01</FILE_NAME_CPF>
    <FILE_NAME_VERIFY_REPORT>LM04_L1TP_029030_19840415_20200903_02_T2_VER.txt</FILE_NAME_VERIFY_REPORT>
    <FILE_NAME_VERIFY_BROWSE>LM04_L1TP_029030_19840415_20200903_02_T2_VER.jpg</FILE_NAME_VERIFY_BROWSE>
    <DATA_SOURCE_ELEVATION>GLS2000</DATA_SOURCE_ELEVATION>
    <GROUND_CONTROL_POINTS_VERSION>5</GROUND_CONTROL_POINTS_VERSION>
    <GROUND_CONTROL_POINTS_MODEL>201</GROUND_CONTROL_POINTS_MODEL>
    <GEOMETRIC_RMSE_MODEL>19.942</GEOMETRIC_RMSE_MODEL>
    <GEOMETRIC_RMSE_MODEL_Y>14.470</GEOMETRIC_RMSE_MODEL_Y>
    <GEOMETRIC_RMSE_MODEL_X>13.722</GEOMETRIC_RMSE_MODEL_X>
    <GROUND_CONTROL_POINTS_VERIFY>348</GROUND_CONTROL_POINTS_VERIFY>
    <GEOMETRIC_RMSE_VERIFY>0.275</GEOMETRIC_RMSE_VERIFY>
    <GEOMETRIC_RMSE_VERIFY_QUAD_UL>0.259</GEOMETRIC_RMSE_VERIFY_QUAD_UL>
    <GEOMETRIC_RMSE_VERIFY_QUAD_UR>0.344</GEOMETRIC_RMSE_VERIFY_QUAD_UR>
    <GEOMETRIC_RMSE_VERIFY_QUAD_LL>0.291</GEOMETRIC_RMSE_VERIFY_QUAD_LL>
    <GEOMETRIC_RMSE_VERIFY_QUAD_LR>0.302</GEOMETRIC_RMSE_VERIFY_QUAD_LR>
    <EPHEMERIS_TYPE>PREDICTIVE</EPHEMERIS_TYPE>
  </LEVEL1_PROCESSING_RECORD>
  <LEVEL1_MIN_MAX_RADIANCE>
    <RADIANCE_MAXIMUM_BAND_1>226.100</RADIANCE_MAXIMUM_BAND_1>
    <RADIANCE_MINIMUM_BAND_1>3.800</RADIANCE_MINIMUM_BAND_1>
    <RADIANCE_MAXIMUM_BAND_2>161.200</RADIANCE_MAXIMUM_BAND_2>
    <RADIANCE_MINIMUM_BAND_2>3.700</RADIANCE_MINIMUM_BAND_2>
    <RADIANCE_MAXIMUM_BAND_3>144.600</RADIANCE_MAXIMUM_BAND_3>
    <RADIANCE_MINIMUM_BAND_3>5.100</RADIANCE_MINIMUM_BAND_3>
    <RADIANCE_MAXIMUM_BAND_4>125.300</RADIANCE_MAXIMUM_BAND_4>
    <RADIANCE_MINIMUM_BAND_4>4.300</RADIANCE_MINIMUM_BAND_4>
  </LEVEL1_MIN_MAX_RADIANCE>
  <LEVEL1_MIN_MAX_REFLECTANCE>
    <REFLECTANCE_MAXIMUM_BAND_1>0.405078</REFLECTANCE_MAXIMUM_BAND_1>
    <REFLECTANCE_MINIMUM_BAND_1>0.006808</REFLECTANCE_MINIMUM_BAND_1>
    <REFLECTANCE_MAXIMUM_BAND_2>0.334445</REFLECTANCE_MAXIMUM_BAND_2>
    <REFLECTANCE_MINIMUM_BAND_2>0.007676</REFLECTANCE_MINIMUM_BAND_2>
    <REFLECTANCE_MAXIMUM_BAND_3>0.370451</REFLECTANCE_MAXIMUM_BAND_3>
    <REFLECTANCE_MINIMUM_BAND_3>0.013066</REFLECTANCE_MINIMUM_BAND_3>
    <REFLECTANCE_MAXIMUM_BAND_4>0.472236</REFLECTANCE_MAXIMUM_BAND_4>
    <REFLECTANCE_MINIMUM_BAND_4>0.016206</REFLECTANCE_MINIMUM_BAND_4>
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
  </LEVEL1_MIN_MAX_PIXEL_VALUE>
  <LEVEL1_RADIOMETRIC_RESCALING>
    <RADIANCE_MULT_BAND_1>8.7520E-01</RADIANCE_MULT_BAND_1>
    <RADIANCE_MULT_BAND_2>6.2008E-01</RADIANCE_MULT_BAND_2>
    <RADIANCE_MULT_BAND_3>5.4921E-01</RADIANCE_MULT_BAND_3>
    <RADIANCE_MULT_BAND_4>4.7638E-01</RADIANCE_MULT_BAND_4>
    <RADIANCE_ADD_BAND_1>2.92480</RADIANCE_ADD_BAND_1>
    <RADIANCE_ADD_BAND_2>3.07992</RADIANCE_ADD_BAND_2>
    <RADIANCE_ADD_BAND_3>4.55079</RADIANCE_ADD_BAND_3>
    <RADIANCE_ADD_BAND_4>3.82362</RADIANCE_ADD_BAND_4>
    <REFLECTANCE_MULT_BAND_1>1.5680E-03</REFLECTANCE_MULT_BAND_1>
    <REFLECTANCE_MULT_BAND_2>1.2865E-03</REFLECTANCE_MULT_BAND_2>
    <REFLECTANCE_MULT_BAND_3>1.4070E-03</REFLECTANCE_MULT_BAND_3>
    <REFLECTANCE_MULT_BAND_4>1.7954E-03</REFLECTANCE_MULT_BAND_4>
    <REFLECTANCE_ADD_BAND_1>0.005240</REFLECTANCE_ADD_BAND_1>
    <REFLECTANCE_ADD_BAND_2>0.006390</REFLECTANCE_ADD_BAND_2>
    <REFLECTANCE_ADD_BAND_3>0.011659</REFLECTANCE_ADD_BAND_3>
    <REFLECTANCE_ADD_BAND_4>0.014411</REFLECTANCE_ADD_BAND_4>
  </LEVEL1_RADIOMETRIC_RESCALING>
  <LEVEL1_PROJECTION_PARAMETERS>
    <MAP_PROJECTION>UTM</MAP_PROJECTION>
    <DATUM>WGS84</DATUM>
    <ELLIPSOID>WGS84</ELLIPSOID>
    <UTM_ZONE>14</UTM_ZONE>
    <GRID_CELL_SIZE_REFLECTIVE>60.00</GRID_CELL_SIZE_REFLECTIVE>
    <ORIENTATION>NORTH_UP</ORIENTATION>
    <RESAMPLING_OPTION>CUBIC_CONVOLUTION</RESAMPLING_OPTION>
    <MAP_PROJECTION_L0RA>SOM</MAP_PROJECTION_L0RA>
  </LEVEL1_PROJECTION_PARAMETERS>
  <PRODUCT_PARAMETERS>
    <DATA_TYPE_L0RP>MSSA_L0RP</DATA_TYPE_L0RP>
    <CORRECTION_GAIN_BAND_1>CPF</CORRECTION_GAIN_BAND_1>
    <CORRECTION_GAIN_BAND_2>CPF</CORRECTION_GAIN_BAND_2>
    <CORRECTION_GAIN_BAND_3>CPF</CORRECTION_GAIN_BAND_3>
    <CORRECTION_GAIN_BAND_4>CPF</CORRECTION_GAIN_BAND_4>
    <GAIN_BAND_1>L</GAIN_BAND_1>
    <GAIN_BAND_2>L</GAIN_BAND_2>
    <GAIN_BAND_3>L</GAIN_BAND_3>
    <GAIN_BAND_4>L</GAIN_BAND_4>
  </PRODUCT_PARAMETERS>
</LANDSAT_METADATA_FILE>
"""


x_size = 100
y_size = 100
date_l1 = datetime(1972, 7, 29, tzinfo=timezone.utc)
date_l4 = datetime(1984, 4, 15, tzinfo=timezone.utc)


@pytest.fixture(scope="session")
def l1_area():
    """Get the landsat 1 area def."""
    pcs_id = "WGS84 / UMSS zone 14N"
    proj4_dict = {"proj": "utm", "zone": 14, "datum": "WGS84", "units": "m", "no_defs": None, "type": "crs"}
    area_extent = (442110.0, 4667550.0, 673530.0, 4887990.0)
    return AreaDefinition("geotiff_area", pcs_id, pcs_id,
                          proj4_dict, x_size, y_size,
                          area_extent)


@pytest.fixture(scope="session")
def l4_area():
    """Get the landsat 1 area def."""
    pcs_id = "WGS84 / UMSS zone 14N"
    proj4_dict = {"proj": "utm", "zone": 14, "datum": "WGS84", "units": "m", "no_defs": None, "type": "crs"}
    area_extent = (540030.0, 4679970.0, 774450.0, 4888350.0)
    return AreaDefinition("geotiff_area", pcs_id, pcs_id,
                          proj4_dict, x_size, y_size,
                          area_extent)


@pytest.fixture(scope="session")
def b4_data():
    """Get the data for the b4 channel."""
    return da.random.randint(12000, 16000,
                             size=(y_size, x_size),
                             chunks=(50, 50)).astype(np.uint16)


def create_tif_file(data, name, area, filename, date):
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
def l1_files_path(tmp_path_factory):
    """Create the path for l1 files."""
    return tmp_path_factory.mktemp("mss_l1_files")


@pytest.fixture(scope="session")
def l4_files_path(tmp_path_factory):
    """Create the path for l4 files."""
    return tmp_path_factory.mktemp("mss_l4_files")


@pytest.fixture(scope="session")
def l1_b4_file(l1_files_path, b4_data, l1_area):
    """Create the file for the b4 channel."""
    data = b4_data
    filename = l1_files_path / "LM01_L1TP_032030_19720729_20200909_02_T2_B4.TIF"
    name = "B4"
    create_tif_file(data, name, l1_area, filename, date_l1)
    return os.fspath(filename)


@pytest.fixture(scope="session")
def l4_b4_file(l4_files_path, b4_data, l4_area):
    """Create the file for the b4 channel."""
    data = b4_data
    filename = l4_files_path / "LM04_L1TP_029030_19840415_20200903_02_T2_B4.TIF"
    name = "B4"
    create_tif_file(data, name, l4_area, filename, date_l4)
    return os.fspath(filename)


@pytest.fixture(scope="session")
def l1_mda_file(l1_files_path):
    """Create the metadata xml file."""
    filename = l1_files_path / "LM01_L1TP_032030_19720729_20200909_02_T2_MTL.xml"
    with open(filename, "wb") as f:
        f.write(l1_metadata_text)
    return os.fspath(filename)


@pytest.fixture(scope="session")
def l4_mda_file(l4_files_path):
    """Create the metadata xml file."""
    filename = l4_files_path / "LM04_L1TP_029030_19840415_20200903_02_T2_MTL.xml"
    with open(filename, "wb") as f:
        f.write(l4_metadata_text)
    return os.fspath(filename)


@pytest.fixture(scope="session")
def l1_all_files(l1_b4_file, l1_mda_file):
    """Return all the files."""
    return l1_b4_file, l1_mda_file


@pytest.fixture(scope="session")
def l4_all_files(l4_b4_file, l4_mda_file):
    """Return all the files."""
    return l4_b4_file, l4_mda_file


@pytest.fixture(scope="session")
def l1_all_fs_files(l1_b4_file, l1_mda_file):
    """Return all the files as FSFile objects."""
    from fsspec.implementations.local import LocalFileSystem

    from satpy.readers.core.remote import FSFile

    fs = LocalFileSystem()
    l1_b4_file, l1_mda_file = (
        FSFile(os.path.abspath(file), fs=fs)
        for file in [l1_b4_file, l1_mda_file]
    )
    return l1_b4_file, l1_mda_file


@pytest.fixture(scope="session")
def l4_all_fs_files(l4_b4_file, l4_mda_file):
    """Return all the files as FSFile objects."""
    from fsspec.implementations.local import LocalFileSystem

    from satpy.readers.core.remote import FSFile

    fs = LocalFileSystem()
    l4_b4_file, l4_mda_file = (
        FSFile(os.path.abspath(file), fs=fs)
        for file in [l4_b4_file, l4_mda_file]
    )
    return l4_b4_file, l4_mda_file


class TestMSSL1_Landsat_1:
    """Test generic image reader."""

    def setup_method(self, tmp_path):
        """Set up the filename and filetype info dicts.."""
        self.filename_info = dict(observation_date=datetime(1972, 7, 29),
                                  platform_type="L",
                                  process_level_correction="L1TP",
                                  spacecraft_id="01",
                                  data_type="M",
                                  collection_id="02")
        self.ftype_info = {"file_type": "granule_B4"}

    def test_basicload(self, l1_area, l1_b4_file, l1_mda_file):
        """Test loading a Landsat Scene."""
        scn = Scene(reader="mss_l1_tif", filenames=[l1_b4_file, l1_mda_file])
        scn.load(["B4"])

        # Check dataset is loaded correctly
        assert scn["B4"].shape == (100, 100)
        assert scn["B4"].attrs["area"] == l1_area
        assert not scn["B4"].attrs["saturated"]
        assert scn["B4"].attrs["_satpy_id"]["wavelength"].min == 0.5
        assert scn["B4"].attrs["_satpy_id"]["wavelength"].central == 0.55
        assert scn["B4"].attrs["_satpy_id"]["wavelength"].max == 0.6

    def test_ch_startend(self, l1_b4_file, l1_mda_file):
        """Test correct retrieval of start/end times."""
        scn = Scene(reader="mss_l1_tif", filenames=[l1_b4_file, l1_mda_file])
        bnds = scn.available_dataset_names()
        assert bnds == ["B4"]

        scn.load(["B4"])
        assert scn.start_time == datetime(1972, 7, 29, 16, 49, 31, tzinfo=timezone.utc)
        assert scn.end_time == datetime(1972, 7, 29, 16, 49, 31, tzinfo=timezone.utc)

    def test_loading_gd(self, l1_mda_file, l1_b4_file):
        """Test loading a Landsat Scene with good channel requests."""
        from satpy.readers.landsat_base import MSSCHReader, MSSMDReader
        good_mda = MSSMDReader(l1_mda_file, self.filename_info, {})
        rdr = MSSCHReader(l1_b4_file, self.filename_info, self.ftype_info, good_mda)

        # Check case with good file data and load request
        rdr.get_dataset({"name": "B4", "calibration": "counts"}, {"standard_name": "test_data", "units": "test_units"})

    def test_loading_badfil(self, l1_mda_file, l1_b4_file):
        """Test loading a Landsat Scene with bad channel requests."""
        from satpy.readers.landsat_base import MSSCHReader, MSSMDReader
        good_mda = MSSMDReader(l1_mda_file, self.filename_info, {})
        rdr = MSSCHReader(l1_b4_file, self.filename_info, self.ftype_info, good_mda)

        ftype = {"standard_name": "test_data", "units": "test_units"}
        # Check case with request to load channel not matching filename
        with pytest.raises(ValueError, match="Requested channel B5 does not match the reader channel B4"):
            rdr.get_dataset({"name": "B5", "calibration": "counts"}, ftype)

    def test_badfiles(self, l1_mda_file, l1_b4_file):
        """Test loading a Landsat Scene with bad data."""
        from satpy.readers.landsat_base import MSSCHReader, MSSMDReader
        bad_fname_info = self.filename_info.copy()
        bad_fname_info["platform_type"] = "B"

        ftype = {"standard_name": "test_data", "units": "test_units"}

        # Test that metadata reader initialises with correct filename
        good_mda = MSSMDReader(l1_mda_file, self.filename_info, ftype)

        # Check metadata reader fails if platform type is wrong
        with pytest.raises(ValueError, match="This reader only supports Landsat data"):
            MSSMDReader(l1_mda_file, bad_fname_info, ftype)

        # Test that metadata reader initialises with correct filename
        MSSCHReader(l1_b4_file, self.filename_info, self.ftype_info, good_mda)

        # Check metadata reader fails if platform type is wrong
        with pytest.raises(ValueError, match="This reader only supports Landsat data"):
            MSSCHReader(l1_b4_file, bad_fname_info, self.ftype_info, good_mda)
        bad_ftype_info = self.ftype_info.copy()
        bad_ftype_info["file_type"] = "granule-b05"
        with pytest.raises(ValueError, match="Invalid file type: granule-b05"):
            MSSCHReader(l1_b4_file, self.filename_info, bad_ftype_info, good_mda)

    def test_calibration_counts(self, l1_all_files, b4_data):
        """Test counts calibration mode for the reader."""
        from satpy import Scene

        scn = Scene(reader="mss_l1_tif", filenames=l1_all_files)
        scn.load(["B4"], calibration="counts")
        np.testing.assert_allclose(scn["B4"].values, b4_data)
        assert scn["B4"].attrs["units"] == "1"
        assert scn["B4"].attrs["standard_name"] == "counts"

    def test_calibration_radiance(self, l1_all_files, b4_data):
        """Test radiance calibration mode for the reader."""
        from satpy import Scene
        exp_b04 = (b4_data * 9.5591e-01 - 18.55591).astype(np.float32)

        scn = Scene(reader="mss_l1_tif", filenames=l1_all_files)
        scn.load(["B4"], calibration="radiance")
        assert scn["B4"].attrs["units"] == "W m-2 um-1 sr-1"
        assert scn["B4"].attrs["standard_name"] == "toa_outgoing_radiance_per_unit_wavelength"
        np.testing.assert_allclose(scn["B4"].values, exp_b04, rtol=1e-4)

    def test_calibration_highlevel(self, l1_all_files, b4_data):
        """Test high level calibration modes for the reader."""
        from satpy import Scene
        exp_b04 = (b4_data * 1.7282e-03 - 0.033547).astype(np.float32) * 100
        scn = Scene(reader="mss_l1_tif", filenames=l1_all_files)
        scn.load(["B4"])

        assert scn["B4"].attrs["units"] == "%"
        assert scn["B4"].attrs["standard_name"] == "toa_bidirectional_reflectance"
        np.testing.assert_allclose(np.array(scn["B4"].values), np.array(exp_b04), rtol=1e-4)

    def test_metadata(self, l1_mda_file):
        """Check that metadata values loaded correctly."""
        from satpy.readers.landsat_base import MSSMDReader
        mda = MSSMDReader(l1_mda_file, self.filename_info, {})

        cal_test_dict = {"B5": (6.4843e-01, -0.74843, 1.3660e-03, -0.001577),
                         "B6": (6.5236e-01, -0.75236, 1.6580e-03, -0.001912),
                         "B7": (6.0866e-01, -0.60866, 2.3287e-03, -0.002329)}

        assert mda.platform_name == "Landsat-1"
        assert mda.earth_sun_distance() == 1.0152109
        assert mda.band_calibration["B5"] == cal_test_dict["B5"]
        assert mda.band_calibration["B6"] == cal_test_dict["B6"]
        assert mda.band_calibration["B7"] == cal_test_dict["B7"]
        assert not mda.band_saturation["B4"]
        assert mda.band_saturation["B5"]
        assert mda.band_saturation["B6"]
        assert not mda.band_saturation["B7"]

    def test_area_def(self, l1_mda_file):
        """Check we can get the area defs properly."""
        from satpy.readers.landsat_base import MSSMDReader
        mda = MSSMDReader(l1_mda_file, self.filename_info, {})

        standard_area = mda.build_area_def("B4")

        assert standard_area.area_extent == (442110.0, 4667550.0, 673530.0, 4887990.0)

    def test_basicload_remote(self, l1_area, l1_all_fs_files):
        """Test loading a Landsat Scene from a fsspec filesystem."""
        scn = Scene(reader="mss_l1_tif", filenames=l1_all_fs_files)
        scn.load(["B4"])

        # Check dataset is loaded correctly
        assert scn["B4"].shape == (100, 100)
        assert scn["B4"].attrs["area"] == l1_area
        assert not scn["B4"].attrs["saturated"]
        assert scn["B4"].attrs["_satpy_id"]["wavelength"].min == 0.5
        assert scn["B4"].attrs["_satpy_id"]["wavelength"].central == 0.55
        assert scn["B4"].attrs["_satpy_id"]["wavelength"].max == 0.6


class TestMSSL1_Landsat_4:
    """Test generic image reader."""

    def setup_method(self, tmp_path):
        """Set up the filename and filetype info dicts.."""
        self.filename_info = dict(observation_date=datetime(1984, 4, 15),
                                  platform_type="L",
                                  process_level_correction="L1TP",
                                  spacecraft_id="04",
                                  data_type="M",
                                  collection_id="02")
        self.ftype_info = {"file_type": "granule_B4"}

    def test_basicload(self, l4_area, l4_b4_file, l4_mda_file):
        """Test loading a Landsat Scene."""
        scn = Scene(reader="mss_l1_tif", filenames=[l4_b4_file, l4_mda_file])
        scn.load(["B4"])

        # Check dataset is loaded correctly
        assert scn["B4"].shape == (100, 100)
        assert scn["B4"].attrs["area"] == l4_area
        assert not scn["B4"].attrs["saturated"]
        assert scn["B4"].attrs["_satpy_id"]["wavelength"].min == 0.8
        assert scn["B4"].attrs["_satpy_id"]["wavelength"].central == 0.95
        assert scn["B4"].attrs["_satpy_id"]["wavelength"].max == 1.1

    def test_ch_startend(self, l4_b4_file, l4_mda_file):
        """Test correct retrieval of start/end times."""
        scn = Scene(reader="mss_l1_tif", filenames=[l4_b4_file, l4_mda_file])
        bnds = scn.available_dataset_names()
        assert bnds == ["B4"]

        scn.load(["B4"])
        assert scn.start_time == datetime(1984, 4, 15, 16, 38, 15, tzinfo=timezone.utc)
        assert scn.end_time == datetime(1984, 4, 15, 16, 38, 15, tzinfo=timezone.utc)

    def test_loading_gd(self, l4_mda_file, l4_b4_file):
        """Test loading a Landsat Scene with good channel requests."""
        from satpy.readers.landsat_base import MSSCHReader, MSSMDReader
        good_mda = MSSMDReader(l4_mda_file, self.filename_info, {})
        rdr = MSSCHReader(l4_b4_file, self.filename_info, self.ftype_info, good_mda)

        # Check case with good file data and load request
        rdr.get_dataset({"name": "B4", "calibration": "counts"}, {"standard_name": "test_data", "units": "test_units"})

    def test_loading_badfil(self, l4_mda_file, l4_b4_file):
        """Test loading a Landsat Scene with bad channel requests."""
        from satpy.readers.landsat_base import MSSCHReader, MSSMDReader
        good_mda = MSSMDReader(l4_mda_file, self.filename_info, {})
        rdr = MSSCHReader(l4_b4_file, self.filename_info, self.ftype_info, good_mda)

        ftype = {"standard_name": "test_data", "units": "test_units"}
        # Check case with request to load channel not matching filename
        with pytest.raises(ValueError, match="Requested channel B5 does not match the reader channel B4"):
            rdr.get_dataset({"name": "B5", "calibration": "counts"}, ftype)

    def test_badfiles(self, l4_mda_file, l4_b4_file):
        """Test loading a Landsat Scene with bad data."""
        from satpy.readers.landsat_base import MSSCHReader, MSSMDReader
        bad_fname_info = self.filename_info.copy()
        bad_fname_info["platform_type"] = "B"

        ftype = {"standard_name": "test_data", "units": "test_units"}

        # Test that metadata reader initialises with correct filename
        good_mda = MSSMDReader(l4_mda_file, self.filename_info, ftype)

        # Check metadata reader fails if platform type is wrong
        with pytest.raises(ValueError, match="This reader only supports Landsat data"):
            MSSMDReader(l4_mda_file, bad_fname_info, ftype)

        # Test that metadata reader initialises with correct filename
        MSSCHReader(l4_b4_file, self.filename_info, self.ftype_info, good_mda)

        # Check metadata reader fails if platform type is wrong
        with pytest.raises(ValueError, match="This reader only supports Landsat data"):
            MSSCHReader(l4_b4_file, bad_fname_info, self.ftype_info, good_mda)
        bad_ftype_info = self.ftype_info.copy()
        bad_ftype_info["file_type"] = "granule-b05"
        with pytest.raises(ValueError, match="Invalid file type: granule-b05"):
            MSSCHReader(l4_b4_file, self.filename_info, bad_ftype_info, good_mda)

    def test_calibration_counts(self, l4_all_files, b4_data):
        """Test counts calibration mode for the reader."""
        from satpy import Scene

        scn = Scene(reader="mss_l1_tif", filenames=l4_all_files)
        scn.load(["B4"], calibration="counts")
        np.testing.assert_allclose(scn["B4"].values, b4_data)
        assert scn["B4"].attrs["units"] == "1"
        assert scn["B4"].attrs["standard_name"] == "counts"

    def test_calibration_radiance(self, l4_all_files, b4_data):
        """Test radiance calibration mode for the reader."""
        from satpy import Scene
        exp_b04 = (b4_data * 4.7638e-01 + 3.82362).astype(np.float32)

        scn = Scene(reader="mss_l1_tif", filenames=l4_all_files)
        scn.load(["B4"], calibration="radiance")
        assert scn["B4"].attrs["units"] == "W m-2 um-1 sr-1"
        assert scn["B4"].attrs["standard_name"] == "toa_outgoing_radiance_per_unit_wavelength"
        np.testing.assert_allclose(scn["B4"].values, exp_b04, rtol=1e-4)

    def test_calibration_highlevel(self, l4_all_files, b4_data):
        """Test high level calibration modes for the reader."""
        from satpy import Scene
        exp_b04 = (b4_data * 1.7954e-03 + 0.014411).astype(np.float32) * 100
        scn = Scene(reader="mss_l1_tif", filenames=l4_all_files)
        scn.load(["B4"])

        assert scn["B4"].attrs["units"] == "%"
        assert scn["B4"].attrs["standard_name"] == "toa_bidirectional_reflectance"
        np.testing.assert_allclose(np.array(scn["B4"].values), np.array(exp_b04), rtol=1e-4)

    def test_metadata(self, l4_mda_file):
        """Check that metadata values loaded correctly."""
        from satpy.readers.landsat_base import MSSMDReader
        mda = MSSMDReader(l4_mda_file, self.filename_info, {})

        cal_test_dict = {"B1": (8.7520e-01, 2.92480, 1.5680e-03, 0.005240),
                         "B2": (6.2008e-01, 3.07992, 1.2865e-03, 0.006390),
                         "B3": (5.4921e-01, 4.55079, 1.4070e-03, 0.011659)}

        assert mda.platform_name == "Landsat-4"
        assert mda.earth_sun_distance() == 1.0035512
        assert mda.band_calibration["B1"] == cal_test_dict["B1"]
        assert mda.band_calibration["B2"] == cal_test_dict["B2"]
        assert mda.band_calibration["B3"] == cal_test_dict["B3"]
        assert not mda.band_saturation["B1"]
        assert mda.band_saturation["B2"]
        assert mda.band_saturation["B3"]
        assert not mda.band_saturation["B4"]

    def test_area_def(self, l4_mda_file):
        """Check we can get the area defs properly."""
        from satpy.readers.landsat_base import MSSMDReader
        mda = MSSMDReader(l4_mda_file, self.filename_info, {})

        standard_area = mda.build_area_def("B4")

        assert standard_area.area_extent == (540030.0, 4679970.0, 774450.0, 4888350.0)

    def test_basicload_remote(self, l4_area, l4_all_fs_files):
        """Test loading a Landsat Scene from a fsspec filesystem."""
        scn = Scene(reader="mss_l1_tif", filenames=l4_all_fs_files)
        scn.load(["B4"])

        # Check dataset is loaded correctly
        assert scn["B4"].shape == (100, 100)
        assert scn["B4"].attrs["area"] == l4_area
        assert not scn["B4"].attrs["saturated"]
        assert scn["B4"].attrs["_satpy_id"]["wavelength"].min == 0.8
        assert scn["B4"].attrs["_satpy_id"]["wavelength"].central == 0.95
        assert scn["B4"].attrs["_satpy_id"]["wavelength"].max == 1.1
