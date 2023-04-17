#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2019 Satpy developers
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
"""Unittesting the Native SEVIRI reader."""

from __future__ import annotations

import os
import unittest
from datetime import datetime
from unittest import mock

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from satpy.readers.eum_base import time_cds_short
from satpy.readers.seviri_l1b_native import ImageBoundaries, NativeMSGFileHandler, Padder, get_available_channels
from satpy.tests.reader_tests.test_seviri_base import ORBIT_POLYNOMIALS, ORBIT_POLYNOMIALS_INVALID
from satpy.tests.reader_tests.test_seviri_l1b_calibration import TestFileHandlerCalibrationBase
from satpy.tests.utils import assert_attrs_equal, make_dataid

CHANNEL_INDEX_LIST = ['VIS006', 'VIS008', 'IR_016', 'IR_039',
                      'WV_062', 'WV_073', 'IR_087', 'IR_097',
                      'IR_108', 'IR_120', 'IR_134', 'HRV']
AVAILABLE_CHANNELS = {}
for item in CHANNEL_INDEX_LIST:
    AVAILABLE_CHANNELS[item] = True

SEC15HDR = '15_SECONDARY_PRODUCT_HEADER'
IDS = 'SelectedBandIDs'

TEST1_HEADER_CHNLIST: dict[str, dict[str, dict]] = {SEC15HDR: {IDS: {}}}
TEST1_HEADER_CHNLIST[SEC15HDR][IDS]['Value'] = 'XX--XX--XX--'

TEST2_HEADER_CHNLIST: dict[str, dict[str, dict]] = {SEC15HDR: {IDS: {}}}
TEST2_HEADER_CHNLIST[SEC15HDR][IDS]['Value'] = 'XX-XXXX----X'

TEST3_HEADER_CHNLIST: dict[str, dict[str, dict]] = {SEC15HDR: {IDS: {}}}
TEST3_HEADER_CHNLIST[SEC15HDR][IDS]['Value'] = 'XXXXXXXXXXXX'

TEST_AREA_EXTENT_EARTHMODEL1_VISIR_FULLDISK = {
    'earth_model': 1,
    'dataset_id': make_dataid(name='VIS006', resolution=3000),
    'is_full_disk': True,
    'is_rapid_scan': 0,
    'fill_disk': False,
    'expected_area_def': {
        'Area ID': 'msg_seviri_fes_3km',
        'Projection': {'a': '6378169000', 'b': '6356583800', 'h': '35785831',
                       'lon_0': '0', 'no_defs': 'None', 'proj': 'geos',
                       'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'},
        'Number of columns': 3712,
        'Number of rows': 3712,
        'Area extent': (5568748.2758, 5568748.2758, -5568748.2758, -5568748.2758)
    }
}

TEST_AREA_EXTENT_EARTHMODEL1_VISIR_RAPIDSCAN = {
    'earth_model': 1,
    'dataset_id': make_dataid(name='VIS006', resolution=3000),
    'is_full_disk': False,
    'is_rapid_scan': 1,
    'fill_disk': False,
    'expected_area_def': {
        'Area ID': 'msg_seviri_rss_3km',
        'Projection': {'a': '6378169000', 'b': '6356583800', 'h': '35785831',
                       'lon_0': '9.5', 'no_defs': 'None', 'proj': 'geos',
                       'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'},
        'Number of columns': 3712,
        'Number of rows': 1392,
        'Area extent': (5568748.275756836, 5568748.275756836, -5568748.275756836, 1392187.068939209)
    }
}

TEST_AREA_EXTENT_EARTHMODEL1_VISIR_RAPIDSCAN_FILL = {
    'earth_model': 1,
    'dataset_id': make_dataid(name='VIS006', resolution=3000),
    'is_full_disk': False,
    'is_rapid_scan': 1,
    'fill_disk': True,
    'expected_area_def': {
        'Area ID': 'msg_seviri_rss_3km',
        'Projection': {'a': '6378169000', 'b': '6356583800', 'h': '35785831',
                       'lon_0': '9.5', 'no_defs': 'None', 'proj': 'geos',
                       'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'},
        'Number of columns': 3712,
        'Number of rows': 3712,
        'Area extent': (5568748.2758, 5568748.2758, -5568748.2758, -5568748.2758)
    }
}

TEST_AREA_EXTENT_EARTHMODEL1_VISIR_ROI = {
    'earth_model': 1,
    'dataset_id': make_dataid(name='VIS006', resolution=3000),
    'is_full_disk': False,
    'is_rapid_scan': 0,
    'fill_disk': False,
    'expected_area_def': {
        'Area ID': 'msg_seviri_fes_3km',
        'Projection': {'a': '6378169000', 'b': '6356583800', 'h': '35785831',
                       'lon_0': '0', 'no_defs': 'None', 'proj': 'geos',
                       'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'},
        'Number of columns': 2516,
        'Number of rows': 1829,
        'Area extent': (5337717.232, 5154692.6389, -2211297.1332, -333044.7514)

    }
}

TEST_AREA_EXTENT_EARTHMODEL1_VISIR_ROI_FILL = {
    'earth_model': 1,
    'dataset_id': make_dataid(name='VIS006', resolution=3000),
    'is_full_disk': False,
    'is_rapid_scan': 0,
    'fill_disk': True,
    'expected_area_def': {
        'Area ID': 'msg_seviri_fes_3km',
        'Projection': {'a': '6378169000', 'b': '6356583800', 'h': '35785831',
                       'lon_0': '0', 'no_defs': 'None', 'proj': 'geos',
                       'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'},
        'Number of columns': 3712,
        'Number of rows': 3712,
        'Area extent': (5568748.2758, 5568748.2758, -5568748.2758, -5568748.2758)

    }
}

TEST_AREA_EXTENT_EARTHMODEL1_HRV_FULLDISK = {
    'earth_model': 1,
    'dataset_id': make_dataid(name='HRV', resolution=1000),
    'is_full_disk': True,
    'is_rapid_scan': 0,
    'fill_disk': False,
    'expected_area_def': {
        'Area ID': 'msg_seviri_fes_1km',
        'Projection': {'a': '6378169000', 'b': '6356583800', 'h': '35785831',
                       'lon_0': '0', 'no_defs': 'None', 'proj': 'geos',
                       'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'},
        'Number of columns': 5568,
        'Number of rows': 11136,
        'Area extent 0': (5567747.920155525, 2625352.665781975, -1000.1343488693237, -5567747.920155525),
        'Area extent 1': (3602483.924627304, 5569748.188853264, -1966264.1298770905, 2625352.665781975)
    }
}

TEST_AREA_EXTENT_EARTHMODEL1_HRV_FULLDISK_FILL = {
    'earth_model': 1,
    'dataset_id': make_dataid(name='HRV', resolution=1000),
    'is_full_disk': True,
    'is_rapid_scan': 0,
    'fill_disk': True,
    'expected_area_def': {
        'Area ID': 'msg_seviri_fes_1km',
        'Projection': {'a': '6378169000', 'b': '6356583800', 'h': '35785831',
                       'lon_0': '0', 'no_defs': 'None', 'proj': 'geos',
                       'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'},
        'Number of columns': 11136,
        'Number of rows': 11136,
        'Area extent': (5567747.920155525, 5569748.188853264, -5569748.188853264, -5567747.920155525)
    }
}

TEST_AREA_EXTENT_EARTHMODEL1_HRV_RAPIDSCAN = {
    'earth_model': 1,
    'dataset_id': make_dataid(name='HRV', resolution=1000),
    'is_full_disk': False,
    'is_rapid_scan': 1,
    'fill_disk': False,
    'expected_area_def': {
        'Area ID': 'msg_seviri_rss_1km',
        'Projection': {'a': '6378169000', 'b': '6356583800', 'h': '35785831',
                       'lon_0': '9.5', 'no_defs': 'None', 'proj': 'geos',
                       'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'},
        'Number of columns': 5568,
        'Number of rows': 8192,
        'Area extent': (5567747.920155525, 2625352.665781975, -1000.1343488693237, -5567747.920155525)
    }
}

TEST_AREA_EXTENT_EARTHMODEL1_HRV_RAPIDSCAN_FILL = {
    'earth_model': 1,
    'dataset_id': make_dataid(name='HRV', resolution=1000),
    'is_full_disk': False,
    'is_rapid_scan': 1,
    'fill_disk': True,
    'expected_area_def': {
        'Area ID': 'msg_seviri_rss_1km',
        'Projection': {'a': '6378169000', 'b': '6356583800', 'h': '35785831',
                       'lon_0': '9.5', 'no_defs': 'None', 'proj': 'geos',
                       'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'},
        'Number of columns': 11136,
        'Number of rows': 11136,
        'Area extent': (5567747.920155525, 5569748.188853264, -5569748.188853264, -5567747.920155525)
    }
}

TEST_AREA_EXTENT_EARTHMODEL1_HRV_ROI = {
    'earth_model': 1,
    'dataset_id': make_dataid(name='HRV', resolution=1000),
    'is_full_disk': False,
    'is_rapid_scan': 0,
    'fill_disk': False,
    'expected_area_def': {
        'Area ID': 'msg_seviri_fes_1km',
        'Projection': {'a': '6378169000', 'b': '6356583800', 'h': '35785831',
                       'lon_0': '0', 'no_defs': 'None', 'proj': 'geos',
                       'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'},
        'Number of columns': 7548,
        'Number of rows': 5487,
        'Area extent': (5336716.885566711, 5155692.568421364, -2212297.179698944, -332044.6038246155)
    }
}

TEST_AREA_EXTENT_EARTHMODEL1_HRV_ROI_FILL = {
    'earth_model': 1,
    'dataset_id': make_dataid(name='HRV', resolution=1000),
    'is_full_disk': False,
    'is_rapid_scan': 0,
    'fill_disk': True,
    'expected_area_def': {
        'Area ID': 'msg_seviri_fes_1km',
        'Projection': {'a': '6378169000', 'b': '6356583800', 'h': '35785831',
                       'lon_0': '0', 'no_defs': 'None', 'proj': 'geos',
                       'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'},
        'Number of columns': 11136,
        'Number of rows': 11136,
        'Area extent': (5567747.920155525, 5569748.188853264, -5569748.188853264, -5567747.920155525)
    }
}

TEST_AREA_EXTENT_EARTHMODEL2_VISIR_FULLDISK = {
    'earth_model': 2,
    'dataset_id': make_dataid(name='VIS006', resolution=3000),
    'is_full_disk': True,
    'is_rapid_scan': 0,
    'fill_disk': False,
    'expected_area_def': {
        'Area ID': 'msg_seviri_fes_3km',
        'Projection': {'a': '6378169000', 'b': '6356583800', 'h': '35785831',
                       'lon_0': '0', 'no_defs': 'None', 'proj': 'geos',
                       'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'},
        'Number of columns': 3712,
        'Number of rows': 3712,
        'Area extent': (5567248.0742, 5570248.4773, -5570248.4773, -5567248.0742)
    }
}

TEST_AREA_EXTENT_EARTHMODEL2_HRV_FULLDISK = {
    'earth_model': 2,
    'dataset_id': make_dataid(name='HRV', resolution=1000),
    'is_full_disk': True,
    'is_rapid_scan': 0,
    'fill_disk': False,
    'expected_area_def': {
        'Area ID': 'msg_seviri_fes_1km',
        'Projection': {'a': '6378169000', 'b': '6356583800', 'h': '35785831',
                       'lon_0': '0', 'no_defs': 'None', 'proj': 'geos',
                       'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'},
        'Number of columns': 5568,
        'Number of rows': 11136,
        'Area extent 0': (5566247.718632221, 2626852.867305279, -2500.3358721733093, -5566247.718632221),
        'Area extent 1': (3600983.723104, 5571248.390376568, -1967764.3314003944, 2626852.867305279)
    }
}

TEST_AREA_EXTENT_EARTHMODEL2_HRV_FULLDISK_FILL = {
    'earth_model': 2,
    'dataset_id': make_dataid(name='HRV', resolution=1000),
    'is_full_disk': True,
    'is_rapid_scan': 0,
    'fill_disk': True,
    'expected_area_def': {
        'Area ID': 'msg_seviri_fes_1km',
        'Projection': {'a': '6378169000', 'b': '6356583800', 'h': '35785831',
                       'lon_0': '0', 'no_defs': 'None', 'proj': 'geos',
                       'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'},
        'Number of columns': 11136,
        'Number of rows': 11136,
        'Area extent': (5566247.718632221, 5571248.390376568, -5571248.390376568, -5566247.718632221)
    }
}

TEST_AREA_EXTENT_EARTHMODEL2_VISIR_RAPIDSCAN = {
    'earth_model': 2,
    'dataset_id': make_dataid(name='VIS006', resolution=3000),
    'is_full_disk': False,
    'is_rapid_scan': 1,
    'fill_disk': False,
    'expected_area_def': {
        'Area ID': 'msg_seviri_rss_3km',
        'Projection': {'a': '6378169000', 'b': '6356583800', 'h': '35785831',
                       'lon_0': '9.5', 'no_defs': 'None', 'proj': 'geos',
                       'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'},
        'Number of columns': 3712,
        'Number of rows': 1392,
        'Area extent': (5567248.074173927, 5570248.477339745, -5570248.477339745, 1393687.2705221176)

    }
}

TEST_AREA_EXTENT_EARTHMODEL2_VISIR_RAPIDSCAN_FILL = {
    'earth_model': 2,
    'dataset_id': make_dataid(name='VIS006', resolution=3000),
    'is_full_disk': False,
    'is_rapid_scan': 1,
    'fill_disk': True,
    'expected_area_def': {
        'Area ID': 'msg_seviri_rss_3km',
        'Projection': {'a': '6378169000', 'b': '6356583800', 'h': '35785831',
                       'lon_0': '9.5', 'no_defs': 'None', 'proj': 'geos',
                       'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'},
        'Number of columns': 3712,
        'Number of rows': 3712,
        'Area extent': (5567248.0742, 5570248.4773, -5570248.4773, -5567248.0742)

    }
}

TEST_AREA_EXTENT_EARTHMODEL2_HRV_RAPIDSCAN = {
    'earth_model': 2,
    'dataset_id': make_dataid(name='HRV', resolution=1000),
    'is_full_disk': False,
    'is_rapid_scan': 1,
    'fill_disk': False,
    'expected_area_def': {
        'Area ID': 'msg_seviri_rss_1km',
        'Projection': {'a': '6378169000', 'b': '6356583800', 'h': '35785831',
                       'lon_0': '0', 'no_defs': 'None', 'proj': 'geos',
                       'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'},
        'Number of columns': 5568,
        'Number of rows': 8192,
        'Area extent': (5566247.718632221, 2626852.867305279, -2500.3358721733093, -5566247.718632221)
    }
}

TEST_AREA_EXTENT_EARTHMODEL2_HRV_RAPIDSCAN_FILL = {
    'earth_model': 2,
    'dataset_id': make_dataid(name='HRV', resolution=1000),
    'is_full_disk': False,
    'is_rapid_scan': 1,
    'fill_disk': True,
    'expected_area_def': {
        'Area ID': 'msg_seviri_rss_1km',
        'Projection': {'a': '6378169000', 'b': '6356583800', 'h': '35785831',
                       'lon_0': '0', 'no_defs': 'None', 'proj': 'geos',
                       'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'},
        'Number of columns': 11136,
        'Number of rows': 11136,
        'Area extent': (5566247.718632221, 5571248.390376568, -5571248.390376568, -5566247.718632221)
    }
}

TEST_AREA_EXTENT_EARTHMODEL2_VISIR_ROI = {
    'earth_model': 2,
    'dataset_id': make_dataid(name='VIS006', resolution=3000),
    'is_full_disk': False,
    'is_rapid_scan': 0,
    'fill_disk': False,
    'expected_area_def': {
        'Area ID': 'msg_seviri_fes_3km',
        'Projection': {'a': '6378169000', 'b': '6356583800', 'h': '35785831',
                       'lon_0': '0', 'no_defs': 'None', 'proj': 'geos',
                       'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'},
        'Number of columns': 2516,
        'Number of rows': 1829,
        'Area extent': (5336217.0304, 5156192.8405, -2212797.3348, -331544.5498)
    }
}

TEST_AREA_EXTENT_EARTHMODEL2_VISIR_ROI_FILL = {
    'earth_model': 2,
    'dataset_id': make_dataid(name='VIS006', resolution=3000),
    'is_full_disk': False,
    'is_rapid_scan': 0,
    'fill_disk': True,
    'expected_area_def': {
        'Area ID': 'msg_seviri_fes_3km',
        'Projection': {'a': '6378169000', 'b': '6356583800', 'h': '35785831',
                       'lon_0': '0', 'no_defs': 'None', 'proj': 'geos',
                       'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'},
        'Number of columns': 3712,
        'Number of rows': 3712,
        'Area extent': (5567248.0742, 5570248.4773, -5570248.4773, -5567248.0742)
    }
}

TEST_AREA_EXTENT_EARTHMODEL2_HRV_ROI = {
    'earth_model': 2,
    'dataset_id': make_dataid(name='HRV', resolution=1000),
    'is_full_disk': False,
    'is_rapid_scan': 0,
    'fill_disk': False,
    'expected_area_def': {
        'Area ID': 'msg_seviri_fes_1km',
        'Projection': {'a': '6378169000', 'b': '6356583800', 'h': '35785831',
                       'lon_0': '0', 'no_defs': 'None', 'proj': 'geos',
                       'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'},
        'Number of columns': 7548,
        'Number of rows': 5487,
        'Area extent': (5335216.684043407, 5157192.769944668, -2213797.381222248, -330544.4023013115)
    }
}

TEST_AREA_EXTENT_EARTHMODEL2_HRV_ROI_FILL = {
    'earth_model': 2,
    'dataset_id': make_dataid(name='HRV', resolution=1000),
    'is_full_disk': False,
    'is_rapid_scan': 0,
    'fill_disk': True,
    'expected_area_def': {
        'Area ID': 'msg_seviri_fes_1km',
        'Projection': {'a': '6378169000', 'b': '6356583800', 'h': '35785831',
                       'lon_0': '0', 'no_defs': 'None', 'proj': 'geos',
                       'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'},
        'Number of columns': 11136,
        'Number of rows': 11136,
        'Area extent': (5566247.718632221, 5571248.390376568, -5571248.390376568, -5566247.718632221)
    }
}

TEST_IS_ROI_FULLDISK = {
    'is_full_disk': True,
    'is_rapid_scan': 0,
    'is_roi': False
}

TEST_IS_ROI_RAPIDSCAN = {
    'is_full_disk': False,
    'is_rapid_scan': 1,
    'is_roi': False
}

TEST_IS_ROI_ROI = {
    'is_full_disk': False,
    'is_rapid_scan': 0,
    'is_roi': True
}

TEST_CALIBRATION_MODE = {
    'earth_model': 1,
    'dataset_id': make_dataid(name='IR_108', calibration='radiance'),
    'is_full_disk': True,
    'is_rapid_scan': 0,
    'calibration': 'radiance',
    'CalSlope': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97],
    'CalOffset': [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
    'GSICSCalCoeff': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97],
    'GSICSOffsetCount': [-51.0, -51.0, -51.0, -51.0, -51.0, -51.0, -51.0, -51.0, -51.0, -51.0, -51.0, -51.0]
}

TEST_PADDER_RSS_ROI = {
    'img_bounds': {'south': [2], 'north': [4], 'east': [2], 'west': [3]},
    'is_full_disk': False,
    'dataset_id': make_dataid(name='VIS006'),
    'dataset': xr.DataArray(np.ones((3, 2)), dims=['y', 'x']).astype(np.float32),
    'final_shape': (5, 5),
    'expected_padded_data': xr.DataArray(np.array([[np.nan, np.nan, np.nan, np.nan, np.nan],
                                                   [np.nan, 1.0, 1.0, np.nan, np.nan],
                                                   [np.nan, 1.0, 1.0, np.nan, np.nan],
                                                   [np.nan, 1.0, 1.0, np.nan, np.nan],
                                                   [np.nan, np.nan, np.nan, np.nan, np.nan]]),
                                         dims=['y', 'x']).astype(np.float32)
}

TEST_PADDER_FES_HRV = {
    'img_bounds': {'south': [1, 4], 'north': [3, 5], 'east': [2, 3], 'west': [3, 4]},
    'is_full_disk': True,
    'dataset_id': make_dataid(name='HRV'),
    'dataset': xr.DataArray(np.ones((5, 2)), dims=['y', 'x']).astype(np.float32),
    'final_shape': (5, 5),
    'expected_padded_data': xr.DataArray(np.array([[np.nan, 1.0, 1.0, np.nan, np.nan],
                                                   [np.nan, 1.0, 1.0, np.nan, np.nan],
                                                   [np.nan, 1.0, 1.0, np.nan, np.nan],
                                                   [np.nan, np.nan, 1.0, 1.0, np.nan],
                                                   [np.nan, np.nan, 1.0, 1.0, np.nan]]),
                                         dims=['y', 'x']).astype(np.float32)

}


class TestNativeMSGFileHandler(unittest.TestCase):
    """Test the NativeMSGFileHandler."""

    def test_get_available_channels(self):
        """Test the derivation of the available channel list."""
        available_chs = get_available_channels(TEST1_HEADER_CHNLIST)
        trues = ['WV_062', 'WV_073', 'IR_108', 'VIS006', 'VIS008', 'IR_120']
        for bandname in AVAILABLE_CHANNELS:
            if bandname in trues:
                self.assertTrue(available_chs[bandname])
            else:
                self.assertFalse(available_chs[bandname])

        available_chs = get_available_channels(TEST2_HEADER_CHNLIST)
        trues = ['VIS006', 'VIS008', 'IR_039', 'WV_062', 'WV_073', 'IR_087', 'HRV']
        for bandname in AVAILABLE_CHANNELS:
            if bandname in trues:
                self.assertTrue(available_chs[bandname])
            else:
                self.assertFalse(available_chs[bandname])

        available_chs = get_available_channels(TEST3_HEADER_CHNLIST)
        for bandname in AVAILABLE_CHANNELS:
            self.assertTrue(available_chs[bandname])


class TestNativeMSGArea(unittest.TestCase):
    """Test NativeMSGFileHandler.get_area_extent.

    The expected results have been verified by manually
    inspecting the output of geoferenced imagery.
    """

    @staticmethod
    def create_test_header(earth_model, dataset_id, is_full_disk, is_rapid_scan, good_qual='OK'):
        """Create mocked NativeMSGFileHandler.

        Contains sufficient attributes for NativeMSGFileHandler.get_area_extent to be able to execute.
        """
        if dataset_id['name'] == 'HRV':
            reference_grid = 'ReferenceGridHRV'
            column_dir_grid_step = 1.0001343488693237
            line_dir_grid_step = 1.0001343488693237
        else:
            reference_grid = 'ReferenceGridVIS_IR'
            column_dir_grid_step = 3.0004031658172607
            line_dir_grid_step = 3.0004031658172607

        if is_full_disk:
            north = 3712
            east = 1
            west = 3712
            south = 1
            n_visir_cols = 3712
            n_visir_lines = 3712
            n_hrv_cols = 11136
            n_hrv_lines = 11136
            ssp_lon = 0
        elif is_rapid_scan:
            north = 3712
            east = 1
            west = 3712
            south = 2321
            n_visir_cols = 3712
            n_visir_lines = 1392
            n_hrv_cols = 11136
            n_hrv_lines = 4176
            ssp_lon = 9.5
        else:
            north = 3574
            east = 78
            west = 2591
            south = 1746
            n_visir_cols = 2516
            n_visir_lines = north - south + 1
            n_hrv_cols = n_visir_cols * 3
            n_hrv_lines = n_visir_lines * 3
            ssp_lon = 0
        header = {
            '15_MAIN_PRODUCT_HEADER': {
                'QQOV': {'Name': 'QQOV',
                         'Value': good_qual}
            },
            '15_DATA_HEADER': {
                'ImageDescription': {
                    reference_grid: {
                        'ColumnDirGridStep': column_dir_grid_step,
                        'LineDirGridStep': line_dir_grid_step,
                        'GridOrigin': 2,  # south-east corner
                    },
                    'ProjectionDescription': {
                        'LongitudeOfSSP': ssp_lon
                    }
                },
                'GeometricProcessing': {
                    'EarthModel': {
                        'TypeOfEarthModel': earth_model,
                        'EquatorialRadius': 6378169.0,
                        'NorthPolarRadius': 6356583.800000001,
                        'SouthPolarRadius': 6356583.800000001,
                    }
                },
                'SatelliteStatus': {
                    'SatelliteDefinition': {
                        'SatelliteId': 324
                    }
                }
            },
            '15_SECONDARY_PRODUCT_HEADER': {
                'NorthLineSelectedRectangle': {'Value': north},
                'EastColumnSelectedRectangle': {'Value': east},
                'WestColumnSelectedRectangle': {'Value': west},
                'SouthLineSelectedRectangle': {'Value': south},
                'SelectedBandIDs': {'Value': 'xxxxxxxxxxxx'},
                'NumberColumnsVISIR': {'Value': n_visir_cols},
                'NumberLinesVISIR': {'Value': n_visir_lines},
                'NumberColumnsHRV': {'Value': n_hrv_cols},
                'NumberLinesHRV': {'Value': n_hrv_lines},
            }

        }

        return header

    @staticmethod
    def create_test_trailer(is_rapid_scan):
        """Create Test Trailer.

        Mocked Trailer with sufficient attributes for
        NativeMSGFileHandler.get_area_extent to be able to execute.
        """
        trailer = {
            '15TRAILER': {
                'ImageProductionStats': {
                    'ActualL15CoverageHRV': {
                        'UpperNorthLineActual': 11136,
                        'UpperWestColumnActual': 7533,
                        'UpperSouthLineActual': 8193,
                        'UpperEastColumnActual': 1966,
                        'LowerNorthLineActual': 8192,
                        'LowerWestColumnActual': 5568,
                        'LowerSouthLineActual': 1,
                        'LowerEastColumnActual': 1
                    },
                    'ActualScanningSummary': {
                        'ReducedScan': is_rapid_scan
                    }
                }
            }
        }

        return trailer

    def prepare_area_defs(self, test_dict):
        """Prepare calculated and expected area definitions for equal checking."""
        earth_model = test_dict['earth_model']
        dataset_id = test_dict['dataset_id']
        is_full_disk = test_dict['is_full_disk']
        is_rapid_scan = test_dict['is_rapid_scan']
        fill_disk = test_dict['fill_disk']
        header = self.create_test_header(earth_model, dataset_id, is_full_disk, is_rapid_scan)
        trailer = self.create_test_trailer(is_rapid_scan)
        expected_area_def = test_dict['expected_area_def']

        with mock.patch('satpy.readers.seviri_l1b_native.np.fromfile') as fromfile, \
                mock.patch('satpy.readers.seviri_l1b_native.recarray2dict') as recarray2dict, \
                mock.patch('satpy.readers.seviri_l1b_native.NativeMSGFileHandler._get_memmap') as _get_memmap, \
                mock.patch('satpy.readers.seviri_l1b_native.NativeMSGFileHandler._read_trailer'), \
                mock.patch(
                    'satpy.readers.seviri_l1b_native.NativeMSGFileHandler._has_archive_header'
                ) as _has_archive_header:
            _has_archive_header.return_value = True
            fromfile.return_value = header
            recarray2dict.side_effect = (lambda x: x)
            _get_memmap.return_value = np.arange(3)
            fh = NativeMSGFileHandler(None, {}, None)
            fh.fill_disk = fill_disk
            fh.header = header
            fh.trailer = trailer
            fh.image_boundaries = ImageBoundaries(header, trailer, fh.mda)
            calc_area_def = fh.get_area_def(dataset_id)

        return (calc_area_def, expected_area_def)

    # Earth model 1 tests
    def test_earthmodel1_visir_fulldisk(self):
        """Test the VISIR FES with the EarthModel 1."""
        calculated, expected = self.prepare_area_defs(
            TEST_AREA_EXTENT_EARTHMODEL1_VISIR_FULLDISK
        )
        np.testing.assert_allclose(np.array(calculated.area_extent),
                                   np.array(expected['Area extent']))
        self.assertEqual(calculated.width, expected['Number of columns'])
        self.assertEqual(calculated.height, expected['Number of rows'])
        self.assertEqual(calculated.area_id, expected['Area ID'])

    def test_earthmodel1_hrv_fulldisk(self):
        """Test the HRV FES with the EarthModel 1."""
        calculated, expected = self.prepare_area_defs(
            TEST_AREA_EXTENT_EARTHMODEL1_HRV_FULLDISK
        )
        np.testing.assert_allclose(np.array(calculated.defs[0].area_extent),
                                   np.array(expected['Area extent 0']))
        np.testing.assert_allclose(np.array(calculated.defs[1].area_extent),
                                   np.array(expected['Area extent 1']))

        self.assertEqual(calculated.width, expected['Number of columns'])
        self.assertEqual(calculated.height, expected['Number of rows'])
        self.assertEqual(calculated.defs[0].area_id, expected['Area ID'])
        self.assertEqual(calculated.defs[1].area_id, expected['Area ID'])

    def test_earthmodel1_hrv_fulldisk_fill(self):
        """Test the HRV FES padded to fulldisk with the EarthModel 1."""
        calculated, expected = self.prepare_area_defs(
            TEST_AREA_EXTENT_EARTHMODEL1_HRV_FULLDISK_FILL
        )
        np.testing.assert_allclose(np.array(calculated.area_extent),
                                   np.array(expected['Area extent']))
        self.assertEqual(calculated.width, expected['Number of columns'])
        self.assertEqual(calculated.height, expected['Number of rows'])
        self.assertEqual(calculated.area_id, expected['Area ID'])

    def test_earthmodel1_visir_rapidscan(self):
        """Test the VISIR RSS with the EarthModel 1."""
        calculated, expected = self.prepare_area_defs(
            TEST_AREA_EXTENT_EARTHMODEL1_VISIR_RAPIDSCAN
        )

        np.testing.assert_allclose(np.array(calculated.area_extent),
                                   np.array(expected['Area extent']))

        self.assertEqual(calculated.width, expected['Number of columns'])
        self.assertEqual(calculated.height, expected['Number of rows'])
        self.assertEqual(calculated.area_id, expected['Area ID'])

    def test_earthmodel1_visir_rapidscan_fill(self):
        """Test the VISIR RSS padded to fulldisk with the EarthModel 1."""
        calculated, expected = self.prepare_area_defs(
            TEST_AREA_EXTENT_EARTHMODEL1_VISIR_RAPIDSCAN_FILL
        )

        np.testing.assert_allclose(np.array(calculated.area_extent),
                                   np.array(expected['Area extent']))

        self.assertEqual(calculated.width, expected['Number of columns'])
        self.assertEqual(calculated.height, expected['Number of rows'])
        self.assertEqual(calculated.area_id, expected['Area ID'])

    def test_earthmodel1_hrv_rapidscan(self):
        """Test the HRV RSS with the EarthModel 1."""
        calculated, expected = self.prepare_area_defs(
            TEST_AREA_EXTENT_EARTHMODEL1_HRV_RAPIDSCAN
        )

        np.testing.assert_allclose(np.array(calculated.area_extent),
                                   np.array(expected['Area extent']))

        self.assertEqual(calculated.width, expected['Number of columns'])
        self.assertEqual(calculated.height, expected['Number of rows'])
        self.assertEqual(calculated.area_id, expected['Area ID'])

    def test_earthmodel1_hrv_rapidscan_fill(self):
        """Test the HRV RSS padded to fulldisk with the EarthModel 1."""
        calculated, expected = self.prepare_area_defs(
            TEST_AREA_EXTENT_EARTHMODEL1_HRV_RAPIDSCAN_FILL
        )

        np.testing.assert_allclose(np.array(calculated.area_extent),
                                   np.array(expected['Area extent']))

        self.assertEqual(calculated.width, expected['Number of columns'])
        self.assertEqual(calculated.height, expected['Number of rows'])
        self.assertEqual(calculated.area_id, expected['Area ID'])

    def test_earthmodel1_visir_roi(self):
        """Test the VISIR ROI with the EarthModel 1."""
        calculated, expected = self.prepare_area_defs(
            TEST_AREA_EXTENT_EARTHMODEL1_VISIR_ROI
        )
        np.testing.assert_allclose(np.array(calculated.area_extent),
                                   np.array(expected['Area extent']))
        self.assertEqual(calculated.width, expected['Number of columns'])
        self.assertEqual(calculated.height, expected['Number of rows'])
        self.assertEqual(calculated.area_id, expected['Area ID'])

    def test_earthmodel1_visir_roi_fill(self):
        """Test the VISIR ROI padded to fulldisk with the EarthModel 1."""
        calculated, expected = self.prepare_area_defs(
            TEST_AREA_EXTENT_EARTHMODEL1_VISIR_ROI_FILL
        )
        np.testing.assert_allclose(np.array(calculated.area_extent),
                                   np.array(expected['Area extent']))
        self.assertEqual(calculated.width, expected['Number of columns'])
        self.assertEqual(calculated.height, expected['Number of rows'])
        self.assertEqual(calculated.area_id, expected['Area ID'])

    def test_earthmodel1_hrv_roi(self):
        """Test the HRV ROI with the EarthModel 1."""
        calculated, expected = self.prepare_area_defs(
            TEST_AREA_EXTENT_EARTHMODEL1_HRV_ROI
        )
        np.testing.assert_allclose(np.array(calculated.area_extent),
                                   np.array(expected['Area extent']))
        self.assertEqual(calculated.width, expected['Number of columns'])
        self.assertEqual(calculated.height, expected['Number of rows'])
        self.assertEqual(calculated.area_id, expected['Area ID'])

    def test_earthmodel1_hrv_roi_fill(self):
        """Test the HRV ROI padded to fulldisk with the EarthModel 1."""
        calculated, expected = self.prepare_area_defs(
            TEST_AREA_EXTENT_EARTHMODEL1_HRV_ROI_FILL
        )
        np.testing.assert_allclose(np.array(calculated.area_extent),
                                   np.array(expected['Area extent']))
        self.assertEqual(calculated.width, expected['Number of columns'])
        self.assertEqual(calculated.height, expected['Number of rows'])
        self.assertEqual(calculated.area_id, expected['Area ID'])

    # Earth model 2 tests
    def test_earthmodel2_visir_fulldisk(self):
        """Test the VISIR FES with the EarthModel 2."""
        calculated, expected = self.prepare_area_defs(
            TEST_AREA_EXTENT_EARTHMODEL2_VISIR_FULLDISK
        )
        np.testing.assert_allclose(np.array(calculated.area_extent),
                                   np.array(expected['Area extent']))
        self.assertEqual(calculated.width, expected['Number of columns'])
        self.assertEqual(calculated.height, expected['Number of rows'])
        self.assertEqual(calculated.area_id, expected['Area ID'])

    def test_earthmodel2_hrv_fulldisk(self):
        """Test the HRV FES with the EarthModel 2."""
        calculated, expected = self.prepare_area_defs(
            TEST_AREA_EXTENT_EARTHMODEL2_HRV_FULLDISK
        )
        np.testing.assert_allclose(np.array(calculated.defs[0].area_extent),
                                   np.array(expected['Area extent 0']))
        np.testing.assert_allclose(np.array(calculated.defs[1].area_extent),
                                   np.array(expected['Area extent 1']))

        self.assertEqual(calculated.width, expected['Number of columns'])
        self.assertEqual(calculated.height, expected['Number of rows'])
        self.assertEqual(calculated.defs[0].area_id, expected['Area ID'])
        self.assertEqual(calculated.defs[1].area_id, expected['Area ID'])

    def test_earthmodel2_hrv_fulldisk_fill(self):
        """Test the HRV FES padded to fulldisk with the EarthModel 2."""
        calculated, expected = self.prepare_area_defs(
            TEST_AREA_EXTENT_EARTHMODEL2_HRV_FULLDISK_FILL
        )
        np.testing.assert_allclose(np.array(calculated.area_extent),
                                   np.array(expected['Area extent']))
        self.assertEqual(calculated.width, expected['Number of columns'])
        self.assertEqual(calculated.height, expected['Number of rows'])
        self.assertEqual(calculated.area_id, expected['Area ID'])

    def test_earthmodel2_visir_rapidscan(self):
        """Test the VISIR RSS with the EarthModel 2."""
        calculated, expected = self.prepare_area_defs(
            TEST_AREA_EXTENT_EARTHMODEL2_VISIR_RAPIDSCAN
        )
        np.testing.assert_allclose(np.array(calculated.area_extent),
                                   np.array(expected['Area extent']))

        self.assertEqual(calculated.width, expected['Number of columns'])
        self.assertEqual(calculated.height, expected['Number of rows'])
        self.assertEqual(calculated.area_id, expected['Area ID'])

    def test_earthmodel2_visir_rapidscan_fill(self):
        """Test the VISIR RSS padded to fulldisk with the EarthModel 2."""
        calculated, expected = self.prepare_area_defs(
            TEST_AREA_EXTENT_EARTHMODEL2_VISIR_RAPIDSCAN_FILL
        )
        np.testing.assert_allclose(np.array(calculated.area_extent),
                                   np.array(expected['Area extent']))

        self.assertEqual(calculated.width, expected['Number of columns'])
        self.assertEqual(calculated.height, expected['Number of rows'])
        self.assertEqual(calculated.area_id, expected['Area ID'])

    def test_earthmodel2_hrv_rapidscan(self):
        """Test the HRV RSS with the EarthModel 2."""
        calculated, expected = self.prepare_area_defs(
            TEST_AREA_EXTENT_EARTHMODEL2_HRV_RAPIDSCAN
        )
        np.testing.assert_allclose(np.array(calculated.area_extent),
                                   np.array(expected['Area extent']))

        self.assertEqual(calculated.width, expected['Number of columns'])
        self.assertEqual(calculated.height, expected['Number of rows'])
        self.assertEqual(calculated.area_id, expected['Area ID'])

    def test_earthmodel2_hrv_rapidscan_fill(self):
        """Test the HRV RSS padded to fulldisk with the EarthModel 2."""
        calculated, expected = self.prepare_area_defs(
            TEST_AREA_EXTENT_EARTHMODEL2_HRV_RAPIDSCAN_FILL
        )
        np.testing.assert_allclose(np.array(calculated.area_extent),
                                   np.array(expected['Area extent']))

        self.assertEqual(calculated.width, expected['Number of columns'])
        self.assertEqual(calculated.height, expected['Number of rows'])
        self.assertEqual(calculated.area_id, expected['Area ID'])

    def test_earthmodel2_visir_roi(self):
        """Test the VISIR ROI with the EarthModel 2."""
        calculated, expected = self.prepare_area_defs(
            TEST_AREA_EXTENT_EARTHMODEL2_VISIR_ROI
        )
        np.testing.assert_allclose(np.array(calculated.area_extent),
                                   np.array(expected['Area extent']))
        self.assertEqual(calculated.width, expected['Number of columns'])
        self.assertEqual(calculated.height, expected['Number of rows'])
        self.assertEqual(calculated.area_id, expected['Area ID'])

    def test_earthmodel2_visir_roi_fill(self):
        """Test the VISIR ROI padded to fulldisk with the EarthModel 2."""
        calculated, expected = self.prepare_area_defs(
            TEST_AREA_EXTENT_EARTHMODEL2_VISIR_ROI_FILL
        )
        np.testing.assert_allclose(np.array(calculated.area_extent),
                                   np.array(expected['Area extent']))
        self.assertEqual(calculated.width, expected['Number of columns'])
        self.assertEqual(calculated.height, expected['Number of rows'])
        self.assertEqual(calculated.area_id, expected['Area ID'])

    def test_earthmodel2_hrv_roi(self):
        """Test the HRV ROI with the EarthModel 2."""
        calculated, expected = self.prepare_area_defs(
            TEST_AREA_EXTENT_EARTHMODEL2_HRV_ROI
        )
        np.testing.assert_allclose(np.array(calculated.area_extent),
                                   np.array(expected['Area extent']))
        self.assertEqual(calculated.width, expected['Number of columns'])
        self.assertEqual(calculated.height, expected['Number of rows'])
        self.assertEqual(calculated.area_id, expected['Area ID'])

    def test_earthmodel2_hrv_roi_fill(self):
        """Test the HRV ROI padded to fulldisk with the EarthModel 2."""
        calculated, expected = self.prepare_area_defs(
            TEST_AREA_EXTENT_EARTHMODEL2_HRV_ROI_FILL
        )
        np.testing.assert_allclose(np.array(calculated.area_extent),
                                   np.array(expected['Area extent']))
        self.assertEqual(calculated.width, expected['Number of columns'])
        self.assertEqual(calculated.height, expected['Number of rows'])
        self.assertEqual(calculated.area_id, expected['Area ID'])

    # Test check for Region Of Interest (ROI) data
    def prepare_is_roi(self, test_dict):
        """Prepare calculated and expected check for region of interest data for equal checking."""
        earth_model = 2
        dataset_id = make_dataid(name='VIS006')
        is_full_disk = test_dict['is_full_disk']
        is_rapid_scan = test_dict['is_rapid_scan']
        header = self.create_test_header(earth_model, dataset_id, is_full_disk, is_rapid_scan)
        trailer = self.create_test_trailer(is_rapid_scan)
        expected_is_roi = test_dict['is_roi']

        with mock.patch('satpy.readers.seviri_l1b_native.np.fromfile') as fromfile, \
                mock.patch('satpy.readers.seviri_l1b_native.recarray2dict') as recarray2dict, \
                mock.patch('satpy.readers.seviri_l1b_native.NativeMSGFileHandler._get_memmap') as _get_memmap, \
                mock.patch('satpy.readers.seviri_l1b_native.NativeMSGFileHandler._read_trailer'), \
                mock.patch(
                    'satpy.readers.seviri_l1b_native.NativeMSGFileHandler._has_archive_header'
                ) as _has_archive_header:
            _has_archive_header.return_value = True
            fromfile.return_value = header
            recarray2dict.side_effect = (lambda x: x)
            _get_memmap.return_value = np.arange(3)
            fh = NativeMSGFileHandler(None, {}, None)
            fh.header = header
            fh.trailer = trailer
            calc_is_roi = fh.is_roi()

        return (calc_is_roi, expected_is_roi)

    def test_is_roi_fulldisk(self):
        """Test check for region of interest with FES data."""
        calculated, expected = self.prepare_is_roi(TEST_IS_ROI_FULLDISK)
        self.assertEqual(calculated, expected)

    def test_is_roi_rapidscan(self):
        """Test check for region of interest with RSS data."""
        calculated, expected = self.prepare_is_roi(TEST_IS_ROI_RAPIDSCAN)
        self.assertEqual(calculated, expected)

    def test_is_roi_roi(self):
        """Test check for region of interest with ROI data."""
        calculated, expected = self.prepare_is_roi(TEST_IS_ROI_ROI)
        self.assertEqual(calculated, expected)


TEST_HEADER_CALIB = {
    'RadiometricProcessing': {
        'Level15ImageCalibration': {
            'CalSlope': TestFileHandlerCalibrationBase.gains_nominal,
            'CalOffset': TestFileHandlerCalibrationBase.offsets_nominal,

        },
        'MPEFCalFeedback': {
            'GSICSCalCoeff': TestFileHandlerCalibrationBase.gains_gsics,
            'GSICSOffsetCount': TestFileHandlerCalibrationBase.offsets_gsics
        }
    },
    'ImageDescription': {
        'Level15ImageProduction': {
            'PlannedChanProcessing': TestFileHandlerCalibrationBase.radiance_types
        }
    },
}


class TestNativeMSGCalibration(TestFileHandlerCalibrationBase):
    """Unit tests for calibration."""

    @pytest.fixture(name='file_handler')
    def file_handler(self):
        """Create a mocked file handler."""
        header = {
            '15_DATA_HEADER': {
                'ImageAcquisition': {
                    'PlannedAcquisitionTime': {
                        'TrueRepeatCycleStart': self.scan_time
                    }
                }
            }
        }
        trailer = {
            '15TRAILER': {
                'ImageProductionStats': {
                    'ActualScanningSummary': {
                        'ForwardScanStart': self.scan_time
                    }
                }
            }
        }
        header['15_DATA_HEADER'].update(TEST_HEADER_CALIB)
        with mock.patch('satpy.readers.seviri_l1b_native.NativeMSGFileHandler.__init__',
                        return_value=None):
            fh = NativeMSGFileHandler()
            fh.header = header
            fh.trailer = trailer
            fh.platform_id = self.platform_id
            return fh

    @pytest.mark.parametrize(
        ('channel', 'calibration', 'calib_mode', 'use_ext_coefs'),
        [
            # VIS channel, internal coefficients
            ('VIS006', 'counts', 'NOMINAL', False),
            ('VIS006', 'radiance', 'NOMINAL', False),
            ('VIS006', 'radiance', 'GSICS', False),
            ('VIS006', 'reflectance', 'NOMINAL', False),
            # VIS channel, external coefficients (mode should have no effect)
            ('VIS006', 'radiance', 'GSICS', True),
            ('VIS006', 'reflectance', 'NOMINAL', True),
            # IR channel, internal coefficients
            ('IR_108', 'counts', 'NOMINAL', False),
            ('IR_108', 'radiance', 'NOMINAL', False),
            ('IR_108', 'radiance', 'GSICS', False),
            ('IR_108', 'brightness_temperature', 'NOMINAL', False),
            ('IR_108', 'brightness_temperature', 'GSICS', False),
            # IR channel, external coefficients (mode should have no effect)
            ('IR_108', 'radiance', 'NOMINAL', True),
            ('IR_108', 'brightness_temperature', 'GSICS', True),
            # HRV channel, internal coefficiens
            ('HRV', 'counts', 'NOMINAL', False),
            ('HRV', 'radiance', 'NOMINAL', False),
            ('HRV', 'radiance', 'GSICS', False),
            ('HRV', 'reflectance', 'NOMINAL', False),
            # HRV channel, external coefficients (mode should have no effect)
            ('HRV', 'radiance', 'GSICS', True),
            ('HRV', 'reflectance', 'NOMINAL', True),
        ]
    )
    def test_calibrate(
            self, file_handler, counts, channel, calibration, calib_mode,
            use_ext_coefs
    ):
        """Test the calibration."""
        external_coefs = self.external_coefs if use_ext_coefs else {}
        expected = self._get_expected(
            channel=channel,
            calibration=calibration,
            calib_mode=calib_mode,
            use_ext_coefs=use_ext_coefs
        )

        fh = file_handler
        fh.calib_mode = calib_mode
        fh.ext_calib_coefs = external_coefs

        dataset_id = make_dataid(name=channel, calibration=calibration)
        res = fh.calibrate(counts, dataset_id)
        xr.testing.assert_allclose(res, expected)


class TestNativeMSGDataset:
    """Tests for getting the dataset."""

    @pytest.fixture
    def file_handler(self):
        """Create a file handler for testing."""
        trailer = {
            '15TRAILER': {
                'ImageProductionStats': {
                    'ActualScanningSummary': {
                        'ForwardScanStart': datetime(2006, 1, 1, 12, 15, 9, 304888),
                        'ForwardScanEnd': datetime(2006, 1, 1, 12, 27, 9, 304888)
                    }
                }
            }
        }
        mda = {
            'channel_list': ['VIS006', 'IR_108'],
            'number_of_lines': 4,
            'number_of_columns': 4,
            'is_full_disk': True,
            'platform_name': 'MSG-3',
            'offset_corrected': True,
            'projection_parameters': {
                'ssp_longitude': 0.0,
                'h': 35785831.0,
                'a': 6378169.0,
                'b': 6356583.8
            }
        }
        header = self._fake_header()
        data = self._fake_data()
        with mock.patch('satpy.readers.seviri_l1b_native.NativeMSGFileHandler.__init__',
                        return_value=None):
            fh = NativeMSGFileHandler()
            fh.header = header
            fh.trailer = trailer
            fh.mda = mda
            fh.dask_array = da.from_array(data)
            fh.platform_id = 324
            fh.fill_disk = False
            fh.calib_mode = 'NOMINAL'
            fh.ext_calib_coefs = {}
            fh.include_raw_metadata = False
            fh.mda_max_array_size = 100
        return fh

    @staticmethod
    def _fake_header():
        header = {
            '15_DATA_HEADER': {
                'SatelliteStatus': {
                    'SatelliteDefinition': {
                        'NominalLongitude': 0.0
                    },
                    'Orbit': {
                        'OrbitPolynomial': ORBIT_POLYNOMIALS
                    }
                },
                'ImageAcquisition': {
                    'PlannedAcquisitionTime': {
                        'TrueRepeatCycleStart': datetime(2006, 1, 1, 12, 15, 0, 0),
                        'PlannedRepeatCycleEnd': datetime(2006, 1, 1, 12, 30, 0, 0),
                    }
                }
            },
        }
        header['15_DATA_HEADER'].update(TEST_HEADER_CALIB)
        return header

    @staticmethod
    def _fake_data():
        num_visir_cols = 5  # will be divided by 1.25 -> 4 columns
        visir_rec = [
            ('line_data', np.uint8, (num_visir_cols,)),
            ('acq_time', time_cds_short)
        ]
        vis006_line1 = (
            [1, 2, 3, 4, 5],  # line_data
            (1, 1000)  # acq_time (days, milliseconds)
        )
        vis006_line2 = ([6, 7, 8, 9, 10], (1, 2000))
        vis006_line3 = ([11, 12, 13, 14, 15], (1, 3000))
        vis006_line4 = ([16, 17, 18, 19, 20], (1, 4000))
        ir108_line1 = ([20, 19, 18, 17, 16], (1, 1000))
        ir108_line2 = ([15, 14, 13, 12, 11], (1, 2000))
        ir108_line3 = ([10, 9, 8, 7, 6], (1, 3000))
        ir108_line4 = ([5, 4, 3, 2, 1], (1, 4000))
        data = np.array(
            [[(vis006_line1,), (ir108_line1,)],
             [(vis006_line2,), (ir108_line2,)],
             [(vis006_line3,), (ir108_line3,)],
             [(vis006_line4,), (ir108_line4,)]],
            dtype=[('visir', visir_rec)]
        )
        return data

    def test_get_dataset(self, file_handler):
        """Test getting the dataset."""
        dataset_id = make_dataid(
            name='VIS006',
            resolution=3000,
            calibration='counts'
        )
        dataset_info = {
            'units': '1',
            'wavelength': (1, 2, 3),
            'standard_name': 'counts'
        }
        dataset = file_handler.get_dataset(dataset_id, dataset_info)
        expected = self._exp_data_array()
        xr.testing.assert_equal(dataset, expected)
        assert 'raw_metadata' not in dataset.attrs
        assert file_handler.start_time == datetime(2006, 1, 1, 12, 15, 0)
        assert file_handler.end_time == datetime(2006, 1, 1, 12, 30, 0)
        assert_attrs_equal(dataset.attrs, expected.attrs, tolerance=1e-4)

    @staticmethod
    def _exp_data_array():
        expected = xr.DataArray(
            np.array([[4., 32., 193., 5.],
                      [24., 112., 514., 266.],
                      [44., 192., 835., 527.],
                      [64., 273., 132., 788.]],
                     dtype=np.float32),
            dims=('y', 'x'),
            attrs={
                'orbital_parameters': {
                    'satellite_actual_longitude': -3.55117540817073,
                    'satellite_actual_latitude': -0.5711243456528018,
                    'satellite_actual_altitude': 35783296.150123544,
                    'satellite_nominal_longitude': 0.0,
                    'satellite_nominal_latitude': 0.0,
                    'projection_longitude': 0.0,
                    'projection_latitude': 0.0,
                    'projection_altitude': 35785831.0
                },
                'time_parameters': {
                    'nominal_start_time': datetime(2006, 1, 1, 12, 15, 0),
                    'nominal_end_time': datetime(2006, 1, 1, 12, 30, 0),
                    'observation_start_time': datetime(2006, 1, 1, 12, 15, 9, 304888),
                    'observation_end_time': datetime(2006, 1, 1, 12, 27, 9, 304888),
                },
                'georef_offset_corrected': True,
                'platform_name': 'MSG-3',
                'sensor': 'seviri',
                'units': '1',
                'wavelength': (1, 2, 3),
                'standard_name': 'counts',
            }
        )
        expected['acq_time'] = ('y', [np.datetime64('1958-01-02 00:00:01'),
                                      np.datetime64('1958-01-02 00:00:02'),
                                      np.datetime64('1958-01-02 00:00:03'),
                                      np.datetime64('1958-01-02 00:00:04')])
        return expected

    def test_get_dataset_with_raw_metadata(self, file_handler):
        """Test provision of raw metadata."""
        file_handler.include_raw_metadata = True
        dataset_id = make_dataid(
            name='VIS006',
            resolution=3000,
            calibration='counts'
        )
        dataset_info = {
            'units': '1',
            'wavelength': (1, 2, 3),
            'standard_name': 'counts'
        }
        res = file_handler.get_dataset(dataset_id, dataset_info)
        assert 'raw_metadata' in res.attrs

    def test_satpos_no_valid_orbit_polynomial(self, file_handler):
        """Test satellite position if there is no valid orbit polynomial."""
        file_handler.header['15_DATA_HEADER']['SatelliteStatus'][
            'Orbit']['OrbitPolynomial'] = ORBIT_POLYNOMIALS_INVALID
        dataset_id = make_dataid(
            name='VIS006',
            resolution=3000,
            calibration='counts'
        )
        dataset_info = {
            'units': '1',
            'wavelength': (1, 2, 3),
            'standard_name': 'counts'
        }
        res = file_handler.get_dataset(dataset_id, dataset_info)
        assert 'satellite_actual_longitude' not in res.attrs[
            'orbital_parameters']


class TestNativeMSGPadder(unittest.TestCase):
    """Test Padder of the native l1b seviri reader."""

    @staticmethod
    def prepare_padder(test_dict):
        """Initialize Padder and pad test data."""
        dataset_id = test_dict['dataset_id']
        img_bounds = test_dict['img_bounds']
        is_full_disk = test_dict['is_full_disk']
        dataset = test_dict['dataset']
        final_shape = test_dict['final_shape']
        expected_padded_data = test_dict['expected_padded_data']

        padder = Padder(dataset_id, img_bounds, is_full_disk)
        padder._final_shape = final_shape
        calc_padded_data = padder.pad_data(dataset)

        return (calc_padded_data, expected_padded_data)

    def test_padder_rss_roi(self):
        """Test padder for RSS and ROI data (applies to both VISIR and HRV)."""
        calculated, expected = self.prepare_padder(TEST_PADDER_RSS_ROI)
        np.testing.assert_array_equal(calculated, expected)

    def test_padder_fes_hrv(self):
        """Test padder for FES HRV data."""
        calculated, expected = self.prepare_padder(TEST_PADDER_FES_HRV)
        np.testing.assert_array_equal(calculated, expected)


class TestNativeMSGFilenames:
    """Test identification of Native format filenames."""

    @pytest.fixture
    def reader(self):
        """Return reader for SEVIRI Native format."""
        from satpy._config import config_search_paths
        from satpy.readers import load_reader

        reader_configs = config_search_paths(
            os.path.join("readers", "seviri_l1b_native.yaml"))
        reader = load_reader(reader_configs)
        return reader

    def test_file_pattern(self, reader):
        """Test file pattern matching."""
        filenames = [
            # Valid
            "MSG2-SEVI-MSG15-0100-NA-20080219094242.289000000Z",
            "MSG2-SEVI-MSG15-0201-NA-20080219094242.289000000Z",
            "MSG2-SEVI-MSG15-0301-NA-20080219094242.289000000Z-123456.nat",
            "MSG2-SEVI-MSG15-0401-NA-20080219094242.289000000Z-20201231181545-123456.nat",
            # Invalid
            "MSG2-SEVI-MSG15-010-NA-20080219094242.289000000Z",
        ]
        files = reader.select_files_from_pathnames(filenames)
        assert len(files) == 4


@pytest.mark.parametrize(
    'file_content,exp_header_size',
    [
        (b'FormatName                  : NATIVE', 450400),  # with ascii header
        (b'foobar', 445286),  # without ascii header
    ]
)
def test_header_type(file_content, exp_header_size):
    """Test identification of the file header type."""
    header = TestNativeMSGArea.create_test_header(
        dataset_id=make_dataid(name='VIS006', resolution=3000),
        earth_model=1,
        is_full_disk=True,
        is_rapid_scan=0
    )
    if file_content == b'foobar':
        header.pop('15_SECONDARY_PRODUCT_HEADER')
    with mock.patch('satpy.readers.seviri_l1b_native.np.fromfile') as fromfile, \
            mock.patch('satpy.readers.seviri_l1b_native.recarray2dict') as recarray2dict, \
            mock.patch('satpy.readers.seviri_l1b_native.NativeMSGFileHandler._get_memmap') as _get_memmap, \
            mock.patch('satpy.readers.seviri_l1b_native.NativeMSGFileHandler._read_trailer'), \
            mock.patch("builtins.open", mock.mock_open(read_data=file_content)):
        fromfile.return_value = header
        recarray2dict.side_effect = (lambda x: x)
        _get_memmap.return_value = np.arange(3)
        fh = NativeMSGFileHandler('myfile', {}, None)
        assert fh.header_type.itemsize == exp_header_size
        assert '15_SECONDARY_PRODUCT_HEADER' in fh.header


def test_header_warning():
    """Test warning is raised for NOK quality flag."""
    header_good = TestNativeMSGArea.create_test_header(
        dataset_id=make_dataid(name='VIS006', resolution=3000),
        earth_model=1,
        is_full_disk=True,
        is_rapid_scan=0,
        good_qual='OK'
    )
    header_bad = TestNativeMSGArea.create_test_header(
        dataset_id=make_dataid(name='VIS006', resolution=3000),
        earth_model=1,
        is_full_disk=True,
        is_rapid_scan=0,
        good_qual='NOK'
    )

    with mock.patch('satpy.readers.seviri_l1b_native.np.fromfile') as fromfile, \
            mock.patch('satpy.readers.seviri_l1b_native.recarray2dict') as recarray2dict, \
            mock.patch('satpy.readers.seviri_l1b_native.NativeMSGFileHandler._get_memmap') as _get_memmap, \
            mock.patch('satpy.readers.seviri_l1b_native.NativeMSGFileHandler._read_trailer'), \
            mock.patch("builtins.open", mock.mock_open(read_data=b'FormatName                  : NATIVE')):
        recarray2dict.side_effect = (lambda x: x)
        _get_memmap.return_value = np.arange(3)

        exp_warning = "The quality flag for this file indicates not OK. Use this data with caution!"

        fromfile.return_value = header_good
        with pytest.warns(None):
            NativeMSGFileHandler('myfile', {}, None)

        fromfile.return_value = header_bad
        with pytest.warns(UserWarning, match=exp_warning):
            NativeMSGFileHandler('myfile', {}, None)
