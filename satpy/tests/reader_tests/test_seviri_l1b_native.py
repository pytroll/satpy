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

import unittest
from unittest import mock

import numpy as np
import pytest
import xarray as xr

from satpy.readers.seviri_l1b_native import (
    NativeMSGFileHandler,
    get_available_channels,
)


from satpy.tests.reader_tests.test_seviri_l1b_calibration import (
    TestFileHandlerCalibrationBase
)
from satpy.tests.utils import make_dataid


CHANNEL_INDEX_LIST = ['VIS006', 'VIS008', 'IR_016', 'IR_039',
                      'WV_062', 'WV_073', 'IR_087', 'IR_097',
                      'IR_108', 'IR_120', 'IR_134', 'HRV']
AVAILABLE_CHANNELS = {}
for item in CHANNEL_INDEX_LIST:
    AVAILABLE_CHANNELS[item] = True


SEC15HDR = '15_SECONDARY_PRODUCT_HEADER'
IDS = 'SelectedBandIDs'

TEST1_HEADER_CHNLIST = {SEC15HDR: {IDS: {}}}
TEST1_HEADER_CHNLIST[SEC15HDR][IDS]['Value'] = 'XX--XX--XX--'

TEST2_HEADER_CHNLIST = {SEC15HDR: {IDS: {}}}
TEST2_HEADER_CHNLIST[SEC15HDR][IDS]['Value'] = 'XX-XXXX----X'

TEST3_HEADER_CHNLIST = {SEC15HDR: {IDS: {}}}
TEST3_HEADER_CHNLIST[SEC15HDR][IDS]['Value'] = 'XXXXXXXXXXXX'

TEST_AREA_EXTENT_EARTHMODEL1_VISIR_FULLDISK = {
    'earth_model': 1,
    'dataset_id': make_dataid(name='VIS006'),
    'is_full_disk': True,
    'is_rapid_scan': 0,
    'expected_area_def': {
        'Area ID': 'geos_seviri_visir',
        'Description': 'SEVIRI low resolution channel area',
        'Projection ID': 'seviri_visir',
        'Projection': {'a': '6378169000', 'b': '6356583800', 'h': '35785831',
                       'lon_0': '0', 'no_defs': 'None', 'proj': 'geos',
                       'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'},
        'Number of columns': 3712,
        'Number of rows': 3712,
        'Area extent': (5568748.2758, 5568748.2758, -5568748.2758, -5568748.2758)
    }
}

TEST_AREA_EXTENT_EARTHMODEL1_VISIR_ROI = {
    'earth_model': 1,
    'dataset_id': make_dataid(name='VIS006'),
    'is_full_disk': False,
    'is_rapid_scan': 0,
    'expected_area_def': {
        'Area ID': 'geos_seviri_visir',
        'Description': 'SEVIRI low resolution channel area',
        'Projection ID': 'seviri_visir',
        'Projection': {'a': '6378169000', 'b': '6356583800', 'h': '35785831',
                       'lon_0': '0', 'no_defs': 'None', 'proj': 'geos',
                       'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'},
        'Number of columns': 2516,
        'Number of rows': 1829,
        'Area extent': (5337717.232, 5154692.6389, -2205296.3269, -333044.7514)
    }
}

TEST_AREA_EXTENT_EARTHMODEL1_HRV_FULLDISK = {
    'earth_model': 1,
    'dataset_id': make_dataid(name='HRV'),
    'is_full_disk': True,
    'is_rapid_scan': 0,
    'expected_area_def': {
        'Area ID': 'geos_seviri_hrv',
        'Description': 'SEVIRI low resolution channel area',
        'Projection ID': 'seviri_hrv',
        'Projection': {'a': '6378169000', 'b': '6356583800', 'h': '35785831',
                       'lon_0': '0', 'no_defs': 'None', 'proj': 'geos',
                       'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'},
        'Number of columns': 5568,
        'Number of rows': 11136,
        'Area extent 0': (5567747.920155525, 2625352.665781975, -1000.1343488693237, -5567747.920155525),
        'Area extent 1': (3602483.924627304, 5569748.188853264, -1966264.1298770905, 2625352.665781975)
    }
}

TEST_AREA_EXTENT_EARTHMODEL1_HRV_RAPIDSCAN = {
    'earth_model': 1,
    'dataset_id': make_dataid(name='HRV'),
    'is_full_disk': False,
    'is_rapid_scan': 1,
    'expected_area_def': {
        'Area ID': 'geos_seviri_hrv',
        'Description': 'SEVIRI low resolution channel area',
        'Projection ID': 'seviri_hrv',
        'Projection': {'a': '6378169000', 'b': '6356583800', 'h': '35785831',
                       'lon_0': '0', 'no_defs': 'None', 'proj': 'geos',
                       'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'},
        'Number of columns': 5568,
        'Number of rows': 4176,
        'Area extent': (5567747.920155525, 2625352.665781975, -1000.1343488693237, -5567747.920155525)
    }
}

TEST_AREA_EXTENT_EARTHMODEL1_HRV_ROI = {
    'earth_model': 1,
    'dataset_id': make_dataid(name='HRV'),
    'is_full_disk': False,
    'is_rapid_scan': 0,
    'expected_area_def': {
        'Area ID': 'geos_seviri_hrv',
        'Description': 'SEVIRI high resolution channel area',
        'Projection ID': 'seviri_hrv',
        'Projection': {'a': '6378169000', 'b': '6356583800', 'h': '35785831',
                       'lon_0': '0', 'no_defs': 'None', 'proj': 'geos',
                       'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'},
        'Number of columns': 11136,
        'Number of rows': 11136,
        'Area extent': (5334716.616868973, 5155692.568421364, -2206296.373605728, -330044.33512687683)
    }
}

TEST_AREA_EXTENT_EARTHMODEL2_VISIR_FULLDISK = {
    'earth_model': 2,
    'dataset_id': make_dataid(name='VIS006'),
    'is_full_disk': True,
    'is_rapid_scan': 0,
    'expected_area_def': {
        'Area ID': 'geos_seviri_visir',
        'Description': 'SEVIRI low resolution channel area',
        'Projection ID': 'seviri_visir',
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
    'dataset_id': make_dataid(name='HRV'),
    'is_full_disk': True,
    'is_rapid_scan': 0,
    'expected_area_def': {
        'Area ID': 'geos_seviri_hrv',
        'Description': 'SEVIRI low resolution channel area',
        'Projection ID': 'seviri_hrv',
        'Projection': {'a': '6378169000', 'b': '6356583800', 'h': '35785831',
                       'lon_0': '0', 'no_defs': 'None', 'proj': 'geos',
                       'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'},
        'Number of columns': 5568,
        'Number of rows': 11136,
        'Area extent 0': (5566247.718632221, 2626852.867305279, -2500.3358721733093, -5566247.718632221),
        'Area extent 1': (3600983.723104, 5571248.390376568, -1967764.3314003944, 2626852.867305279)
    }
}

TEST_AREA_EXTENT_EARTHMODEL2_HRV_RAPIDSCAN = {
    'earth_model': 2,
    'dataset_id': make_dataid(name='HRV'),
    'is_full_disk': False,
    'is_rapid_scan': 1,
    'expected_area_def': {
        'Area ID': 'geos_seviri_hrv',
        'Description': 'SEVIRI low resolution channel area',
        'Projection ID': 'seviri_hrv',
        'Projection': {'a': '6378169000', 'b': '6356583800', 'h': '35785831',
                       'lon_0': '0', 'no_defs': 'None', 'proj': 'geos',
                       'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'},
        'Number of columns': 5568,
        'Number of rows': 4176,
        'Area extent': (5566247.718632221, 2626852.867305279, -2500.3358721733093, -5566247.718632221)
    }
}

TEST_AREA_EXTENT_EARTHMODEL2_VISIR_ROI = {
    'earth_model': 2,
    'dataset_id': make_dataid(name='VIS006'),
    'is_full_disk': False,
    'is_rapid_scan': 0,
    'expected_area_def': {
        'Area ID': 'geos_seviri_visir',
        'Description': 'SEVIRI low resolution channel area',
        'Projection ID': 'seviri_visir',
        'Projection': {'a': '6378169000', 'b': '6356583800', 'h': '35785831',
                       'lon_0': '0', 'no_defs': 'None', 'proj': 'geos',
                       'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'},
        'Number of columns': 2516,
        'Number of rows': 1829,
        'Area extent': (5336217.0304, 5156192.8405, -2206796.5285, -331544.5498)
    }
}

TEST_AREA_EXTENT_EARTHMODEL2_HRV_ROI = {
    'earth_model': 2,
    'dataset_id': make_dataid(name='HRV'),
    'is_full_disk': False,
    'is_rapid_scan': 0,
    'expected_area_def': {
        'Area ID': 'geos_seviri_hrv',
        'Description': 'SEVIRI high resolution channel area',
        'Projection ID': 'seviri_hrv',
        'Projection': {'a': '6378169000', 'b': '6356583800', 'h': '35785831',
                       'lon_0': '0', 'no_defs': 'None', 'proj': 'geos',
                       'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'},
        'Number of columns': 11136,
        'Number of rows': 11136,
        'Area extent': (5333216.415345669, 5157192.769944668, -2207796.575129032, -328544.13360357285)
    }
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

# This should preferably be put in a helper-module
# Fixme!


def assertNumpyArraysEqual(self, other):
    """Assert that Numpy arrays are equal."""
    if self.shape != other.shape:
        raise AssertionError("Shapes don't match")
    if not np.allclose(self, other):
        raise AssertionError("Elements don't match!")


class TestNativeMSGFileHandler(unittest.TestCase):
    """Test the NativeMSGFileHandler."""

    def test_get_available_channels(self):
        """Test the derivation of the available channel list."""
        available_chs = get_available_channels(TEST1_HEADER_CHNLIST)
        trues = ['WV_062', 'WV_073', 'IR_108', 'VIS006', 'VIS008', 'IR_120']
        for bandname in AVAILABLE_CHANNELS.keys():
            if bandname in trues:
                self.assertTrue(available_chs[bandname])
            else:
                self.assertFalse(available_chs[bandname])

        available_chs = get_available_channels(TEST2_HEADER_CHNLIST)
        trues = ['VIS006', 'VIS008', 'IR_039', 'WV_062', 'WV_073', 'IR_087', 'HRV']
        for bandname in AVAILABLE_CHANNELS.keys():
            if bandname in trues:
                self.assertTrue(available_chs[bandname])
            else:
                self.assertFalse(available_chs[bandname])

        available_chs = get_available_channels(TEST3_HEADER_CHNLIST)
        for bandname in AVAILABLE_CHANNELS.keys():
            self.assertTrue(available_chs[bandname])


class TestNativeMSGArea(unittest.TestCase):
    """Test NativeMSGFileHandler.get_area_extent.

    The expected results have been verified by manually
    inspecting the output of geoferenced imagery.
    """

    @staticmethod
    def create_test_header(earth_model, dataset_id, is_full_disk, is_rapid_scan):
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
            n_hrv_lines = 11136
        elif is_rapid_scan:
            north = 3712
            east = 1
            west = 3712
            south = 2321
            n_visir_cols = 3712
            n_visir_lines = 1392
            n_hrv_lines = 4176
        else:
            north = 3574
            east = 78
            west = 2591
            south = 1746
            n_visir_cols = 2516
            n_visir_lines = north - south + 1
            n_hrv_lines = 11136

        header = {
            '15_DATA_HEADER': {
                'ImageDescription': {
                    reference_grid: {
                        'ColumnDirGridStep': column_dir_grid_step,
                        'LineDirGridStep': line_dir_grid_step,
                        'GridOrigin': 2,  # south-east corner
                    },
                    'ProjectionDescription': {
                        'LongitudeOfSSP': 0.0
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
                'NumberColumnsHRV': {'Value': 11136},
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
        header = self.create_test_header(earth_model, dataset_id, is_full_disk, is_rapid_scan)
        trailer = self.create_test_trailer(is_rapid_scan)
        expected_area_def = test_dict['expected_area_def']

        with mock.patch('satpy.readers.seviri_l1b_native.np.fromfile') as fromfile:
            fromfile.return_value = header
            with mock.patch('satpy.readers.seviri_l1b_native.recarray2dict') as recarray2dict:
                recarray2dict.side_effect = (lambda x: x)
                with mock.patch('satpy.readers.seviri_l1b_native.NativeMSGFileHandler._get_memmap') as _get_memmap:
                    _get_memmap.return_value = np.arange(3)
                    with mock.patch('satpy.readers.seviri_l1b_native.NativeMSGFileHandler._read_trailer'):

                        fh = NativeMSGFileHandler(None, {}, None)
                        fh.header = header
                        fh.trailer = trailer
                        calc_area_def = fh.get_area_def(dataset_id)

        return (calc_area_def, expected_area_def)

    # Earth model 1 tests
    def test_earthmodel1_visir_fulldisk(self):
        """Test the VISIR Fulldisk with the EarthModel 1."""
        calculated, expected = self.prepare_area_defs(
            TEST_AREA_EXTENT_EARTHMODEL1_VISIR_FULLDISK
        )
        assertNumpyArraysEqual(np.array(calculated.area_extent),
                               np.array(expected['Area extent']))
        self.assertEqual(calculated.width, expected['Number of columns'])
        self.assertEqual(calculated.height, expected['Number of rows'])
        self.assertEqual(calculated.proj_id,  expected['Projection ID'])

    def test_earthmodel1_hrv_fulldisk(self):
        """Test the HRV Fulldisk with the EarthModel 1."""
        calculated, expected = self.prepare_area_defs(
            TEST_AREA_EXTENT_EARTHMODEL1_HRV_FULLDISK
        )
        assertNumpyArraysEqual(np.array(calculated.defs[0].area_extent),
                               np.array(expected['Area extent 0']))
        assertNumpyArraysEqual(np.array(calculated.defs[1].area_extent),
                               np.array(expected['Area extent 1']))

        self.assertEqual(calculated.width, expected['Number of columns'])
        self.assertEqual(calculated.height, expected['Number of rows'])
        self.assertEqual(calculated.defs[0].proj_id, expected['Projection ID'])
        self.assertEqual(calculated.defs[1].proj_id, expected['Projection ID'])

    def test_earthmodel1_hrv_rapidscan(self):
        """Test the HRV Fulldisk with the EarthModel 1."""
        calculated, expected = self.prepare_area_defs(
            TEST_AREA_EXTENT_EARTHMODEL1_HRV_RAPIDSCAN
        )

        assertNumpyArraysEqual(np.array(calculated.area_extent),
                               np.array(expected['Area extent']))

        self.assertEqual(calculated.width, expected['Number of columns'])
        self.assertEqual(calculated.height, expected['Number of rows'])
        self.assertEqual(calculated.proj_id, expected['Projection ID'])

    def test_earthmodel1_visir_roi(self):
        """Test the VISIR ROI with the EarthModel 1."""
        calculated, expected = self.prepare_area_defs(
            TEST_AREA_EXTENT_EARTHMODEL1_VISIR_ROI
        )
        assertNumpyArraysEqual(np.array(calculated.area_extent),
                               np.array(expected['Area extent']))
        self.assertEqual(calculated.width, expected['Number of columns'])
        self.assertEqual(calculated.height, expected['Number of rows'])
        self.assertEqual(calculated.proj_id,  expected['Projection ID'])

    def test_earthmodel1_hrv_roi(self):
        """Test the HRV ROI with the EarthModel 1."""
        calculated, expected = self.prepare_area_defs(
            TEST_AREA_EXTENT_EARTHMODEL1_HRV_ROI
        )
        assertNumpyArraysEqual(np.array(calculated.area_extent),
                               np.array(expected['Area extent']))
        self.assertEqual(calculated.width, expected['Number of columns'])
        self.assertEqual(calculated.height, expected['Number of rows'])
        self.assertEqual(calculated.proj_id,  expected['Projection ID'])

    # Earth model 2 tests
    def test_earthmodel2_visir_fulldisk(self):
        """Test the VISIR Fulldisk with the EarthModel 2."""
        calculated, expected = self.prepare_area_defs(
            TEST_AREA_EXTENT_EARTHMODEL2_VISIR_FULLDISK
        )
        assertNumpyArraysEqual(np.array(calculated.area_extent),
                               np.array(expected['Area extent']))
        self.assertEqual(calculated.width, expected['Number of columns'])
        self.assertEqual(calculated.height, expected['Number of rows'])
        self.assertEqual(calculated.proj_id,  expected['Projection ID'])

    def test_earthmodel2_hrv_fulldisk(self):
        """Test the HRV Fulldisk with the EarthModel 2."""
        calculated, expected = self.prepare_area_defs(
            TEST_AREA_EXTENT_EARTHMODEL2_HRV_FULLDISK
        )
        assertNumpyArraysEqual(np.array(calculated.defs[0].area_extent), np.array(expected['Area extent 0']))
        assertNumpyArraysEqual(np.array(calculated.defs[1].area_extent), np.array(expected['Area extent 1']))

        self.assertEqual(calculated.width, expected['Number of columns'])
        self.assertEqual(calculated.height, expected['Number of rows'])
        self.assertEqual(calculated.defs[0].proj_id,  expected['Projection ID'])
        self.assertEqual(calculated.defs[1].proj_id,  expected['Projection ID'])

    def test_earthmodel2_hrv_rapidscan(self):
        """Test the HRV Fulldisk with the EarthModel 1."""
        calculated, expected = self.prepare_area_defs(
            TEST_AREA_EXTENT_EARTHMODEL2_HRV_RAPIDSCAN
        )
        assertNumpyArraysEqual(np.array(calculated.area_extent),
                               np.array(expected['Area extent']))

        self.assertEqual(calculated.width, expected['Number of columns'])
        self.assertEqual(calculated.height, expected['Number of rows'])
        self.assertEqual(calculated.proj_id, expected['Projection ID'])

    def test_earthmodel2_visir_roi(self):
        """Test the VISIR ROI with the EarthModel 2."""
        calculated, expected = self.prepare_area_defs(
            TEST_AREA_EXTENT_EARTHMODEL2_VISIR_ROI
        )
        assertNumpyArraysEqual(np.array(calculated.area_extent),
                               np.array(expected['Area extent']))
        self.assertEqual(calculated.width, expected['Number of columns'])
        self.assertEqual(calculated.height, expected['Number of rows'])
        self.assertEqual(calculated.proj_id,  expected['Projection ID'])

    def test_earthmodel2_hrv_roi(self):
        """Test the HRV ROI with the EarthModel 2."""
        calculated, expected = self.prepare_area_defs(
            TEST_AREA_EXTENT_EARTHMODEL2_HRV_ROI
        )
        assertNumpyArraysEqual(np.array(calculated.area_extent),
                               np.array(expected['Area extent']))
        self.assertEqual(calculated.width, expected['Number of columns'])
        self.assertEqual(calculated.height, expected['Number of rows'])
        self.assertEqual(calculated.proj_id,  expected['Projection ID'])


class TestNativeMSGCalibration(TestFileHandlerCalibrationBase):
    """Unit tests for calibration."""

    @pytest.fixture(name='file_handler')
    def file_handler(self):
        """Create a mocked file handler."""
        header = {
            '15_DATA_HEADER': {
                'RadiometricProcessing': {
                    'Level15ImageCalibration': {
                        'CalSlope': self.gains_nominal,
                        'CalOffset': self.offsets_nominal,

                    },
                    'MPEFCalFeedback': {
                        'GSICSCalCoeff': self.gains_gsics,
                        'GSICSOffsetCount': self.offsets_gsics
                    }
                },
                'ImageDescription': {
                    'Level15ImageProduction': {
                        'PlannedChanProcessing': self.radiance_types
                    }
                },
                'ImageAcquisition': {
                    'PlannedAcquisitionTime': {
                        'TrueRepeatCycleStart': self.scan_time
                    }
                }
            }
        }
        with mock.patch('satpy.readers.seviri_l1b_native.NativeMSGFileHandler.__init__',
                        return_value=None):
            fh = NativeMSGFileHandler()
            fh.header = header
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
