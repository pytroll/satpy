#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017-2018 PyTroll Community

# Author(s):

#   Adam.Dybbroe <adam.dybbroe@smhi.se>
#   Sauli Joro <sauli.joro@eumetsat.int>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Unittesting the Native SEVIRI reader.
"""

import sys

import numpy as np
import xarray as xr

from satpy.readers.seviri_l1b_native import (
    NativeMSGFileHandler,
    get_available_channels,
)
from satpy.dataset import DatasetID

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

try:
    from unittest import mock
except ImportError:
    import mock


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
    'dsid': DatasetID(name='VIS006'),
    'is_full_disk': True,
    'expected_area_extent': (
        -5568748.275756836, -5568748.275756836,
        5568748.275756836, 5568748.275756836
    )
}

TEST_AREA_EXTENT_EARTHMODEL1_VISIR_ROI = {
    'earth_model': 1,
    'dsid': DatasetID(name='VIS006'),
    'is_full_disk': False,
    'expected_area_extent': (
        -2205296.3268756866, -333044.75140571594,
        5337717.231988907, 5154692.638874054
    )
}

TEST_AREA_EXTENT_EARTHMODEL1_HRV_FULLDISK = None

TEST_AREA_EXTENT_EARTHMODEL1_HRV_ROI = {
    'earth_model': 1,
    'dsid': DatasetID(name='HRV'),
    'is_full_disk': False,
    'expected_area_extent': (
        -2204296.1049079895, -332044.6038246155,
        5336716.885566711, 5153692.299723625
    )
}

TEST_AREA_EXTENT_EARTHMODEL2_VISIR_FULLDISK = {
    'earth_model': 2,
    'dsid': DatasetID(name='VIS006'),
    'is_full_disk': True,
    'expected_area_extent': (
        -5570248.477339745, -5567248.074173927,
        5567248.074173927, 5570248.477339745
    )
}

TEST_AREA_EXTENT_EARTHMODEL2_HRV_FULLDISK = None

TEST_AREA_EXTENT_EARTHMODEL2_VISIR_ROI = {
    'earth_model': 2,
    'dsid': DatasetID(name='VIS006'),
    'is_full_disk': False,
    'expected_area_extent': (
        -2206796.5284585953, -331544.5498228073,
        5336217.030405998, 5156192.840456963
    )
}

TEST_AREA_EXTENT_EARTHMODEL2_HRV_ROI = {
    'earth_model': 2,
    'dsid': DatasetID(name='HRV'),
    'is_full_disk': False,
    'expected_area_extent': (
        -2205796.3064312935, -330544.4023013115,
        5335216.684043407, 5155192.501246929
    )
}

TEST_CALIBRATION_MODE = {
    'earth_model': 1,
    'dsid': DatasetID(name='IR_108', calibration='radiance'),
    'is_full_disk': True,
    'expected_area_extent': (
        -5568748.275756836, -5568748.275756836,
        5568748.275756836, 5568748.275756836
    ),
    'calibration': 'radiance',
    'CalSlope': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97],
    'CalOffset': [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
    'GSICSCalCoeff': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97],
    'GSICSOffsetCount': [-51.0, -51.0, -51.0, -51.0, -51.0, -51.0, -51.0, -51.0, -51.0, -51.0, -51.0, -51.0]
}

# This should preferably be put in a helper-module
# Fixme!


def assertNumpyArraysEqual(self, other):

    if self.shape != other.shape:
        raise AssertionError("Shapes don't match")
    if not np.allclose(self, other):
        raise AssertionError("Elements don't match!")


class TestNativeMSGFileHandler(unittest.TestCase):

    """Test the NativeMSGFileHandler."""

    def setUp(self):
        pass

    def test_get_available_channels(self):
        """Test the derivation of the available channel list"""

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

    def tearDown(self):
        pass


class TestNativeMSGAreaExtent(unittest.TestCase):
    """Test NativeMSGFileHandler.get_area_extent
    The expected results have been verified by manually
    inspecting the output of geoferenced imagery.
    """
    @staticmethod
    def create_test_header(earth_model, dsid, is_full_disk):
        """
        Mocked NativeMSGFileHandler with sufficient attributes for
        NativeMSGFileHandler.get_area_extent to be able to execute.
        """

        if dsid.name == 'HRV':
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
        else:
            north = 3574
            east = 78
            west = 2591
            south = 1746
            n_visir_cols = 2516
            n_visir_lines = north - south + 1

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
                'NumberLinesHRV': {'Value': 11136},
            }

        }

        return header

    def prepare_area_extents(self, test_dict):

        earth_model = test_dict['earth_model']
        dsid = test_dict['dsid']
        is_full_disk = test_dict['is_full_disk']
        header = self.create_test_header(earth_model, dsid, is_full_disk)

        expected_area_extent = (
            np.array(test_dict['expected_area_extent'])
        )

        with mock.patch('satpy.readers.seviri_l1b_native.np.fromfile') as fromfile:
            fromfile.return_value = header
            with mock.patch('satpy.readers.seviri_l1b_native.recarray2dict') as recarray2dict:
                recarray2dict.side_effect = (lambda x: x)
                with mock.patch('satpy.readers.seviri_l1b_native.NativeMSGFileHandler._get_memmap') as _get_memmap:
                    _get_memmap.return_value = np.arange(3)
                    with mock.patch('satpy.readers.seviri_l1b_native.NativeMSGFileHandler._read_trailer'):

                        fh = NativeMSGFileHandler(None, {}, None)
                        fh.header = header
                        calc_area_extent = np.array(
                            fh.get_area_extent(dsid)
                        )

        return (calc_area_extent, expected_area_extent)

    def setUp(self):
        pass

    # Earth model 1 tests
    def test_earthmodel1_visir_fulldisk(self):

        calculated, expected = self.prepare_area_extents(
            TEST_AREA_EXTENT_EARTHMODEL1_VISIR_FULLDISK
        )
        assertNumpyArraysEqual(calculated, expected)

    # def test_earthmodel1_hrv_fulldisk(self):
    #     # Not implemented
    #     pass

    def test_earthmodel1_visir_roi(self):

        calculated, expected = self.prepare_area_extents(
            TEST_AREA_EXTENT_EARTHMODEL1_VISIR_ROI
        )
        assertNumpyArraysEqual(calculated, expected)

    def test_earthmodel1_hrv_roi(self):

        calculated, expected = self.prepare_area_extents(
            TEST_AREA_EXTENT_EARTHMODEL1_HRV_ROI
        )
        assertNumpyArraysEqual(calculated, expected)

    # Earth model 2 tests
    def test_earthmodel2_visir_fulldisk(self):

        calculated, expected = self.prepare_area_extents(
            TEST_AREA_EXTENT_EARTHMODEL2_VISIR_FULLDISK
        )
        assertNumpyArraysEqual(calculated, expected)

    # def test_earthmodel2_hrv_fulldisk(self):
    #     # Not implemented
    #     pass

    def test_earthmodel2_visir_roi(self):

        calculated, expected = self.prepare_area_extents(
            TEST_AREA_EXTENT_EARTHMODEL2_VISIR_ROI
        )
        assertNumpyArraysEqual(calculated, expected)

    def test_earthmodel2_hrv_roi(self):

        calculated, expected = self.prepare_area_extents(
            TEST_AREA_EXTENT_EARTHMODEL2_HRV_ROI
        )
        assertNumpyArraysEqual(calculated, expected)

    def tearDown(self):
        pass


class TestNativeMSGCalibrationMode(unittest.TestCase):
    """Test NativeMSGFileHandler.get_area_extent
    The expected results have been verified by manually
    inspecting the output of geoferenced imagery.
    """
    @staticmethod
    def create_test_header(earth_model, dsid, is_full_disk):
        """
        Mocked NativeMSGFileHandler with sufficient attributes for
        NativeMSGFileHandler.get_area_extent to be able to execute.
        """

        if dsid.name == 'HRV':
            # reference_grid = 'ReferenceGridHRV'
            column_dir_grid_step = 1.0001343488693237
            line_dir_grid_step = 1.0001343488693237
        else:
            # reference_grid = 'ReferenceGridVIS_IR'
            column_dir_grid_step = 3.0004031658172607
            line_dir_grid_step = 3.0004031658172607

        if is_full_disk:
            north = 3712
            east = 1
            west = 3712
            south = 1
            n_visir_cols = 3712
            n_visir_lines = 3712
        else:
            north = 3574
            east = 78
            west = 2591
            south = 1746
            n_visir_cols = 2516
            n_visir_lines = north - south + 1

        header = {
            '15_DATA_HEADER': {
                'ImageDescription': {
                    'reference_grid': {
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
                'RadiometricProcessing': {
                    'Level15ImageCalibration': {
                        'CalSlope': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97],
                        'CalOffset': [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],

                    },
                    'MPEFCalFeedback': {
                        'GSICSCalCoeff': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                                          0.7, 0.8, 0.9, 0.95, 0.96, 0.97],
                        'GSICSOffsetCount': [-51.0, -51.0, -51.0, -51.0, -51.0, -51.0,
                                             -51.0, -51.0, -51.0, -51.0, -51.0, -51.0]
                    },
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
                'NumberLinesHRV': {'Value': 11136},
            }

        }

        return header

    def calibration_mode_test(self, test_dict, cal_mode):
        # dummy data array
        data = xr.DataArray([255., 200., 300.])

        earth_model = test_dict['earth_model']
        dsid = test_dict['dsid']
        index = CHANNEL_INDEX_LIST.index(dsid.name)

        # determine the cal coeffs needed for the expected data calculation
        if cal_mode == 'nominal':
            cal_slope = test_dict['CalSlope'][index]
            cal_offset = test_dict['CalOffset'][index]
        else:
            cal_slope_arr = test_dict['GSICSCalCoeff']
            cal_offset_arr = test_dict['GSICSOffsetCount']
            cal_offset = cal_offset_arr[index] * cal_slope_arr[index]
            cal_slope = cal_slope_arr[index]

        is_full_disk = test_dict['is_full_disk']
        header = self.create_test_header(earth_model, dsid, is_full_disk)

        with mock.patch('satpy.readers.seviri_l1b_native.np.fromfile') as fromfile:
            fromfile.return_value = header
            with mock.patch('satpy.readers.seviri_l1b_native.recarray2dict') as recarray2dict:
                recarray2dict.side_effect = (lambda x: x)
                with mock.patch('satpy.readers.seviri_l1b_native.NativeMSGFileHandler._get_memmap') as _get_memmap:
                    _get_memmap.return_value = np.arange(3)
                    with mock.patch('satpy.readers.seviri_l1b_native.NativeMSGFileHandler._read_trailer'):
                        # Create an instance of the native msg reader
                        # with the calibration mode to test
                        fh = NativeMSGFileHandler(None, {}, None, calib_mode=cal_mode)

                        # Caluculate the expected calibration values using the coeefs
                        # from the test data set
                        expected = fh._convert_to_radiance(data, cal_slope, cal_offset)

                        # Calculate the calibrated vaues using the cal coeffs from the
                        # test header and using the correct calibration mode values
                        fh.header = header
                        calculated = fh.calibrate(data, dsid)

        return (expected.data, calculated.data)

    def setUp(self):
        pass

    def test_calibration_mode_nominal(self):
        # Test using the Nominal calibration mode
        expected, calculated = self.calibration_mode_test(
            TEST_CALIBRATION_MODE,
            'nominal',
        )
        assertNumpyArraysEqual(calculated, expected)

    def test_calibration_mode_gsics(self):
        # Test using the GSICS calibration mode
        expected, calculated = self.calibration_mode_test(
            TEST_CALIBRATION_MODE,
            'gsics',
        )
        assertNumpyArraysEqual(calculated, expected)

    def test_calibration_mode_dummy(self):
        # pass in a calibration mode that is not recognised by the reader
        # and an exception will be raised
        self.assertRaises(NotImplementedError, self.calibration_mode_test,
                          TEST_CALIBRATION_MODE,
                          'dummy',
                          )

    def tearDown(self):
        pass


def suite():
    """The test suite for test_scene.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestNativeMSGFileHandler))
    mysuite.addTest(loader.loadTestsFromTestCase(TestNativeMSGAreaExtent))
    mysuite.addTest(loader.loadTestsFromTestCase(TestNativeMSGCalibrationMode))
    return mysuite


if __name__ == "__main__":
    # So you can run tests from this module individually.
    unittest.main()
