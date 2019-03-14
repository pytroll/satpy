# -*- coding: utf-8 -*-

# Copyright (c) 2017-2018 Martin Raspaud

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

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
"""The HRIT msg reader tests package.
"""

import sys
from datetime import datetime

import numpy as np
import xarray as xr

from satpy.readers.seviri_l1b_hrit import (HRITMSGFileHandler, HRITMSGPrologueFileHandler,
                                           HRITMSGEpilogueFileHandler)
from satpy.readers.seviri_base import CHANNEL_NAMES, VIS_CHANNELS
from satpy.dataset import DatasetID

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

try:
    from unittest import mock
except ImportError:
    import mock


def new_get_hd(instance, hdr_info):
    instance.mda = {'spectral_channel_id': 'bla'}
    instance.mda.setdefault('number_of_bits_per_pixel', 10)

    instance.mda['projection_parameters'] = {'a': 6378169.00,
                                             'b': 6356583.80,
                                             'h': 35785831.00,
                                             'SSP_longitude': 0.0}
    instance.mda['total_header_length'] = 12


class TestHRITMSGFileHandler(unittest.TestCase):
    """Test the HRITFileHandler."""

    @mock.patch('satpy.readers.seviri_l1b_hrit.np.fromfile')
    def setUp(self, fromfile):
        """Setup the hrit file handler for testing."""
        m = mock.mock_open()
        fromfile.return_value = np.array([(1, 2)], dtype=[('total_header_length', int),
                                                          ('hdr_id', int)])

        with mock.patch('satpy.readers.hrit_base.open', m, create=True) as newopen:
            with mock.patch('satpy.readers.seviri_l1b_hrit.CHANNEL_NAMES'):
                with mock.patch.object(HRITMSGFileHandler, '_get_hd', new=new_get_hd):
                    newopen.return_value.__enter__.return_value.tell.return_value = 1
                    prologue = mock.MagicMock()
                    prologue.prologue = {"SatelliteStatus": {"SatelliteDefinition": {"SatelliteId": 324}},
                                         'GeometricProcessing': {'EarthModel': {'TypeOfEarthModel': 2,
                                                                                'NorthPolarRadius': 10,
                                                                                'SouthPolarRadius': 10,
                                                                                'EquatorialRadius': 10}},
                                         'ImageDescription': {'ProjectionDescription': {'LongitudeOfSSP': 0.0}}}
                    self.reader = HRITMSGFileHandler(
                        'filename',
                        {'platform_shortname': 'MSG3',
                         'start_time': datetime(2016, 3, 3, 0, 0),
                         'service': 'MSG'},
                        {'filetype': 'info'},
                        prologue,
                        mock.MagicMock())
                    ncols = 3712
                    nlines = 464
                    nbits = 10
                    self.reader.mda['number_of_bits_per_pixel'] = nbits
                    self.reader.mda['number_of_lines'] = nlines
                    self.reader.mda['number_of_columns'] = ncols
                    self.reader.mda['data_field_length'] = nlines * ncols * nbits
                    self.reader.mda['cfac'] = 5
                    self.reader.mda['lfac'] = 5
                    self.reader.mda['coff'] = 10
                    self.reader.mda['loff'] = 10
                    self.reader.mda['projection_parameters'] = {}
                    self.reader.mda['projection_parameters']['a'] = 6378169.0
                    self.reader.mda['projection_parameters']['b'] = 6356583.8
                    self.reader.mda['projection_parameters']['h'] = 35785831.0
                    self.reader.mda['projection_parameters']['SSP_longitude'] = 44

    def test_get_xy_from_linecol(self):
        """Test get_xy_from_linecol."""
        x__, y__ = self.reader.get_xy_from_linecol(0, 0, (10, 10), (5, 5))
        self.assertEqual(-131072, x__)
        self.assertEqual(131072, y__)
        x__, y__ = self.reader.get_xy_from_linecol(10, 10, (10, 10), (5, 5))
        self.assertEqual(0, x__)
        self.assertEqual(0, y__)
        x__, y__ = self.reader.get_xy_from_linecol(20, 20, (10, 10), (5, 5))
        self.assertEqual(131072, x__)
        self.assertEqual(-131072, y__)

    def test_get_area_extent(self):
        res = self.reader.get_area_extent((20, 20), (10, 10), (5, 5), 33)
        exp = (-71717.44995740513, -79266.655216079365,
               79266.655216079365, 71717.44995740513)
        self.assertTupleEqual(res, exp)

    def test_get_area_def(self):
        area = self.reader.get_area_def(DatasetID('VIS006'))
        self.assertEqual(area.proj_dict, {'a': 6378169.0,
                                          'b': 6356583.8,
                                          'h': 35785831.0,
                                          'lon_0': 44.0,
                                          'proj': 'geos',
                                          'units': 'm'})
        self.assertEqual(area.area_extent,
                         (-77771774058.38356, -3720765401003.719,
                          30310525626438.438, 77771774058.38356))

    @mock.patch('satpy.readers.hrit_base.np.memmap')
    def test_read_band(self, memmap):
        nbits = self.reader.mda['number_of_bits_per_pixel']
        memmap.return_value = np.random.randint(0, 256,
                                                size=int((464 * 3712 * nbits) / 8),
                                                dtype=np.uint8)
        res = self.reader.read_band('VIS006', None)
        self.assertEqual(res.compute().shape, (464, 3712))

    @mock.patch('satpy.readers.hrit_base.HRITFileHandler.__init__', return_value=None)
    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITMSGFileHandler._get_header', autospec=True)
    @mock.patch('satpy.readers.seviri_base.SEVIRICalibrationHandler._convert_to_radiance')
    def test_calibrate(self, _convert_to_radiance, get_header, *mocks):
        """Test selection of calibration coefficients"""
        shp = (10, 10)
        counts = xr.DataArray(np.zeros(shp))
        nominal_gain = np.arange(1, 13)
        nominal_offset = np.arange(-1, -13, -1)
        gsics_gain = np.arange(0.1, 1.3, 0.1)
        gsics_offset = np.arange(-0.1, -1.3, -0.1)

        # Mock prologue & epilogue
        pro = mock.MagicMock(prologue={'RadiometricProcessing': {
            'Level15ImageCalibration': {'CalSlope': nominal_gain,
                                        'CalOffset': nominal_offset},
            'MPEFCalFeedback': {'GSICSCalCoeff': gsics_gain,
                                'GSICSOffsetCount': gsics_offset}
        }})
        epi = mock.MagicMock(epilogue=None)

        # Mock header readout
        mda = {'image_segment_line_quality': {'line_validity': np.zeros(shp[0]),
                                              'line_radiometric_quality': np.zeros(shp[0]),
                                              'line_geometric_quality': np.zeros(shp[0])}}

        def get_header_patched(self):
            self.mda = mda

        get_header.side_effect = get_header_patched

        # Test selection of calibration coefficients
        #
        # a) Default: Nominal calibration
        reader = HRITMSGFileHandler(filename=None, filename_info=None, filetype_info=None,
                                    prologue=pro, epilogue=epi)
        for ch_id, ch_name in CHANNEL_NAMES.items():
            reader.channel_name = ch_name
            reader.mda['spectral_channel_id'] = ch_id
            reader.calibrate(data=counts, calibration='radiance')
            _convert_to_radiance.assert_called_with(mock.ANY, nominal_gain[ch_id - 1],
                                                    nominal_offset[ch_id - 1])

        # b) GSICS calibration for IR channels, nominal calibration for VIS channels
        reader = HRITMSGFileHandler(filename=None, filename_info=None, filetype_info=None,
                                    prologue=pro, epilogue=epi, calib_mode='GSICS')
        for ch_id, ch_name in CHANNEL_NAMES.items():
            if ch_name in VIS_CHANNELS:
                gain, offset = nominal_gain[ch_id - 1], nominal_offset[ch_id - 1]
            else:
                gain, offset = gsics_gain[ch_id - 1], gsics_offset[ch_id - 1]

            reader.channel_name = ch_name
            reader.mda['spectral_channel_id'] = ch_id
            reader.calibrate(data=counts, calibration='radiance')
            _convert_to_radiance.assert_called_with(mock.ANY, gain, offset)

        # c) External calibration coefficients for selected channels, GSICS coefs for remaining
        #    IR channels, nominal coefs for remaining VIS channels
        coefs = {'VIS006': {'gain': 1.234, 'offset': -0.1},
                 'IR_108': {'gain': 2.345, 'offset': -0.2}}
        reader = HRITMSGFileHandler(filename=None, filename_info=None, filetype_info=None,
                                    prologue=pro, epilogue=epi, ext_calib_coefs=coefs,
                                    calib_mode='GSICS')
        for ch_id, ch_name in CHANNEL_NAMES.items():
            if ch_name in coefs.keys():
                gain, offset = coefs[ch_name]['gain'], coefs[ch_name]['offset']
            elif ch_name not in VIS_CHANNELS:
                gain, offset = gsics_gain[ch_id - 1], gsics_offset[ch_id - 1]
            else:
                gain, offset = nominal_gain[ch_id - 1], nominal_offset[ch_id - 1]

            reader.channel_name = ch_name
            reader.mda['spectral_channel_id'] = ch_id
            reader.calibrate(data=counts, calibration='radiance')
            _convert_to_radiance.assert_called_with(mock.ANY, gain, offset)

        # d) Invalid mode
        self.assertRaises(ValueError, HRITMSGFileHandler, filename=None, filename_info=None,
                          filetype_info=None, prologue=pro, epilogue=epi, calib_mode='invalid')


class TestHRITMSGPrologueFileHandler(unittest.TestCase):
    """Test the HRIT prologue file handler."""

    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITMSGPrologueFileHandler.read_prologue')
    @mock.patch('satpy.readers.hrit_base.HRITFileHandler.__init__', autospec=True)
    def test_calibrate(self, init, *mocks):
        """Test whether the prologue file handler accepts extra calibration keywords"""
        def init_patched(self, *args, **kwargs):
            self.mda = {}
        init.side_effect = init_patched

        HRITMSGPrologueFileHandler(filename=None,
                                   filename_info={'service': ''},
                                   filetype_info=None,
                                   ext_calib_coefs={},
                                   calib_mode='nominal')


class TestHRITMSGEpilogueFileHandler(unittest.TestCase):
    """Test the HRIT epilogue file handler."""

    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITMSGEpilogueFileHandler.read_epilogue')
    @mock.patch('satpy.readers.hrit_base.HRITFileHandler.__init__', autospec=True)
    def test_calibrate(self, init, *mocks):
        """Test whether the epilogue file handler accepts extra calibration keywords"""

        def init_patched(self, *args, **kwargs):
            self.mda = {}

        init.side_effect = init_patched

        HRITMSGEpilogueFileHandler(filename=None,
                                   filename_info={'service': ''},
                                   filetype_info=None,
                                   ext_calib_coefs={},
                                   calib_mode='nominal')


def suite():
    """The test suite for test_scene.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    tests = [TestHRITMSGFileHandler, TestHRITMSGPrologueFileHandler, TestHRITMSGEpilogueFileHandler]
    for test in tests:
        mysuite.addTest(loader.loadTestsFromTestCase(test))
    return mysuite


if __name__ == '__main__':
    unittest.main()
