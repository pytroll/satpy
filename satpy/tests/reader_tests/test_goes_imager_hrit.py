#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
"""The hrit msg reader tests package."""

import unittest
import datetime
import numpy as np
from unittest import mock
from xarray import DataArray
from satpy.readers.goes_imager_hrit import (make_gvar_float, make_sgs_time,
                                            HRITGOESPrologueFileHandler, sgs_time,
                                            HRITGOESFileHandler, ALTITUDE)


class TestGVARFloat(unittest.TestCase):
    def test_fun(self):
        test_data = [(-1.0, b"\xbe\xf0\x00\x00"),
                     (-0.1640625, b"\xbf\xd6\x00\x00"),
                     (0.0, b"\x00\x00\x00\x00"),
                     (0.1640625, b"\x40\x2a\x00\x00"),
                     (1.0, b"\x41\x10\x00\x00"),
                     (100.1640625, b"\x42\x64\x2a\x00")]

        for expected, str_val in test_data:
            val = np.frombuffer(str_val, dtype='>i4')
            self.assertEqual(expected, make_gvar_float(val))


class TestMakeSGSTime(unittest.TestCase):
    def test_fun(self):
        # 2018-129 (may 9th), 21:33:27.999
        tcds = np.array([(32, 24, 18, 146, 19, 50, 121, 153)], dtype=sgs_time)
        expected = datetime.datetime(2018, 5, 9, 21, 33, 27, 999000)
        self.assertEqual(make_sgs_time(tcds[0]), expected)


test_pro = {'TISTR': datetime.datetime(2018, 5, 9, 21, 33, 27, 999000),
            'TCurr': datetime.datetime(2018, 5, 9, 21, 33, 27, 999000),
            'TCLMT': datetime.datetime(2018, 5, 9, 21, 33, 27, 999000),
            'SubSatLongitude': 100.1640625,
            'TCHED': datetime.datetime(2018, 5, 9, 21, 33, 27, 999000),
            'TLTRL': datetime.datetime(2018, 5, 9, 21, 33, 27, 999000),
            'TIPFS': datetime.datetime(2018, 5, 9, 21, 33, 27, 999000),
            'TISPC': datetime.datetime(2018, 5, 9, 21, 33, 27, 999000),
            'ReferenceLatitude': 0.0,
            'TIIRT': datetime.datetime(2018, 5, 9, 21, 33, 27, 999000),
            'TLHED': datetime.datetime(2018, 5, 9, 21, 33, 27, 999000),
            'TIVIT': datetime.datetime(2018, 5, 9, 21, 33, 27, 999000),
            'SubSatLatitude': 0.0,
            'TIECL': datetime.datetime(2018, 5, 9, 21, 33, 27, 999000),
            'ReferenceLongitude': 100.1640625,
            'TCTRL': datetime.datetime(2018, 5, 9, 21, 33, 27, 999000),
            'TLRAN': datetime.datetime(2018, 5, 9, 21, 33, 27, 999000),
            'TINFS': datetime.datetime(2018, 5, 9, 21, 33, 27, 999000),
            'TIBBC': datetime.datetime(2018, 5, 9, 21, 33, 27, 999000),
            'TIONA': datetime.datetime(2018, 5, 9, 21, 33, 27, 999000),
            'ReferenceDistance': 100.1640625,
            'SatelliteID': 15}


class TestHRITGOESPrologueFileHandler(unittest.TestCase):
    """Test the HRITFileHandler."""

    @mock.patch('satpy.readers.goes_imager_hrit.recarray2dict')
    @mock.patch('satpy.readers.goes_imager_hrit.np.fromfile')
    @mock.patch('satpy.readers.goes_imager_hrit.HRITFileHandler.__init__')
    def test_init(self, new_fh_init, fromfile, recarray2dict):
        """Setup the hrit file handler for testing."""
        recarray2dict.side_effect = lambda x: x[0]
        new_fh_init.return_value.filename = 'filename'
        HRITGOESPrologueFileHandler.filename = 'filename'
        HRITGOESPrologueFileHandler.mda = {'total_header_length': 1}
        ret = {}
        the_time = np.array([(32, 24, 18, 146, 19, 50, 121, 153)], dtype=sgs_time)[0]
        for key in ['TCurr', 'TCHED', 'TCTRL', 'TLHED', 'TLTRL', 'TIPFS',
                    'TINFS', 'TISPC', 'TIECL', 'TIBBC', 'TISTR', 'TLRAN',
                    'TIIRT', 'TIVIT', 'TCLMT', 'TIONA']:
            ret[key] = the_time
        ret['SubSatLatitude'] = np.frombuffer(b"\x00\x00\x00\x00", dtype='>i4')[0]
        ret['ReferenceLatitude'] = np.frombuffer(b"\x00\x00\x00\x00", dtype='>i4')[0]
        ret['SubSatLongitude'] = np.frombuffer(b"\x42\x64\x2a\x00", dtype='>i4')[0]
        ret['ReferenceLongitude'] = np.frombuffer(b"\x42\x64\x2a\x00", dtype='>i4')[0]
        ret['ReferenceDistance'] = np.frombuffer(b"\x42\x64\x2a\x00", dtype='>i4')[0]
        ret['SatelliteID'] = 15
        fromfile.return_value = [ret]
        m = mock.mock_open()
        with mock.patch('satpy.readers.goes_imager_hrit.open', m, create=True) as newopen:
            newopen.return_value.__enter__.return_value.seek.return_value = 1
            self.reader = HRITGOESPrologueFileHandler(
                'filename', {'platform_shortname': 'GOES15',
                             'start_time': datetime.datetime(2016, 3, 3, 0, 0),
                             'service': 'test_service'},
                {'filetype': 'info'})

        self.assertEqual(test_pro, self.reader.prologue)


class TestHRITGOESFileHandler(unittest.TestCase):
    """Test the HRITFileHandler."""

    @mock.patch('satpy.readers.goes_imager_hrit.HRITFileHandler.__init__')
    def setUp(self, new_fh_init):
        """Setup the hrit file handler for testing."""
        blob = '$HALFTONE:=10\r\n_NAME:=albedo\r\n_UNIT:=percent\r\n0:=0.0\r\n1023:=100.0\r\n'.encode()
        mda = {'projection_parameters': {'SSP_longitude': -123.0},
               'spectral_channel_id': 1,
               'image_data_function': blob}
        HRITGOESFileHandler.filename = 'filename'
        HRITGOESFileHandler.mda = mda
        self.prologue = mock.MagicMock()
        self.prologue.prologue = test_pro
        self.reader = HRITGOESFileHandler('filename', {}, {}, self.prologue)

    def test_init(self):
        blob = '$HALFTONE:=10\r\n_NAME:=albedo\r\n_UNIT:=percent\r\n0:=0.0\r\n1023:=100.0\r\n'.encode()
        mda = {'spectral_channel_id': 1,
               'projection_parameters': {'SSP_longitude': 100.1640625},
               'image_data_function': blob}
        self.assertEqual(self.reader.mda, mda)

    @mock.patch('satpy.readers.goes_imager_hrit.HRITFileHandler.get_dataset')
    def test_get_dataset(self, base_get_dataset):
        key = mock.MagicMock()
        key.calibration = 'reflectance'
        base_get_dataset.return_value = DataArray(np.arange(25).reshape(5, 5))
        res = self.reader.get_dataset(key, {})
        expected = np.array([[np.nan, 0.097752, 0.195503, 0.293255, 0.391007],
                             [0.488759, 0.58651, 0.684262, 0.782014, 0.879765],
                             [0.977517, 1.075269, 1.173021, 1.270772, 1.368524],
                             [1.466276, 1.564027, 1.661779, 1.759531, 1.857283],
                             [1.955034, 2.052786, 2.150538, 2.248289, 2.346041]])

        self.assertTrue(np.allclose(res.values, expected, equal_nan=True))
        self.assertEqual(res.attrs['units'], '%')
        self.assertDictEqual(res.attrs['orbital_parameters'],
                             {'projection_longitude': self.reader.mda['projection_parameters']['SSP_longitude'],
                              'projection_latitude': 0.0,
                              'projection_altitude': ALTITUDE})
