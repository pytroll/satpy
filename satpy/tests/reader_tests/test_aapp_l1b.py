#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 Satpy developers
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
"""Test module for the avhrr aapp l1b reader."""

import unittest
import numpy as np
from satpy.readers.aapp_l1b import _HEADERTYPE, _SCANTYPE, AVHRRAAPPL1BFile
import tempfile
import datetime
from satpy import DatasetID


class TestAAPPL1B(unittest.TestCase):
    """Test the filehandler."""

    def setUp(self):
        """Set up the test case."""
        self._header = np.zeros(1, dtype=_HEADERTYPE)
        self._data = np.zeros(3, dtype=_SCANTYPE)
        self._header['satid'][0] = 13
        self._header['radtempcnv'][0] = [[267194, -171669, 1002811],
                                         [930310,  -59084, 1001600],
                                         [828600,  -37854, 1001147]]

        self._data['scnlinyr'][:] = 2020
        self._data['scnlindy'][:] = 8
        self._data['scnlintime'][0] = 30195225
        self._data['scnlintime'][1] = 30195389
        self._data['scnlintime'][2] = 30195556
        self._data['scnlinbit'][0] = -16383
        self._data['scnlinbit'][1] = -16383
        self._data['scnlinbit'][2] = -16384
        calvis = np.array([[[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [543489984, -21941870, 1592440064, -545027008, 499]],
                           [[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [540780032,  -22145690, 1584350080, -543935616, 500]],
                           [[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [257550000, -10449420, 1812019968, -785690304, 499]]])
        self._data['calvis'][:] = calvis
        self._data['calir'] = [[[[0, -2675, 2655265],
                                 [0, 0, 0]],
                                [[33605, -260786, 226818992],
                                 [0, 0, 0]],
                                [[13869, -249508, 234624768],
                                 [0, 0, 0]]],
                               [[[0, -2675, 2655265],
                                 [0, 0, 0]],
                                [[33609, -260810, 226837328],
                                 [0, 0, 0]],
                                [[13870, -249520, 234638704],
                                 [0, 0, 0]]],
                               [[[0, 0, 0],
                                 [0, 0, 0]],
                                [[33614, -260833, 226855664],
                                 [0, 0, 0]],
                                [[13871, -249531, 234652640],
                                 [0, 0, 0]]]]
        self._data['hrpt'] = np.ones_like(self._data['hrpt']) * (np.arange(2048) // 2)[np.newaxis, :, np.newaxis]

        self.filename_info = {'platform_shortname': 'metop03', 'start_time': datetime.datetime(2020, 1, 8, 8, 19),
                              'orbit_number': 6071}
        self.filetype_info = {'file_reader': AVHRRAAPPL1BFile,
                              'file_patterns': ['hrpt_{platform_shortname}_{start_time:%Y%m%d_%H%M}_{orbit_number:05d}.l1b'],  # noqa
                              'file_type': 'avhrr_aapp_l1b'}

    def test_read(self):
        """Test the reading."""
        with tempfile.TemporaryFile() as tmpfile:
            self._header.tofile(tmpfile)
            tmpfile.seek(22016, 0)
            self._data.tofile(tmpfile)

            fh = AVHRRAAPPL1BFile(tmpfile, self.filename_info, self.filetype_info)
            info = {}
            mins = []
            maxs = []
            for name in ['1', '2', '3a']:
                key = DatasetID(name=name, calibration='reflectance')
                res = fh.get_dataset(key, info)
                assert(res.min() == 0)
                assert(res.max() >= 100)
                mins.append(res.min().values)
                maxs.append(res.max().values)
                if name == '3a':
                    assert(np.all(np.isnan(res[:2, :])))

            for name in ['3b', '4', '5']:
                key = DatasetID(name=name, calibration='reflectance')
                res = fh.get_dataset(key, info)
                mins.append(res.min().values)
                maxs.append(res.max().values)
                if name == '3b':
                    assert(np.all(np.isnan(res[2:, :])))

            np.testing.assert_allclose(mins, [0., 0., 0., 204.10106939, 103.23477235, 106.42609758])
            np.testing.assert_allclose(maxs, [108.40391775, 107.68545158, 106.80061233,
                                              337.71416096, 355.15898219, 350.87182166])

    def test_angles(self):
        """Test reading the angles."""
        with tempfile.TemporaryFile() as tmpfile:
            self._header.tofile(tmpfile)
            tmpfile.seek(22016, 0)
            self._data.tofile(tmpfile)

            fh = AVHRRAAPPL1BFile(tmpfile, self.filename_info, self.filetype_info)
            info = {}
            key = DatasetID(name='solar_zenith_angle')
            res = fh.get_dataset(key, info)
            assert(np.all(res == 0))

    def test_navigation(self):
        """Test reading the lon and lats."""
        with tempfile.TemporaryFile() as tmpfile:
            self._header.tofile(tmpfile)
            tmpfile.seek(22016, 0)
            self._data.tofile(tmpfile)

            fh = AVHRRAAPPL1BFile(tmpfile, self.filename_info, self.filetype_info)
            info = {}
            key = DatasetID(name='longitude')
            res = fh.get_dataset(key, info)
            assert(np.all(res == 0))
            key = DatasetID(name='latitude')
            res = fh.get_dataset(key, info)
            assert(np.all(res == 0))
