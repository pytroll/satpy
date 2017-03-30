#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017 Adam.Dybbroe

# Author(s):

#   Adam.Dybbroe <a000680@c20671.ad.smhi.se>

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

"""Unittesting the native msg reader
"""

import os
import sys
from datetime import datetime

import mock
import numpy as np

CAL_DTYPE = np.array([[(0.0208876,  -1.06526761), (0.0278805,  -1.42190546),
                       (0.0235881,  -1.20299312), (0.00365867,  -0.18659201),
                       (0.00831811,  -0.42422367), (0.03862197,  -1.96972038),
                       (0.12674432,  -6.46396025), (0.10396091,  -5.30200645),
                       (0.20503568, -10.45681949), (0.22231115, -11.33786848),
                       (0.1576069,  -8.03795174), (0.0373969,  -1.90724192)]],
                     dtype=[('CalSlope', '>f8'), ('CalOffset', '>f8')])
IR_108_RADIANCES = np.array([[15.30867688,  15.37944118,  15.45020548,
                              15.52096978,  15.59173408],
                             [15.66249838,  15.73326268,  15.80402698,
                              15.87479128,  15.94555558],
                             [16.01631988,  16.08708418,  16.15784848,
                              16.22861278,  16.29937708]],
                            dtype=np.float64)


from satpy.readers.native_msg import NativeMSGFileHandler

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


def assertNumpyArraysEqual(self, other):
    if self.shape != other.shape:
        raise AssertionError("Shapes don't match")
    if not np.allclose(self, other):
        raise AssertionError("Elements don't match!")


class TestNativeMSGFileHandler(unittest.TestCase):

    """Test the NativeMSGFileHandler."""

    @mock.patch('satpy.readers.native_msg.NativeMSGFileHandler._get_header')
    @mock.patch('satpy.readers.native_msg.NativeMSGFileHandler._get_filedtype')
    @mock.patch('satpy.readers.native_msg.NativeMSGFileHandler._get_memmap')
    def setUp(self, _get_memmap, _get_filedtype, _get_header):
        """Setup the natve MSG file handler for testing."""

        hdr = {}
        hdr['15_DATA_HEADER'] = {}
        hdr['15_DATA_HEADER']['RadiometricProcessing'] = {
            'Level15ImageCalibration': CAL_DTYPE}

        _get_header.return_value = None
        _get_filedtype.return_value = None
        _get_filedtype.return_value = None

        self.reader = NativeMSGFileHandler('filename',
                                           {'platform_shortname': 'MSG3',
                                            'start_time': datetime(2017, 3, 26, 10, 0)},
                                           {'filetype': 'info'})

        self.reader.header = hdr
        self.reader.channel_order_list = [
            'WV_062', 'IR_087', 'IR_108', 'IR_134']
        self.available_channels = {'WV_062': True, 'IR_087': True,
                                   'IR_108': True, 'IR_134': True}

    def test_convert_to_radiance(self):
        """Test the conversion from counts to radiance method"""

        data = np.ma.ones((3, 5)) * 700 + np.arange(0, 45, 3).reshape(3, 5)
        key_name = 'IR_108'
        self.reader.convert_to_radiance(data, key_name)
        assertNumpyArraysEqual(data.data, IR_108_RADIANCES)

    def tearDown(self):
        pass


def suite():
    """The test suite for test_scene.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestNativeMSGFileHandler))
    return mysuite

if __name__ == "__main__":
    # So you can run tests from this module individually.
    unittest.main()
