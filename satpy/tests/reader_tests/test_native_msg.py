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

import sys
from datetime import datetime

import numpy as np
from satpy.readers.native_msg import NativeMSGFileHandler

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

try:
    from unittest import mock
except ImportError:
    import mock


CAL_DTYPE = np.array([[(0.0208876,  -1.06526761), (0.0278805,  -1.42190546),
                       (0.0235881,  -1.20299312), (0.00365867,  -0.18659201),
                       (0.00831811,  -0.42422367), (0.03862197,  -1.96972038),
                       (0.12674432,  -6.46396025), (0.10396091,  -5.30200645),
                       (0.20503568, -10.45681949), (0.22231115, -11.33786848),
                       (0.1576069,  -8.03795174), (0.0373969,  -1.90724192)]],
                     dtype=[('CalSlope', '>f8'), ('CalOffset', '>f8')])
IR_108_RADIANCES = np.ma.array([[133.06815651,  133.68326355,  134.29837059,  134.91347763,
                                 135.52858467],
                                [136.14369171,  136.75879875,  137.37390579,  137.98901283,
                                 138.60411987],
                                [139.21922691,  139.83433395,  140.44944099,  141.06454803,
                                 141.67965507]],
                               mask=False, dtype=np.float64)

VIS006_RADIANCES = np.ma.array([[13.55605239,  13.61871519,  13.68137799,  13.74404079,
                                 13.80670359],
                                [13.86936639,  13.93202919,  13.99469199,  14.05735479,
                                 14.12001759],
                                [14.18268039,  14.24534319,  14.30800599,  14.37066879,
                                 14.43333159]], mask=False, dtype=np.float64)

VIS006_REFLECTANCES = np.array([[65.00454035,  65.30502359,  65.60550682,  65.90599006,
                                 66.2064733],
                                [66.50695654,  66.80743977,  67.10792301,  67.40840625,
                                 67.70888949],
                                [68.00937272,  68.30985596,  68.6103392,  68.91082244,
                                 69.21130567]], dtype=np.float64)

IR_108_TBS = np.array([[311.77913132,  312.11070275,  312.44143083,  312.77132215,
                        313.10038322],
                       [313.42862046,  313.75604023,  314.0826488,  314.40845236,
                        314.73345704],
                       [315.05766888,  315.38109386,  315.70373788,  316.02560677,
                        316.34670629]], dtype=np.float64)


CHANNEL_ORDER_LIST = ['VIS006', 'VIS008', 'IR_016', 'IR_039',
                      'WV_062', 'WV_073', 'IR_087', 'IR_097',
                      'IR_108', 'IR_120', 'IR_134', 'HRV']
AVAILABLE_CHANNELS = {}
for item in CHANNEL_ORDER_LIST:
    AVAILABLE_CHANNELS[item] = True

# Calibration type = Effective radiances
CALIBRATION_TYPE = np.array(
    [[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]], dtype=np.uint8)


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

        hdr['15_DATA_HEADER']['ImageDescription'] = {}
        hdr['15_DATA_HEADER']['ImageDescription']['Level15ImageProduction'] = {
            'PlannedChanProcessing': CALIBRATION_TYPE}

        _get_header.return_value = None
        _get_filedtype.return_value = None

        self.reader = NativeMSGFileHandler('filename',
                                           {'platform_shortname': 'MSG3',
                                            'start_time': datetime(2017, 3, 26, 10, 0)},
                                           {'filetype': 'info'})

        self.reader.platform_name = 'Meteosat-10'
        self.reader.platform_id = 323
        self.reader.header = hdr
        self.reader.channel_order_list = CHANNEL_ORDER_LIST
        #self.reader.available_channels = AVAILABLE_CHANNELS

    def test_convert_to_radiance(self):
        """Test the conversion from counts to radiance method"""

        data = np.ma.ones((3, 5)) * 700 + np.arange(0, 45, 3).reshape(3, 5)
        key_name = 'IR_108'
        self.reader.convert_to_radiance(data, key_name)
        assertNumpyArraysEqual(data.data, IR_108_RADIANCES.data)

    def test_vis_calibrate(self):
        """Test the visible calibration: from radiances to reflectances"""

        key_name = 'VIS006'
        data = VIS006_RADIANCES[:]
        self.reader._vis_calibrate(data, key_name)
        assertNumpyArraysEqual(data.data, VIS006_REFLECTANCES)

    def test_ir_calibrate(self):
        """Test the IR calibration: from radiances to brightness temperatures"""

        key_name = 'IR_108'
        data = IR_108_RADIANCES[:]
        self.reader._ir_calibrate(data, key_name)
        assertNumpyArraysEqual(data.data, IR_108_TBS)

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
