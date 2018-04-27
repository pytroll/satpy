#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017-2018 PyTroll Community

# Author(s):

#   Adam.Dybbroe <adam.dybbroe@smhi.se>

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
from satpy.readers.msg_base import SEVIRICalibrationHandler

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

try:
    from unittest import mock
except ImportError:
    import mock


TEST1_INPUT = np.array([[377.,  377.,  377.,  376.,  375.],
                        [376.,  375.,  376.,  374.,  374.],
                        [374.,  373.,  373.,  374.,  374.],
                        [347.,  345.,  345.,  348.,  347.],
                        [306.,  306.,  307.,  307.,  308.]], dtype=np.float32)

TEST1_OUTPUT = np.array([[66.84162903,  66.84162903,  66.84162903,  66.63659668,
                          66.4315567],
                         [66.63659668,  66.4315567,  66.63659668,  66.22652435,
                          66.22652435],
                         [66.22652435,  66.02148438,  66.02148438,  66.22652435,
                          66.22652435],
                         [60.69055939,  60.28048706,  60.28048706,  60.89559937,
                          60.69055939],
                         [52.28409576,  52.28409576,  52.48912811,  52.48912811,
                          52.69416809]], dtype=np.float32)

TEST1_GAIN = 0.20503567620766011
TEST1_OFFSET = -10.456819486590666

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


# Calibration type = Effective radiances
CALIBRATION_TYPE = np.array(
    [[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]], dtype=np.uint8)


# This should preferably be put in a helper-module
# Fixme!
def assertNumpyArraysEqual(self, other):
    if self.shape != other.shape:
        raise AssertionError("Shapes don't match")
    if not np.allclose(self, other):
        raise AssertionError("Elements don't match!")


class TestSEVIRICalibrationHandler(unittest.TestCase):

    """Test the SEVIRICalibrationHandler class in the msg_base module"""

    def setUp(self):
        """Setup the SEVIRI Calibration handler for testing."""

        hdr = {}
        hdr['15_DATA_HEADER'] = {}
        hdr['15_DATA_HEADER']['RadiometricProcessing'] = {
            'Level15ImageCalibration': CAL_DTYPE}

        hdr['15_DATA_HEADER']['ImageDescription'] = {}
        hdr['15_DATA_HEADER']['ImageDescription']['Level15ImageProduction'] = {
            'PlannedChanProcessing': CALIBRATION_TYPE}

        self.handler = SEVIRICalibrationHandler()

    def test_convert_to_radiance(self):
        """Test the conversion from counts to radiance method"""

        data = TEST1_INPUT
        gain = TEST1_GAIN
        offset = TEST1_OFFSET
        result = self.handler._convert_to_radiance(data, gain, offset)
        assertNumpyArraysEqual(result, TEST1_OUTPUT)

    def test_erads2bt(self):

        pass

    def test_srads2bt(self):

        pass

    def test_tl15(self):

        pass

    def test_ir_calibrate(self):

        pass

    def test_vis_calibrate(self):

        pass

    def tearDown(self):
        pass


def suite():
    """The test suite for test_scene.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestSEVIRICalibrationHandler))
    return mysuite

if __name__ == "__main__":
    # So you can run tests from this module individually.
    unittest.main()
