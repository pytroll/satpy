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

"""Unit testing the enhancements functions, e.g. cira_stretch
"""

import unittest
import numpy as np
from satpy.enhancements import cira_stretch

CH1 = np.ma.arange(-110, 890, 10) * 0.95
CH1.mask = np.logical_and(np.arange(100) * 0.1 > 3, np.arange(100) * 0.1 < 8)

RESMASK = np.array([False, False, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False, False,
                    False, False, False, False,  True,  True,  True,  True,  True,
                    True,  True,  True,  True,  True,  True,  True,  True,  True,
                    True,  True,  True,  True,  True,  True,  True,  True,  True,
                    True,  True,  True,  True,  True,  True,  True,  True,  True,
                    True,  True,  True,  True,  True,  True,  True,  True,  True,
                    True,  True,  True,  True,  True,  True,  True,  True, False,
                    False, False, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False, False, False], dtype=bool)

RESULT = np.ma.array([-1.04500000e+00,  -9.50000000e-01,  -8.55000000e-01,
                      -7.60000000e-01,  -6.65000000e-01,  -5.70000000e-01,
                      -4.75000000e-01,  -3.80000000e-01,  -2.85000000e-01,
                      -1.90000000e-01,  -9.50000000e-02,   0.00000000e+00,
                      9.50000000e-02,   1.90000000e-01,   2.85000000e-01,
                      3.80000000e-01,   4.75000000e-01,   5.70000000e-01,
                      6.65000000e-01,   7.60000000e-01,   8.55000000e-01,
                      9.50000000e-01,   1.04500000e+00,   1.14000000e+00,
                      1.23500000e+00,   1.33000000e+00,   1.42500000e+00,
                      1.52000000e+00,   1.61500000e+00,   1.71000000e+00,
                      1.80500000e+00,   1.90000000e+02,   1.99500000e+02,
                      2.09000000e+02,   2.18500000e+02,   2.28000000e+02,
                      2.37500000e+02,   2.47000000e+02,   2.56500000e+02,
                      2.66000000e+02,   2.75500000e+02,   2.85000000e+02,
                      2.94500000e+02,   3.04000000e+02,   3.13500000e+02,
                      3.23000000e+02,   3.32500000e+02,   3.42000000e+02,
                      3.51500000e+02,   3.61000000e+02,   3.70500000e+02,
                      3.80000000e+02,   3.89500000e+02,   3.99000000e+02,
                      4.08500000e+02,   4.18000000e+02,   4.27500000e+02,
                      4.37000000e+02,   4.46500000e+02,   4.56000000e+02,
                      4.65500000e+02,   4.75000000e+02,   4.84500000e+02,
                      4.94000000e+02,   5.03500000e+02,   5.13000000e+02,
                      5.22500000e+02,   5.32000000e+02,   5.41500000e+02,
                      5.51000000e+02,   5.60500000e+02,   5.70000000e+02,
                      5.79500000e+02,   5.89000000e+02,   5.98500000e+02,
                      6.08000000e+02,   6.17500000e+02,   6.27000000e+02,
                      6.36500000e+02,   6.46000000e+02,   6.55500000e+00,
                      6.65000000e+00,   6.74500000e+00,   6.84000000e+00,
                      6.93500000e+00,   7.03000000e+00,   7.12500000e+00,
                      7.22000000e+00,   7.31500000e+00,   7.41000000e+00,
                      7.50500000e+00,   7.60000000e+00,   7.69500000e+00,
                      7.79000000e+00,   7.88500000e+00,   7.98000000e+00,
                      8.07500000e+00,   8.17000000e+00,   8.26500000e+00,
                      8.36000000e+00], mask=RESMASK)


def assertNumpyArraysEqual(self, other):
    if self.shape != other.shape:
        raise AssertionError("Shapes don't match")
    if not np.ma.allclose(self, other):
        raise AssertionError("Elements don't match!")


class TrollImageMock(object):

    def __init__(self):
        self.channels = []


class TestSatinHelpers(unittest.TestCase):

    '''Class for testing satpy.satin'''

    def setUp(self):
        """Setup the test"""
        pass

    def test_cira_stretch(self):
        """Test applying the cira_stretch"""

        img = TrollImageMock()
        img.channels.append(CH1)
        cira_stretch(img)
        assertNumpyArraysEqual(CH1, img.channels[0])

    def tearDown(self):
        """Clean up"""
        pass


def suite():
    """The test suite for test_satin_helpers.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestHelpers))

    return mysuite

if __name__ == "__main__":
    # So you can run tests from this module individually.
    unittest.main()
