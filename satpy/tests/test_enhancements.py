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

CH1 = np.ma.arange(-210, 790, 100) * 0.95
CH1.mask = np.logical_and(np.arange(10) * 0.1 > 0.3, np.arange(10) * 0.1 < 0.8)

RESMASK = np.array(
    [False,  False,  False,  True,  True,  True,  True,  True, False, False], dtype=bool)
RESULT = np.ma.array([np.nan,         np.nan,      np.nan,  1.93196611,  2.25647721,
                      2.4401216,  2.56878821,  2.66791969,  1.20691137,  1.24110186], mask=RESMASK)


def assertNumpyArraysEqual(self, other):
    if self.shape != other.shape:
        raise AssertionError("Shapes don't match")
    if not np.allclose(self, other, equal_nan=True):
        raise AssertionError("Elements don't match!")


class TrollImageMock(object):

    def __init__(self):
        self.channels = []


class TestEnhancementStretch(unittest.TestCase):

    '''Class for testing satpy.satin'''

    def setUp(self):
        """Setup the test"""
        pass

    def test_cira_stretch(self):
        """Test applying the cira_stretch"""

        img = TrollImageMock()
        img.channels.append(CH1)
        cira_stretch(img)
        assertNumpyArraysEqual(RESULT.data, img.channels[0].data)
        assertNumpyArraysEqual(RESULT.mask, img.channels[0].mask)

    def tearDown(self):
        """Clean up"""
        pass


def suite():
    """The test suite for test_satin_helpers.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestEnhancementStretch))

    return mysuite

if __name__ == "__main__":
    # So you can run tests from this module individually.
    unittest.main()
