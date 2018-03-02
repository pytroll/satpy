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
import xarray as xr


class TestEnhancementStretch(unittest.TestCase):

    """Class for testing enhancements in satpy.enhancements"""

    def setUp(self):
        """Setup the test"""
        data = np.arange(-210, 790, 100).reshape((2, 5)) * 0.95
        data[0, 0] = np.nan  # one bad value for testing
        self.ch1 = xr.DataArray(data, dims=('y', 'x'), attrs={'test': 'test'})

    def test_cira_stretch(self):
        """Test applying the cira_stretch"""
        from trollimage.xrimage import XRImage
        from satpy.enhancements import cira_stretch

        img = XRImage(self.ch1)
        cira_stretch(img)

        expected = np.array([[
            [np.nan, np.nan, np.nan, 0.79630132, 0.95947296],
            [1.05181359, 1.11651012, 1.16635571, 1.20691137, 1.24110186]]])
        np.testing.assert_allclose(img.data.values, expected)
        self.assertEqual(img.data.attrs, self.ch1.attrs)

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
