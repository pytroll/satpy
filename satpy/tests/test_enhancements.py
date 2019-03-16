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
import dask.array as da


class TestEnhancementStretch(unittest.TestCase):
    """Class for testing enhancements in satpy.enhancements."""

    def setUp(self):
        """Setup the test"""
        data = np.arange(-210, 790, 100).reshape((2, 5)) * 0.95
        data[0, 0] = np.nan  # one bad value for testing
        crefl_data = np.arange(-210, 790, 100).reshape((2, 5)) * 0.95
        crefl_data /= 5.605
        crefl_data[0, 0] = np.nan  # one bad value for testing
        crefl_data[0, 1] = 0.
        self.ch1 = xr.DataArray(data, dims=('y', 'x'), attrs={'test': 'test'})
        self.ch2 = xr.DataArray(crefl_data, dims=('y', 'x'), attrs={'test': 'test'})
        rgb_data = np.stack([data, data, data])
        self.rgb = xr.DataArray(rgb_data, dims=('bands', 'y', 'x'),
                                coords={'bands': ['R', 'G', 'B']})

    def _test_enhancement(self, func, data, expected, **kwargs):
        """Helper for testing enhancement functions."""
        from trollimage.xrimage import XRImage

        pre_attrs = data.attrs
        img = XRImage(data)
        func(img, **kwargs)

        self.assertIsInstance(img.data.data, da.Array)
        self.assertListEqual(sorted(pre_attrs.keys()),
                             sorted(img.data.attrs.keys()),
                             "DataArray attributes were not preserved")

        np.testing.assert_allclose(img.data.values, expected, atol=1.e-6, rtol=0)

    def test_cira_stretch(self):
        """Test applying the cira_stretch"""
        from satpy.enhancements import cira_stretch

        expected = np.array([[
            [np.nan, -7.04045974, -7.04045974, 0.79630132, 0.95947296],
            [1.05181359, 1.11651012, 1.16635571, 1.20691137, 1.24110186]]])
        self._test_enhancement(cira_stretch, self.ch1, expected)

    def test_lookup(self):
        from satpy.enhancements import lookup
        expected = np.array([[
            [0., 0., 0., 0.333333, 0.705882],
            [1., 1., 1., 1., 1.]]])
        lut = np.arange(256.)
        self._test_enhancement(lookup, self.ch1, expected, luts=lut)

        expected = np.array([[[0., 0., 0., 0.333333, 0.705882],
                              [1., 1., 1., 1., 1.]],
                             [[0., 0., 0., 0.333333, 0.705882],
                              [1., 1., 1., 1., 1.]],
                             [[0., 0., 0., 0.333333, 0.705882],
                              [1., 1., 1., 1., 1.]]])
        lut = np.arange(256.)
        lut = np.vstack((lut, lut, lut)).T
        self._test_enhancement(lookup, self.rgb, expected, luts=lut)

    def test_colorize(self):
        from satpy.enhancements import colorize
        from trollimage.colormap import brbg
        expected = np.array([[
            [np.nan, 3.29409498e-01, 3.29409498e-01,
             4.35952940e-06, 4.35952940e-06],
            [4.35952940e-06, 4.35952940e-06, 4.35952940e-06,
             4.35952940e-06, 4.35952940e-06]],
           [[np.nan, 1.88249866e-01, 1.88249866e-01,
             2.35302110e-01, 2.35302110e-01],
            [2.35302110e-01, 2.35302110e-01, 2.35302110e-01,
             2.35302110e-01, 2.35302110e-01]],
           [[np.nan, 1.96102817e-02, 1.96102817e-02,
             1.88238767e-01, 1.88238767e-01],
            [1.88238767e-01, 1.88238767e-01, 1.88238767e-01,
             1.88238767e-01, 1.88238767e-01]]])
        self._test_enhancement(colorize, self.ch1, expected, palettes=brbg)

    def test_palettize(self):
        from satpy.enhancements import palettize
        from trollimage.colormap import brbg
        expected = np.array([[[10, 0, 0, 10, 10], [10, 10, 10, 10, 10]]])
        self._test_enhancement(palettize, self.ch1, expected, palettes=brbg)

    def test_three_d_effect(self):
        from satpy.enhancements import three_d_effect
        expected = np.array([[
            [np.nan, np.nan, -389.5, -294.5, 826.5],
            [np.nan, np.nan, 85.5, 180.5, 1301.5]]])
        self._test_enhancement(three_d_effect, self.ch1, expected)

    def test_crefl_scaling(self):
        from satpy.enhancements import crefl_scaling
        expected = np.array([[
            [np.nan, 0., 0., 0.44378, 0.631734],
            [0.737562, 0.825041, 0.912521, 1., 1.]]])
        self._test_enhancement(crefl_scaling, self.ch2, expected, idx=[0., 25., 55., 100., 255.],
                               sc=[0., 90., 140., 175., 255.])

    def test_btemp_threshold(self):
        """Test applying the cira_stretch"""
        from satpy.enhancements import btemp_threshold

        expected = np.array([[
            [np.nan, 0.946207, 0.892695, 0.839184, 0.785672],
            [0.73216, 0.595869, 0.158745, -0.278379, -0.715503]]])
        self._test_enhancement(btemp_threshold, self.ch1, expected,
                               min_in=-200, max_in=500, threshold=350)

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
