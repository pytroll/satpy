#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 Satpy developers
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
"""Unit testing for the ABI enhancement functions."""

import unittest
import numpy as np
import xarray as xr
import dask.array as da


class TestABIEnhancement(unittest.TestCase):
    """Test the ABI enhancement functions."""

    def setUp(self):
        """Create fake data for the tests."""
        data = da.linspace(0, 1, 16).reshape((4, 4))
        self.da = xr.DataArray(data, dims=('y', 'x'), attrs={'test': 'test'})

    def test_cimss_true_color_contrast(self):
        """Test the cimss_true_color_contrast enhancement."""
        from satpy.enhancements.abi import cimss_true_color_contrast
        from trollimage.xrimage import XRImage

        expected = np.array([[
            [0., 0., 0.05261956, 0.13396146],
            [0.21530335, 0.29664525, 0.37798715, 0.45932905],
            [0.54067095, 0.62201285, 0.70335475, 0.78469665],
            [0.86603854, 0.94738044, 1., 1.],
            ]])
        img = XRImage(self.da)
        cimss_true_color_contrast(img)
        np.testing.assert_almost_equal(img.data.compute(), expected)
