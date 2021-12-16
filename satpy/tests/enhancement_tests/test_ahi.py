#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2021 Satpy developers
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
"""Unit testing for the AHI enhancement function."""

import dask.array as da
import numpy as np
import xarray as xr


class TestAHIEnhancement():
    """Test the AHI enhancement functions."""

    def setup(self):
        """Create test data."""
        data = da.arange(-100, 1000, 110).reshape(2, 5)
        rgb_data = np.stack([data, data, data])
        self.rgb = xr.DataArray(rgb_data, dims=('bands', 'y', 'x'),
                                coords={'bands': ['R', 'G', 'B']})

    def test_jma_true_color_reproduction(self):
        """Test the jma_true_color_reproduction enhancement."""
        from trollimage.xrimage import XRImage

        from satpy.enhancements.ahi import jma_true_color_reproduction

        expected = [[[-109.98, 10.998, 131.976, 252.954, 373.932],
                    [494.91, 615.888, 736.866, 857.844, 978.822]],

                    [[-97.6, 9.76, 117.12, 224.48, 331.84],
                    [439.2, 546.56, 653.92, 761.28, 868.64]],

                    [[-94.27, 9.427, 113.124, 216.821, 320.518],
                    [424.215, 527.912, 631.609, 735.306, 839.003]]]

        img = XRImage(self.rgb)
        jma_true_color_reproduction(img)
        np.testing.assert_almost_equal(img.data.compute(), expected)
