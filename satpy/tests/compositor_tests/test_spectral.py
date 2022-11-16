#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 Satpy developers
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
"""Tests for spectral correction compositors."""

import unittest

import dask.array as da
import numpy as np
import xarray as xr

from satpy.composites.spectral import NDVIHybridGreen


class TestSpectralComposites(unittest.TestCase):
    """Test composites for spectral channel corrections."""

    def test_ndvi_hybrid_green(self):
        """Test generation of NDVI-scaled hybrid green channel."""
        vis_05 = xr.DataArray(da.from_array([[0.25, 0.30], [0.20, 0.30]], chunks=25),
                              dims=('y', 'x'),
                              attrs={'name': 'vis05'})
        vis_06 = xr.DataArray(da.from_array([[0.25, 0.30], [0.25, 0.35]], chunks=25),
                              dims=('y', 'x'),
                              attrs={'name': 'vis06'})
        vis_08 = xr.DataArray(da.from_array([[0.35, 0.35], [0.28, 0.65]], chunks=25),
                              dims=('y', 'x'),
                              attrs={'name': 'vis08'})

        comp = NDVIHybridGreen('ndvi_hybrid_green', fractions=(0.15, 0.05), prerequisites=(0.51, 0.65, 0.85),
                               standard_name='toa_bidirectional_reflectance')

        res = comp((vis_05, vis_06, vis_08))
        self.assertIsInstance(res, xr.DataArray)
        self.assertIsInstance(res.data, da.Array)
        self.assertEqual(res.attrs['name'], 'ndvi_hybrid_green')
        self.assertEqual(res.attrs['standard_name'],
                         'toa_bidirectional_reflectance')
        data = res.values
        np.testing.assert_array_almost_equal(data, np.array([[0.2633, 0.3071], [0.2115, 0.3420]]), decimal=4)
