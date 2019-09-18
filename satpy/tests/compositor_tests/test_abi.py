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
"""Tests for ABI compositors.
"""

import sys

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


class TestABIComposites(unittest.TestCase):

    """Test ABI-specific composites."""

    def test_load_composite_yaml(self):
        """Test loading the yaml for this sensor."""
        from satpy.composites import CompositorLoader
        cl = CompositorLoader()
        cl.load_sensor_composites('abi')

    def test_simulated_green(self):
        """Test creating a fake 'green' band."""
        import xarray as xr
        import dask.array as da
        import numpy as np
        from satpy.composites.abi import SimulatedGreen
        from pyresample.geometry import AreaDefinition
        rows = 5
        cols = 10
        area = AreaDefinition(
            'test', 'test', 'test',
            {'proj': 'eqc', 'lon_0': 0.0,
             'lat_0': 0.0},
            cols, rows,
            (-20037508.34, -10018754.17, 20037508.34, 10018754.17))

        comp = SimulatedGreen('green', prerequisites=('C01', 'C02', 'C03'),
                              standard_name='toa_bidirectional_reflectance')
        c01 = xr.DataArray(da.zeros((rows, cols), chunks=25) + 0.25,
                           dims=('y', 'x'),
                           attrs={'name': 'C01', 'area': area})
        c02 = xr.DataArray(da.zeros((rows, cols), chunks=25) + 0.30,
                           dims=('y', 'x'),
                           attrs={'name': 'C02', 'area': area})
        c03 = xr.DataArray(da.zeros((rows, cols), chunks=25) + 0.35,
                           dims=('y', 'x'),
                           attrs={'name': 'C03', 'area': area})
        res = comp((c01, c02, c03))
        self.assertIsInstance(res, xr.DataArray)
        self.assertIsInstance(res.data, da.Array)
        self.assertEqual(res.attrs['name'], 'green')
        self.assertEqual(res.attrs['standard_name'],
                         'toa_bidirectional_reflectance')
        data = res.compute()
        np.testing.assert_allclose(data, 0.28025)


def suite():
    """The test suite for test_abi.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestABIComposites))
    return mysuite
