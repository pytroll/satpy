#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 PyTroll developers
#
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Tests for AHI compositors.
"""

import sys

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


class TestAHIComposites(unittest.TestCase):

    """Test AHI-specific composites."""

    def test_load_composite_yaml(self):
        """Test loading the yaml for this sensor."""
        from satpy.composites import CompositorLoader
        cl = CompositorLoader()
        cl.load_sensor_composites('abi')

    def test_corrected_green(self):
        """Test adjusting the 'green' band."""
        import xarray as xr
        import dask.array as da
        import numpy as np
        from satpy.composites.ahi import GreenCorrector
        from pyresample.geometry import AreaDefinition
        rows = 5
        cols = 10
        area = AreaDefinition(
            'test', 'test', 'test',
            {'proj': 'eqc', 'lon_0': 0.0,
             'lat_0': 0.0},
            cols, rows,
            (-20037508.34, -10018754.17, 20037508.34, 10018754.17))

        comp = GreenCorrector('green', prerequisites=(0.51, 0.85),
                              standard_name='toa_bidirectional_reflectance')
        c01 = xr.DataArray(da.zeros((rows, cols), chunks=25) + 0.25,
                           dims=('y', 'x'),
                           attrs={'name': 'C01', 'area': area})
        c02 = xr.DataArray(da.zeros((rows, cols), chunks=25) + 0.30,
                           dims=('y', 'x'),
                           attrs={'name': 'C02', 'area': area})
        res = comp((c01, c02))
        self.assertIsInstance(res, xr.DataArray)
        self.assertIsInstance(res.data, da.Array)
        self.assertEqual(res.attrs['name'], 'green')
        self.assertEqual(res.attrs['standard_name'],
                         'toa_bidirectional_reflectance')
        data = res.compute()
        np.testing.assert_allclose(data, 0.2575)


def suite():
    """The test suite for test_ahi.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestAHIComposites))
    return mysuite
