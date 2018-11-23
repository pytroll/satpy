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
"""Tests for compositors.
"""

from satpy.tests.compositor_tests import test_abi, test_ahi, test_viirs

try:
    from unittest import mock
except ImportError:
    import mock

import unittest


class TestCheckArea(unittest.TestCase):

    """Test the utility method 'check_areas'."""

    def _get_test_ds(self, shape=(50, 100), dims=('y', 'x')):
        """Helper method to get a fake DataArray."""
        import xarray as xr
        import dask.array as da
        from pyresample.geometry import AreaDefinition
        data = da.random.random(shape, chunks=25)
        area = AreaDefinition(
            'test', 'test', 'test',
            {'proj': 'eqc', 'lon_0': 0.0,
             'lat_0': 0.0},
            shape[dims.index('x')], shape[dims.index('y')],
            (-20037508.34, -10018754.17, 20037508.34, 10018754.17))
        attrs = {'area': area}
        return xr.DataArray(data, dims=dims, attrs=attrs)

    def test_single_ds(self):
        """Test a single dataset is returned unharmed."""
        from satpy.composites import CompositeBase
        ds1 = self._get_test_ds()
        comp = CompositeBase('test_comp')
        ret_datasets = comp.check_areas((ds1,))
        self.assertIs(ret_datasets[0], ds1)

    def test_mult_ds_area(self):
        """Test multiple datasets successfully pass."""
        from satpy.composites import CompositeBase
        ds1 = self._get_test_ds()
        ds2 = self._get_test_ds()
        comp = CompositeBase('test_comp')
        ret_datasets = comp.check_areas((ds1, ds2))
        self.assertIs(ret_datasets[0], ds1)
        self.assertIs(ret_datasets[1], ds2)

    def test_mult_ds_no_area(self):
        """Test that all datasets must have an area attribute."""
        from satpy.composites import CompositeBase
        ds1 = self._get_test_ds()
        ds2 = self._get_test_ds()
        del ds2.attrs['area']
        comp = CompositeBase('test_comp')
        self.assertRaises(ValueError, comp.check_areas, (ds1, ds2))

    def test_mult_ds_diff_area(self):
        """Test that datasets with different areas fail."""
        from satpy.composites import CompositeBase, IncompatibleAreas
        from pyresample.geometry import AreaDefinition
        ds1 = self._get_test_ds()
        ds2 = self._get_test_ds()
        ds2.attrs['area'] = AreaDefinition(
            'test', 'test', 'test',
            {'proj': 'eqc', 'lon_0': 0.0,
             'lat_0': 0.0},
            100, 50,
            (-30037508.34, -20018754.17, 10037508.34, 18754.17))
        comp = CompositeBase('test_comp')
        self.assertRaises(IncompatibleAreas, comp.check_areas, (ds1, ds2))

    def test_mult_ds_diff_dims(self):
        """Test that datasets with different dimensions still pass."""
        from satpy.composites import CompositeBase
        # x is still 50, y is still 100, even though they are in
        # different order
        ds1 = self._get_test_ds(shape=(50, 100), dims=('y', 'x'))
        ds2 = self._get_test_ds(shape=(3, 100, 50), dims=('bands', 'x', 'y'))
        comp = CompositeBase('test_comp')
        ret_datasets = comp.check_areas((ds1, ds2))
        self.assertIs(ret_datasets[0], ds1)
        self.assertIs(ret_datasets[1], ds2)

    def test_mult_ds_diff_size(self):
        """Test that datasets with different sizes fail."""
        from satpy.composites import CompositeBase, IncompatibleAreas
        # x is 50 in this one, 100 in ds2
        # y is 100 in this one, 50 in ds2
        ds1 = self._get_test_ds(shape=(50, 100), dims=('x', 'y'))
        ds2 = self._get_test_ds(shape=(3, 50, 100), dims=('bands', 'y', 'x'))
        comp = CompositeBase('test_comp')
        self.assertRaises(IncompatibleAreas, comp.check_areas, (ds1, ds2))


class TestDayNightCompositor(unittest.TestCase):
    """Test DayNightCompositor."""

    def setUp(self):
        """Create test data."""
        import xarray as xr
        import dask.array as da
        import numpy as np
        from datetime import datetime
        bands = ['R', 'G', 'B']
        start_time = datetime(2018, 1, 1, 18, 0, 0)

        # RGB
        a = np.zeros((3, 2, 2), dtype=np.float)
        a[:, 0, 0] = 0.1
        a[:, 0, 1] = 0.2
        a[:, 1, 0] = 0.3
        a[:, 1, 1] = 0.4
        a = da.from_array(a, a.shape)
        self.data_a = xr.DataArray(a, attrs={'test': 'a', 'start_time': start_time},
                                   coords={'bands': bands}, dims=('bands', 'y', 'x'))
        b = np.zeros((3, 2, 2), dtype=np.float)
        b[:, 0, 0] = np.nan
        b[:, 0, 1] = 0.25
        b[:, 1, 0] = 0.50
        b[:, 1, 1] = 0.75
        b = da.from_array(b, b.shape)
        self.data_b = xr.DataArray(b, attrs={'test': 'b', 'start_time': start_time},
                                   coords={'bands': bands}, dims=('bands', 'y', 'x'))

        sza = np.array([[80., 86.], [94., 100.]])
        sza = da.from_array(sza, sza.shape)
        self.sza = xr.DataArray(sza, dims=('y', 'x'))

        # fake area
        my_area = mock.MagicMock()
        lons = np.array([[-95., -94.], [-93., -92.]])
        lons = da.from_array(lons, lons.shape)
        lats = np.array([[40., 41.], [42., 43.]])
        lats = da.from_array(lats, lats.shape)
        my_area.get_lonlats_dask.return_value = (lons, lats)
        self.data_a.attrs['area'] = my_area
        self.data_b.attrs['area'] = my_area
        # not used except to check that it matches the data arrays
        self.sza.attrs['area'] = my_area

    def test_basic_sza(self):
        """Test compositor when SZA data is included"""
        import numpy as np
        from satpy.composites import DayNightCompositor
        comp = DayNightCompositor(name='dn_test')
        res = comp((self.data_a, self.data_b, self.sza))
        res = res.compute()
        expected = np.array([[0., 0.2985455], [0.51680423, 1.]])
        np.testing.assert_allclose(res.values[0], expected)

    def test_basic_area(self):
        """Test compositor when SZA data is not provided."""
        import numpy as np
        from satpy.composites import DayNightCompositor
        comp = DayNightCompositor(name='dn_test')
        res = comp((self.data_a, self.data_b))
        res = res.compute()
        expected = np.array([[0., 0.33164983], [0.66835017, 1.]])
        np.testing.assert_allclose(res.values[0], expected)


class TestFillingCompositor(unittest.TestCase):

    def test_fill(self):
        import numpy as np
        import xarray as xr
        from satpy.composites import FillingCompositor
        comp = FillingCompositor(name='fill_test')
        filler = xr.DataArray(np.array([1, 2, 3, 4, 3, 2, 1]))
        red = xr.DataArray(np.array([1, 2, 3, np.nan, 3, 2, 1]))
        green = xr.DataArray(np.array([np.nan, 2, 3, 4, 3, 2, np.nan]))
        blue = xr.DataArray(np.array([4, 3, 2, 1, 2, 3, 4]))
        res = comp([filler, red, green, blue])
        np.testing.assert_allclose(res.sel(bands='R').data, filler.data)
        np.testing.assert_allclose(res.sel(bands='G').data, filler.data)
        np.testing.assert_allclose(res.sel(bands='B').data, blue.data)


class TestLuminanceSharpeningCompositor(unittest.TestCase):
    """Test luminance sharpening compositor."""

    def test_compositor(self):
        """Test luminance sharpening compositor."""
        import numpy as np
        import xarray as xr
        from satpy.composites import LuminanceSharpeningCompositor
        comp = LuminanceSharpeningCompositor(name='test')
        # Three shades of grey
        rgb_arr = np.array([1, 50, 100, 200, 1, 50, 100, 200, 1, 50, 100, 200])
        rgb = xr.DataArray(rgb_arr.reshape((3, 2, 2)),
                           dims=['bands', 'y', 'x'])
        # 100 % luminance -> all result values ~1.0
        lum = xr.DataArray(np.array([[100., 100.], [100., 100.]]),
                           dims=['y', 'x'])
        res = comp([lum, rgb])
        np.testing.assert_allclose(res.data, 1., atol=1e-9)
        # 50 % luminance, all result values ~0.5
        lum = xr.DataArray(np.array([[50., 50.], [50., 50.]]),
                           dims=['y', 'x'])
        res = comp([lum, rgb])
        np.testing.assert_allclose(res.data, 0.5, atol=1e-9)
        # 30 % luminance, all result values ~0.3
        lum = xr.DataArray(np.array([[30., 30.], [30., 30.]]),
                           dims=['y', 'x'])
        res = comp([lum, rgb])
        np.testing.assert_allclose(res.data, 0.3, atol=1e-9)
        # 0 % luminance, all values ~0.0
        lum = xr.DataArray(np.array([[0., 0.], [0., 0.]]),
                           dims=['y', 'x'])
        res = comp([lum, rgb])
        np.testing.assert_allclose(res.data, 0.0, atol=1e-9)


class TestSandwichCompositor(unittest.TestCase):
    """Test sandwich compositor."""

    @mock.patch('satpy.composites.enhance2dataset')
    def test_compositor(self, e2d):
        """Test luminance sharpening compositor."""
        import numpy as np
        import xarray as xr
        from satpy.composites import SandwichCompositor

        rgb_arr = np.random.random((3, 2, 2))
        rgb = xr.DataArray(rgb_arr, dims=['bands', 'y', 'x'])
        lum_arr = 100 * np.random.random((2, 2))
        lum = xr.DataArray(lum_arr, dims=['y', 'x'])

        # Make enhance2dataset return unmodified dataset
        e2d.return_value = rgb
        comp = SandwichCompositor(name='test')

        res = comp([lum, rgb])

        for i in range(3):
            np.testing.assert_allclose(res.data[i, :, :],
                                       rgb_arr[i, :, :] * lum_arr / 100.)


class TestInlineComposites(unittest.TestCase):
    """Test inline composites."""

    def test_inline_composites(self):
        """Test that inline composites are working."""
        from satpy.composites import CompositorLoader
        cl_ = CompositorLoader()
        cl_.load_sensor_composites('visir')
        comps = cl_.compositors
        # Check that "fog" product has all its prerequisites defined
        keys = comps['visir'].keys()
        fog = [comps['visir'][dsid] for dsid in keys if "fog" == dsid.name][0]
        self.assertEqual(fog.attrs['prerequisites'][0], 'fog_dep_0')
        self.assertEqual(fog.attrs['prerequisites'][1], 'fog_dep_1')
        self.assertEqual(fog.attrs['prerequisites'][2], 10.8)

        # Check that the sub-composite dependencies use wavelengths
        # (numeric values)
        keys = comps['visir'].keys()
        fog_dep_ids = [dsid for dsid in keys if "fog_dep" in dsid.name]
        self.assertEqual(comps['visir'][fog_dep_ids[0]].attrs['prerequisites'],
                         [12.0, 10.8])
        self.assertEqual(comps['visir'][fog_dep_ids[1]].attrs['prerequisites'],
                         [10.8, 8.7])

        # Check the same for SEVIRI and verify channel names are used
        # in the sub-composite dependencies instead of wavelengths
        cl_ = CompositorLoader()
        cl_.load_sensor_composites('seviri')
        comps = cl_.compositors
        keys = comps['seviri'].keys()
        fog_dep_ids = [dsid for dsid in keys if "fog_dep" in dsid.name]
        self.assertEqual(comps['seviri'][fog_dep_ids[0]].attrs['prerequisites'],
                         ['IR_120', 'IR_108'])
        self.assertEqual(comps['seviri'][fog_dep_ids[1]].attrs['prerequisites'],
                         ['IR_108', 'IR_087'])


def suite():
    """Test suite for all reader tests"""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTests(test_abi.suite())
    mysuite.addTests(test_ahi.suite())
    mysuite.addTests(test_viirs.suite())
    mysuite.addTest(loader.loadTestsFromTestCase(TestCheckArea))
    mysuite.addTest(loader.loadTestsFromTestCase(TestDayNightCompositor))
    mysuite.addTest(loader.loadTestsFromTestCase(TestFillingCompositor))
    mysuite.addTest(loader.loadTestsFromTestCase(TestSandwichCompositor))
    mysuite.addTest(loader.loadTestsFromTestCase(TestLuminanceSharpeningCompositor))
    mysuite.addTest(loader.loadTestsFromTestCase(TestInlineComposites))

    return mysuite


if __name__ == '__main__':
    unittest.main()
