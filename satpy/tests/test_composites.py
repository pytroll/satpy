#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018-2020 Satpy developers
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
"""Tests for compositors in composites/__init__.py."""

import unittest
from datetime import datetime
from unittest import mock

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr


class TestMatchDataArrays(unittest.TestCase):
    """Test the utility method 'match_data_arrays'."""

    def _get_test_ds(self, shape=(50, 100), dims=('y', 'x')):
        """Get a fake DataArray."""
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
        ret_datasets = comp.match_data_arrays((ds1,))
        self.assertIs(ret_datasets[0], ds1)

    def test_mult_ds_area(self):
        """Test multiple datasets successfully pass."""
        from satpy.composites import CompositeBase
        ds1 = self._get_test_ds()
        ds2 = self._get_test_ds()
        comp = CompositeBase('test_comp')
        ret_datasets = comp.match_data_arrays((ds1, ds2))
        self.assertIs(ret_datasets[0], ds1)
        self.assertIs(ret_datasets[1], ds2)

    def test_mult_ds_no_area(self):
        """Test that all datasets must have an area attribute."""
        from satpy.composites import CompositeBase
        ds1 = self._get_test_ds()
        ds2 = self._get_test_ds()
        del ds2.attrs['area']
        comp = CompositeBase('test_comp')
        self.assertRaises(ValueError, comp.match_data_arrays, (ds1, ds2))

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
        self.assertRaises(IncompatibleAreas, comp.match_data_arrays, (ds1, ds2))

    def test_mult_ds_diff_dims(self):
        """Test that datasets with different dimensions still pass."""
        from satpy.composites import CompositeBase
        # x is still 50, y is still 100, even though they are in
        # different order
        ds1 = self._get_test_ds(shape=(50, 100), dims=('y', 'x'))
        ds2 = self._get_test_ds(shape=(3, 100, 50), dims=('bands', 'x', 'y'))
        comp = CompositeBase('test_comp')
        ret_datasets = comp.match_data_arrays((ds1, ds2))
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
        self.assertRaises(IncompatibleAreas, comp.match_data_arrays, (ds1, ds2))

    def test_nondimensional_coords(self):
        """Test the removal of non-dimensional coordinates when compositing."""
        from satpy.composites import CompositeBase
        ds = self._get_test_ds(shape=(2, 2))
        ds['acq_time'] = ('y', [0, 1])
        comp = CompositeBase('test_comp')
        ret_datasets = comp.match_data_arrays([ds, ds])
        self.assertNotIn('acq_time', ret_datasets[0].coords)


class TestRatioSharpenedCompositors(unittest.TestCase):
    """Test RatioSharpenedRGB and SelfSharpendRGB compositors."""

    def setUp(self):
        """Create test data."""
        from pyresample.geometry import AreaDefinition
        area = AreaDefinition('test', 'test', 'test',
                              {'proj': 'merc'}, 2, 2,
                              (-2000, -2000, 2000, 2000))
        attrs = {'area': area,
                 'start_time': datetime(2018, 1, 1, 18),
                 'modifiers': tuple(),
                 'resolution': 1000,
                 'name': 'test_vis'}
        ds1 = xr.DataArray(da.ones((2, 2), chunks=2, dtype=np.float64),
                           attrs=attrs, dims=('y', 'x'),
                           coords={'y': [0, 1], 'x': [0, 1]})
        self.ds1 = ds1
        ds2 = xr.DataArray(da.ones((2, 2), chunks=2, dtype=np.float64) + 2,
                           attrs=attrs, dims=('y', 'x'),
                           coords={'y': [0, 1], 'x': [0, 1]})
        ds2.attrs['name'] += '2'
        self.ds2 = ds2
        ds3 = xr.DataArray(da.ones((2, 2), chunks=2, dtype=np.float64) + 3,
                           attrs=attrs, dims=('y', 'x'),
                           coords={'y': [0, 1], 'x': [0, 1]})
        ds3.attrs['name'] += '3'
        self.ds3 = ds3
        ds4 = xr.DataArray(da.ones((2, 2), chunks=2, dtype=np.float64) + 4,
                           attrs=attrs, dims=('y', 'x'),
                           coords={'y': [0, 1], 'x': [0, 1]})
        ds4.attrs['name'] += '4'
        ds4.attrs['resolution'] = 500
        self.ds4 = ds4

        # high res version
        ds4 = xr.DataArray(da.ones((4, 4), chunks=2, dtype=np.float64) + 4,
                           attrs=attrs.copy(), dims=('y', 'x'),
                           coords={'y': [0, 1, 2, 3], 'x': [0, 1, 2, 3]})
        ds4.attrs['name'] += '4'
        ds4.attrs['resolution'] = 500
        ds4.attrs['rows_per_scan'] = 1
        ds4.attrs['area'] = AreaDefinition('test', 'test', 'test',
                                           {'proj': 'merc'}, 4, 4,
                                           (-2000, -2000, 2000, 2000))
        self.ds4_big = ds4

    def test_bad_color(self):
        """Test that only valid band colors can be provided."""
        from satpy.composites import RatioSharpenedRGB
        self.assertRaises(ValueError, RatioSharpenedRGB, name='true_color', high_resolution_band='bad')

    def test_match_data_arrays(self):
        """Test that all of the areas have to be the same resolution."""
        from satpy.composites import RatioSharpenedRGB, IncompatibleAreas
        comp = RatioSharpenedRGB(name='true_color')
        self.assertRaises(IncompatibleAreas, comp, (self.ds1, self.ds2, self.ds3), optional_datasets=(self.ds4_big,))

    def test_more_than_three_datasets(self):
        """Test that only 3 datasets can be passed."""
        from satpy.composites import RatioSharpenedRGB
        comp = RatioSharpenedRGB(name='true_color')
        self.assertRaises(ValueError, comp, (self.ds1, self.ds2, self.ds3, self.ds1),
                          optional_datasets=(self.ds4_big,))

    def test_basic_no_high_res(self):
        """Test that three datasets can be passed without optional high res."""
        from satpy.composites import RatioSharpenedRGB
        comp = RatioSharpenedRGB(name='true_color')
        res = comp((self.ds1, self.ds2, self.ds3))
        self.assertEqual(res.shape, (3, 2, 2))

    def test_basic_no_sharpen(self):
        """Test that color None does no sharpening."""
        from satpy.composites import RatioSharpenedRGB
        comp = RatioSharpenedRGB(name='true_color', high_resolution_band=None)
        res = comp((self.ds1, self.ds2, self.ds3), optional_datasets=(self.ds4,))
        self.assertEqual(res.shape, (3, 2, 2))

    def test_basic_red(self):
        """Test that basic high resolution red can be passed."""
        from satpy.composites import RatioSharpenedRGB
        comp = RatioSharpenedRGB(name='true_color')
        res = comp((self.ds1, self.ds2, self.ds3), optional_datasets=(self.ds4,))
        res = res.values
        self.assertEqual(res.shape, (3, 2, 2))
        np.testing.assert_allclose(res[0], self.ds4.values)
        np.testing.assert_allclose(res[1], np.array([[4.5, 4.5], [4.5, 4.5]], dtype=np.float64))
        np.testing.assert_allclose(res[2], np.array([[6, 6], [6, 6]], dtype=np.float64))

    def test_self_sharpened_no_high_res(self):
        """Test for exception when no high res band is specified."""
        from satpy.composites import SelfSharpenedRGB
        comp = SelfSharpenedRGB(name='true_color', high_resolution_band=None)
        self.assertRaises(ValueError, comp, (self.ds1, self.ds2, self.ds3))

    def test_self_sharpened_basic(self):
        """Test that three datasets can be passed without optional high res."""
        from satpy.composites import SelfSharpenedRGB
        comp = SelfSharpenedRGB(name='true_color')
        res = comp((self.ds1, self.ds2, self.ds3))
        res = res.values
        self.assertEqual(res.shape, (3, 2, 2))
        np.testing.assert_allclose(res[0], self.ds1.values)
        np.testing.assert_allclose(res[1], np.array([[3, 3], [3, 3]], dtype=np.float64))
        np.testing.assert_allclose(res[2], np.array([[4, 4], [4, 4]], dtype=np.float64))


class TestSunZenithCorrector(unittest.TestCase):
    """Test case for the zenith corrector."""

    def setUp(self):
        """Create test data."""
        from pyresample.geometry import AreaDefinition
        area = AreaDefinition('test', 'test', 'test',
                              {'proj': 'merc'}, 2, 2,
                              (-2000, -2000, 2000, 2000))
        bigger_area = AreaDefinition('test', 'test', 'test',
                                     {'proj': 'merc'}, 4, 4,
                                     (-2000, -2000, 2000, 2000))
        attrs = {'area': area,
                 'start_time': datetime(2018, 1, 1, 18),
                 'modifiers': tuple(),
                 'name': 'test_vis'}
        ds1 = xr.DataArray(da.ones((2, 2), chunks=2, dtype=np.float64),
                           attrs=attrs, dims=('y', 'x'),
                           coords={'y': [0, 1], 'x': [0, 1]})
        self.ds1 = ds1
        ds2 = xr.DataArray(da.ones((4, 4), chunks=2, dtype=np.float64),
                           attrs=attrs, dims=('y', 'x'),
                           coords={'y': [0, 0.5, 1, 1.5], 'x': [0, 0.5, 1, 1.5]})
        ds2.attrs['area'] = bigger_area
        self.ds2 = ds2
        self.sza = xr.DataArray(
            np.rad2deg(np.arccos(da.from_array([[0.0149581333, 0.0146694376], [0.0150812684, 0.0147925727]],
                                               chunks=2))),
            attrs={'area': area},
            dims=('y', 'x'),
            coords={'y': [0, 1], 'x': [0, 1]},
        )

    def test_basic_default_not_provided(self):
        """Test default limits when SZA isn't provided."""
        from satpy.composites import SunZenithCorrector
        comp = SunZenithCorrector(name='sza_test', modifiers=tuple())
        res = comp((self.ds1,), test_attr='test')
        np.testing.assert_allclose(res.values, np.array([[22.401667, 22.31777], [22.437503, 22.353533]]))
        self.assertIn('y', res.coords)
        self.assertIn('x', res.coords)
        ds1 = self.ds1.copy().drop_vars(('y', 'x'))
        res = comp((ds1,), test_attr='test')
        np.testing.assert_allclose(res.values, np.array([[22.401667, 22.31777], [22.437503, 22.353533]]))
        self.assertNotIn('y', res.coords)
        self.assertNotIn('x', res.coords)

    def test_basic_lims_not_provided(self):
        """Test custom limits when SZA isn't provided."""
        from satpy.composites import SunZenithCorrector
        comp = SunZenithCorrector(name='sza_test', modifiers=tuple(), correction_limit=90)
        res = comp((self.ds1,), test_attr='test')
        np.testing.assert_allclose(res.values, np.array([[66.853262, 68.168939], [66.30742, 67.601493]]))

    def test_basic_default_provided(self):
        """Test default limits when SZA is provided."""
        from satpy.composites import SunZenithCorrector
        comp = SunZenithCorrector(name='sza_test', modifiers=tuple())
        res = comp((self.ds1, self.sza), test_attr='test')
        np.testing.assert_allclose(res.values, np.array([[22.401667, 22.31777], [22.437503, 22.353533]]))

    def test_basic_lims_provided(self):
        """Test custom limits when SZA is provided."""
        from satpy.composites import SunZenithCorrector
        comp = SunZenithCorrector(name='sza_test', modifiers=tuple(), correction_limit=90)
        res = comp((self.ds1, self.sza), test_attr='test')
        np.testing.assert_allclose(res.values, np.array([[66.853262, 68.168939], [66.30742, 67.601493]]))

    def test_imcompatible_areas(self):
        """Test sunz correction on incompatible areas."""
        from satpy.composites import SunZenithCorrector, IncompatibleAreas
        comp = SunZenithCorrector(name='sza_test', modifiers=tuple(), correction_limit=90)
        with pytest.raises(IncompatibleAreas):
            comp((self.ds2, self.sza), test_attr='test')


class TestDifferenceCompositor(unittest.TestCase):
    """Test case for the difference compositor."""

    def setUp(self):
        """Create test data."""
        from pyresample.geometry import AreaDefinition
        area = AreaDefinition('test', 'test', 'test',
                              {'proj': 'merc'}, 2, 2,
                              (-2000, -2000, 2000, 2000))
        attrs = {'area': area,
                 'start_time': datetime(2018, 1, 1, 18),
                 'modifiers': tuple(),
                 'resolution': 1000,
                 'name': 'test_vis'}
        ds1 = xr.DataArray(da.ones((2, 2), chunks=2, dtype=np.float64),
                           attrs=attrs, dims=('y', 'x'),
                           coords={'y': [0, 1], 'x': [0, 1]})
        self.ds1 = ds1
        ds2 = xr.DataArray(da.ones((2, 2), chunks=2, dtype=np.float64) + 2,
                           attrs=attrs, dims=('y', 'x'),
                           coords={'y': [0, 1], 'x': [0, 1]})
        ds2.attrs['name'] += '2'
        self.ds2 = ds2

        # high res version
        ds2 = xr.DataArray(da.ones((4, 4), chunks=2, dtype=np.float64) + 4,
                           attrs=attrs.copy(), dims=('y', 'x'),
                           coords={'y': [0, 1, 2, 3], 'x': [0, 1, 2, 3]})
        ds2.attrs['name'] += '2'
        ds2.attrs['resolution'] = 500
        ds2.attrs['rows_per_scan'] = 1
        ds2.attrs['area'] = AreaDefinition('test', 'test', 'test',
                                           {'proj': 'merc'}, 4, 4,
                                           (-2000, -2000, 2000, 2000))
        self.ds2_big = ds2

    def test_basic_diff(self):
        """Test that a basic difference composite works."""
        from satpy.composites import DifferenceCompositor
        comp = DifferenceCompositor(name='diff')
        res = comp((self.ds1, self.ds2))
        np.testing.assert_allclose(res.values, -2)

    def test_bad_areas_diff(self):
        """Test that a difference where resolutions are different fails."""
        from satpy.composites import DifferenceCompositor, IncompatibleAreas
        comp = DifferenceCompositor(name='diff')
        # too many arguments
        self.assertRaises(ValueError, comp, (self.ds1, self.ds2, self.ds2_big))
        # different resolution
        self.assertRaises(IncompatibleAreas, comp, (self.ds1, self.ds2_big))


class TestDayNightCompositor(unittest.TestCase):
    """Test DayNightCompositor."""

    def setUp(self):
        """Create test data."""
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
        my_area.get_lonlats.return_value = (lons, lats)
        self.data_a.attrs['area'] = my_area
        self.data_b.attrs['area'] = my_area
        # not used except to check that it matches the data arrays
        self.sza.attrs['area'] = my_area

    def test_basic_sza(self):
        """Test compositor when SZA data is included."""
        from satpy.composites import DayNightCompositor
        comp = DayNightCompositor(name='dn_test')
        res = comp((self.data_a, self.data_b, self.sza))
        res = res.compute()
        expected = np.array([[0., 0.22122352], [0.5, 1.]])
        np.testing.assert_allclose(res.values[0], expected)

    def test_basic_area(self):
        """Test compositor when SZA data is not provided."""
        from satpy.composites import DayNightCompositor
        comp = DayNightCompositor(name='dn_test')
        res = comp((self.data_a, self.data_b))
        res = res.compute()
        expected = np.array([[0., 0.33164983], [0.66835017, 1.]])
        np.testing.assert_allclose(res.values[0], expected)


class TestFillingCompositor(unittest.TestCase):
    """Test case for the filling compositor."""

    def test_fill(self):
        """Test filling."""
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
        from satpy.composites import SandwichCompositor

        rgb_arr = da.from_array(np.random.random((3, 2, 2)), chunks=2)
        rgb = xr.DataArray(rgb_arr, dims=['bands', 'y', 'x'])
        lum_arr = da.from_array(100 * np.random.random((2, 2)), chunks=2)
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
        self.assertEqual(fog.attrs['prerequisites'][0], '_fog_dep_0')
        self.assertEqual(fog.attrs['prerequisites'][1], '_fog_dep_1')
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


class TestNIRReflectance(unittest.TestCase):
    """Test NIR reflectance compositor."""

    @mock.patch('satpy.composites.sun_zenith_angle')
    @mock.patch('satpy.composites.NIRReflectance.apply_modifier_info')
    @mock.patch('satpy.composites.Calculator')
    def test_compositor(self, calculator, apply_modifier_info, sza):
        """Test NIR reflectance compositor."""
        import numpy as np
        import xarray as xr
        import dask.array as da
        refl_arr = np.random.random((2, 2))
        refl = da.from_array(refl_arr)
        refl_from_tbs = mock.MagicMock()
        refl_from_tbs.return_value = refl
        calculator.return_value = mock.MagicMock(
            reflectance_from_tbs=refl_from_tbs)

        from satpy.composites import NIRReflectance

        nir_arr = np.random.random((2, 2))
        nir = xr.DataArray(da.from_array(nir_arr), dims=['y', 'x'])
        platform = 'Meteosat-11'
        sensor = 'seviri'
        chan_name = 'IR_039'
        nir.attrs['platform_name'] = platform
        nir.attrs['sensor'] = sensor
        nir.attrs['name'] = chan_name
        get_lonlats = mock.MagicMock()
        lons, lats = 1, 2
        get_lonlats.return_value = (lons, lats)
        area = mock.MagicMock(get_lonlats=get_lonlats)
        nir.attrs['area'] = area
        start_time = 1
        nir.attrs['start_time'] = start_time
        ir_arr = 100 * np.random.random((2, 2))
        ir_ = xr.DataArray(da.from_array(ir_arr), dims=['y', 'x'])
        ir_.attrs['area'] = area
        sunz_arr = 100 * np.random.random((2, 2))
        sunz = xr.DataArray(da.from_array(sunz_arr), dims=['y', 'x'])
        sunz.attrs['standard_name'] = 'solar_zenith_angle'
        sunz.attrs['area'] = area
        sunz2 = da.from_array(sunz_arr)
        sza.return_value = sunz2

        comp = NIRReflectance(name='test')
        info = {'modifiers': None}
        res = comp([nir, ir_], optional_datasets=[sunz], **info)
        self.assertEqual(res.attrs['units'], '%')
        self.assertEqual(res.attrs['platform_name'], platform)
        self.assertEqual(res.attrs['sensor'], sensor)
        self.assertEqual(res.attrs['name'], chan_name)
        self.assertEqual(res.attrs['sunz_threshold'], None)
        calculator.assert_called()
        calculator.assert_called_with('Meteosat-11', 'seviri', 'IR_039', sunz_threshold=None)
        self.assertTrue(apply_modifier_info.call_args[0][0] is nir)
        self.assertTrue(comp._refl3x is calculator.return_value)
        refl_from_tbs.reset_mock()

        res = comp([nir, ir_], optional_datasets=[], **info)
        get_lonlats.assert_called()
        sza.assert_called_with(start_time, lons, lats)
        refl_from_tbs.assert_called_with(sunz2, nir.data, ir_.data, tb_ir_co2=None)
        refl_from_tbs.reset_mock()

        co2_arr = np.random.random((2, 2))
        co2 = xr.DataArray(da.from_array(co2_arr), dims=['y', 'x'])
        co2.attrs['wavelength'] = [12.0, 13.0, 14.0]
        co2.attrs['units'] = 'K'
        res = comp([nir, ir_], optional_datasets=[co2], **info)
        refl_from_tbs.assert_called_with(sunz2, nir.data, ir_.data, tb_ir_co2=co2.data)

        comp = NIRReflectance(name='test', sunz_threshold=84.0)
        info = {'modifiers': None}
        res = comp([nir, ir_], optional_datasets=[sunz], **info)
        self.assertEqual(res.attrs['sunz_threshold'], 84.0)
        calculator.assert_called_with('Meteosat-11', 'seviri', 'IR_039', sunz_threshold=84.0)


class TestNIREmissivePartFromReflectance(unittest.TestCase):
    """Test the NIR Emissive part from reflectance compositor."""

    @mock.patch('satpy.composites.sun_zenith_angle')
    @mock.patch('satpy.composites.NIREmissivePartFromReflectance.apply_modifier_info')
    @mock.patch('satpy.composites.Calculator')
    def test_compositor(self, calculator, apply_modifier_info, sza):
        """Test the NIR emissive part from reflectance compositor."""
        import numpy as np
        import xarray as xr
        import dask.array as da

        refl_arr = np.random.random((2, 2))
        refl = da.from_array(refl_arr)

        refl_from_tbs = mock.MagicMock()
        refl_from_tbs.return_value = refl
        calculator.return_value = mock.MagicMock(reflectance_from_tbs=refl_from_tbs)

        emissive_arr = np.random.random((2, 2))
        emissive = da.from_array(emissive_arr)
        emissive_part = mock.MagicMock()
        emissive_part.return_value = emissive
        calculator.return_value = mock.MagicMock(emissive_part_3x=emissive_part)

        from satpy.composites import NIREmissivePartFromReflectance

        comp = NIREmissivePartFromReflectance(name='test', sunz_threshold=86.0)
        info = {'modifiers': None}

        platform = 'NOAA-20'
        sensor = 'viirs'
        chan_name = 'M12'

        get_lonlats = mock.MagicMock()
        lons, lats = 1, 2
        get_lonlats.return_value = (lons, lats)
        area = mock.MagicMock(get_lonlats=get_lonlats)

        nir_arr = np.random.random((2, 2))
        nir = xr.DataArray(da.from_array(nir_arr), dims=['y', 'x'])
        nir.attrs['platform_name'] = platform
        nir.attrs['sensor'] = sensor
        nir.attrs['name'] = chan_name
        nir.attrs['area'] = area
        ir_arr = np.random.random((2, 2))
        ir_ = xr.DataArray(da.from_array(ir_arr), dims=['y', 'x'])
        ir_.attrs['area'] = area

        sunz_arr = 100 * np.random.random((2, 2))
        sunz = xr.DataArray(da.from_array(sunz_arr), dims=['y', 'x'])
        sunz.attrs['standard_name'] = 'solar_zenith_angle'
        sunz.attrs['area'] = area
        sunz2 = da.from_array(sunz_arr)
        sza.return_value = sunz2

        res = comp([nir, ir_], optional_datasets=[sunz], **info)
        self.assertEqual(res.attrs['sunz_threshold'], 86.0)
        self.assertEqual(res.attrs['units'], 'K')
        self.assertEqual(res.attrs['platform_name'], platform)
        self.assertEqual(res.attrs['sensor'], sensor)
        self.assertEqual(res.attrs['name'], chan_name)
        calculator.assert_called_with('NOAA-20', 'viirs', 'M12', sunz_threshold=86.0)


class TestColormapCompositor(unittest.TestCase):
    """Test the ColormapCompositor."""

    def test_build_colormap(self):
        """Test colormap building."""
        from satpy.composites import ColormapCompositor
        cmap_comp = ColormapCompositor('test_cmap_compositor')
        palette = np.array([[0, 0, 0], [127, 127, 127], [255, 255, 255]])
        cmap, sqpal = cmap_comp.build_colormap(palette, np.uint8, {})
        self.assertTrue(np.allclose(cmap.values, [0, 1]))
        self.assertTrue(np.allclose(sqpal, palette / 255.0))

        palette = xr.DataArray(np.array([[0, 0, 0], [127, 127, 127], [255, 255, 255]]),
                               dims=['value', 'band'])
        palette.attrs['palette_meanings'] = [2, 3, 4]
        cmap, sqpal = cmap_comp.build_colormap(palette, np.uint8, {})
        self.assertTrue(np.allclose(cmap.values, [2, 3, 4]))
        self.assertTrue(np.allclose(sqpal, palette / 255.0))


class TestPaletteCompositor(unittest.TestCase):
    """Test the PaletteCompositor."""

    def test_call(self):
        """Test palette compositing."""
        from satpy.composites import PaletteCompositor
        cmap_comp = PaletteCompositor('test_cmap_compositor')
        palette = xr.DataArray(np.array([[0, 0, 0], [127, 127, 127], [255, 255, 255]]),
                               dims=['value', 'band'])
        palette.attrs['palette_meanings'] = [2, 3, 4]

        data = xr.DataArray(np.array([[4, 3, 2], [2, 3, 4]], dtype=np.uint8), dims=['y', 'x'])
        res = cmap_comp([data, palette])
        exp = np.array([[[1., 0.498039, 0.],
                         [0., 0.498039, 1.]],
                        [[1., 0.498039, 0.],
                         [0., 0.498039, 1.]],
                        [[1., 0.498039, 0.],
                         [0., 0.498039, 1.]]])
        self.assertTrue(np.allclose(res, exp))


class TestCloudTopHeightCompositor(unittest.TestCase):
    """Test the CloudTopHeightCompositor."""

    def test_call(self):
        """Test the CloudTopHeight composite generation."""
        from satpy.composites.cloud_products import CloudTopHeightCompositor
        cmap_comp = CloudTopHeightCompositor('test_cmap_compositor')
        palette = xr.DataArray(np.array([[0, 0, 0], [127, 127, 127], [255, 255, 255]]),
                               dims=['value', 'band'])
        palette.attrs['palette_meanings'] = [2, 3, 4]
        status = xr.DataArray(np.array([[1, 0, 1], [1, 0, 65535]]), dims=['y', 'x'],
                              attrs={'_FillValue': 65535})
        data = xr.DataArray(np.array([[4, 3, 2], [2, 3, 4]], dtype=np.uint8),
                            dims=['y', 'x'])
        res = cmap_comp([data, palette, status])
        exp = np.array([[[0., 0.49803922, 0.],
                         [0., 0.49803922, np.nan]],
                        [[0., 0.49803922, 0.],
                         [0., 0.49803922, np.nan]],
                        [[0., 0.49803922, 0.],
                         [0., 0.49803922, np.nan]]])
        np.testing.assert_allclose(res, exp)


class TestPrecipCloudsCompositor(unittest.TestCase):
    """Test the PrecipClouds compositor."""

    def test_call(self):
        """Test the precip composite generation."""
        from satpy.composites.cloud_products import PrecipCloudsRGB
        cmap_comp = PrecipCloudsRGB('test_precip_compositor')

        data_light = xr.DataArray(np.array([[80, 70, 60, 0], [20, 30, 40, 255]], dtype=np.uint8),
                                  dims=['y', 'x'], attrs={'_FillValue': 255})
        data_moderate = xr.DataArray(np.array([[60, 50, 40, 0], [20, 30, 40, 255]], dtype=np.uint8),
                                     dims=['y', 'x'], attrs={'_FillValue': 255})
        data_intense = xr.DataArray(np.array([[40, 30, 20, 0], [20, 30, 40, 255]], dtype=np.uint8),
                                    dims=['y', 'x'], attrs={'_FillValue': 255})
        data_flags = xr.DataArray(np.array([[0, 0, 4, 0], [0, 0, 0, 0]], dtype=np.uint8),
                                  dims=['y', 'x'])
        res = cmap_comp([data_light, data_moderate, data_intense, data_flags])

        exp = np.array([[[0.24313725, 0.18235294, 0.12156863, np.nan],
                         [0.12156863, 0.18235294, 0.24313725, np.nan]],
                        [[0.62184874, 0.51820728, 0.41456583, np.nan],
                         [0.20728291, 0.31092437, 0.41456583, np.nan]],
                        [[0.82913165, 0.7254902, 0.62184874, np.nan],
                         [0.20728291, 0.31092437, 0.41456583, np.nan]]])

        np.testing.assert_allclose(res, exp)


class TestSingleBandCompositor(unittest.TestCase):
    """Test the single-band compositor."""

    def setUp(self):
        """Create test data."""
        from satpy.composites import SingleBandCompositor
        self.comp = SingleBandCompositor(name='test')

        all_valid = np.ones((2, 2))
        self.all_valid = xr.DataArray(all_valid, dims=['y', 'x'])

    def test_call(self):
        """Test calling the compositor."""
        # Dataset with extra attributes
        all_valid = self.all_valid
        all_valid.attrs['sensor'] = 'foo'
        attrs = {'foo': 'bar', 'resolution': 333, 'units': 'K',
                 'calibration': 'BT', 'wavelength': 10.8}
        self.comp.attrs['resolution'] = None
        res = self.comp([self.all_valid], **attrs)
        # Verify attributes
        self.assertEqual(res.attrs.get('sensor'), 'foo')
        self.assertTrue('foo' in res.attrs)
        self.assertEqual(res.attrs.get('foo'), 'bar')
        self.assertTrue('units' in res.attrs)
        self.assertTrue('calibration' in res.attrs)
        self.assertFalse('modifiers' in res.attrs)
        self.assertEqual(res.attrs['wavelength'], 10.8)
        self.assertEqual(res.attrs['resolution'], 333)


class TestGenericCompositor(unittest.TestCase):
    """Test generic compositor."""

    def setUp(self):
        """Create test data."""
        from satpy.composites import GenericCompositor
        self.comp = GenericCompositor(name='test')
        self.comp2 = GenericCompositor(name='test2', common_channel_mask=False)

        all_valid = np.ones((1, 2, 2))
        self.all_valid = xr.DataArray(all_valid, dims=['bands', 'y', 'x'])
        first_invalid = np.reshape(np.array([np.nan, 1., 1., 1.]), (1, 2, 2))
        self.first_invalid = xr.DataArray(first_invalid,
                                          dims=['bands', 'y', 'x'])
        second_invalid = np.reshape(np.array([1., np.nan, 1., 1.]), (1, 2, 2))
        self.second_invalid = xr.DataArray(second_invalid,
                                           dims=['bands', 'y', 'x'])
        wrong_shape = np.reshape(np.array([1., 1., 1.]), (1, 3, 1))
        self.wrong_shape = xr.DataArray(wrong_shape, dims=['bands', 'y', 'x'])

    def test_masking(self):
        """Test masking in generic compositor."""
        # Single channel
        res = self.comp([self.all_valid])
        np.testing.assert_allclose(res.data, 1., atol=1e-9)
        # Three channels, one value invalid
        res = self.comp([self.all_valid, self.all_valid, self.first_invalid])
        correct = np.reshape(np.array([np.nan, 1., 1., 1.]), (2, 2))
        for i in range(3):
            np.testing.assert_almost_equal(res.data[i, :, :], correct)
        # Three channels, two values invalid
        res = self.comp([self.all_valid, self.first_invalid, self.second_invalid])
        correct = np.reshape(np.array([np.nan, np.nan, 1., 1.]), (2, 2))
        for i in range(3):
            np.testing.assert_almost_equal(res.data[i, :, :], correct)

    def test_concat_datasets(self):
        """Test concatenation of datasets."""
        from satpy.composites import IncompatibleAreas
        res = self.comp._concat_datasets([self.all_valid], 'L')
        num_bands = len(res.bands)
        self.assertEqual(num_bands, 1)
        self.assertEqual(res.shape[0], num_bands)
        self.assertTrue(res.bands[0] == 'L')
        res = self.comp._concat_datasets([self.all_valid, self.all_valid], 'LA')
        num_bands = len(res.bands)
        self.assertEqual(num_bands, 2)
        self.assertEqual(res.shape[0], num_bands)
        self.assertTrue(res.bands[0] == 'L')
        self.assertTrue(res.bands[1] == 'A')
        self.assertRaises(IncompatibleAreas, self.comp._concat_datasets,
                          [self.all_valid, self.wrong_shape], 'LA')

    def test_get_sensors(self):
        """Test getting sensors from the dataset attributes."""
        res = self.comp._get_sensors([self.all_valid])
        self.assertIsNone(res)
        dset1 = self.all_valid
        dset1.attrs['sensor'] = 'foo'
        res = self.comp._get_sensors([dset1])
        self.assertEqual(res, 'foo')
        dset2 = self.first_invalid
        dset2.attrs['sensor'] = 'bar'
        res = self.comp._get_sensors([dset1, dset2])
        self.assertTrue('foo' in res)
        self.assertTrue('bar' in res)
        self.assertEqual(len(res), 2)
        self.assertTrue(isinstance(res, set))

    @mock.patch('satpy.composites.GenericCompositor._get_sensors')
    @mock.patch('satpy.composites.combine_metadata')
    @mock.patch('satpy.composites.check_times')
    @mock.patch('satpy.composites.GenericCompositor.match_data_arrays')
    def test_call_with_mock(self, match_data_arrays, check_times, combine_metadata, get_sensors):
        """Test calling generic compositor."""
        from satpy.composites import IncompatibleAreas
        combine_metadata.return_value = dict()
        get_sensors.return_value = 'foo'
        # One dataset, no mode given
        res = self.comp([self.all_valid])
        self.assertEqual(res.shape[0], 1)
        self.assertEqual(res.attrs['mode'], 'L')
        match_data_arrays.assert_not_called()
        # This compositor has been initialized without common masking, so the
        # masking shouldn't have been called
        projectables = [self.all_valid, self.first_invalid, self.second_invalid]
        match_data_arrays.return_value = projectables
        res = self.comp2(projectables)
        match_data_arrays.assert_called_once()
        match_data_arrays.reset_mock()
        # Dataset for alpha given, so shouldn't be masked
        projectables = [self.all_valid, self.all_valid]
        match_data_arrays.return_value = projectables
        res = self.comp(projectables)
        match_data_arrays.assert_called_once()
        match_data_arrays.reset_mock()
        # When areas are incompatible, masking shouldn't happen
        match_data_arrays.side_effect = IncompatibleAreas()
        self.assertRaises(IncompatibleAreas,
                          self.comp, [self.all_valid, self.wrong_shape])
        match_data_arrays.assert_called_once()

    def test_call(self):
        """Test calling generic compositor."""
        # Multiple datasets with extra attributes
        all_valid = self.all_valid
        all_valid.attrs['sensor'] = 'foo'
        attrs = {'foo': 'bar', 'resolution': 333}
        self.comp.attrs['resolution'] = None
        res = self.comp([self.all_valid, self.first_invalid], **attrs)
        # Verify attributes
        self.assertEqual(res.attrs.get('sensor'), 'foo')
        self.assertTrue('foo' in res.attrs)
        self.assertEqual(res.attrs.get('foo'), 'bar')
        self.assertTrue('units' not in res.attrs)
        self.assertTrue('calibration' not in res.attrs)
        self.assertTrue('modifiers' not in res.attrs)
        self.assertIsNone(res.attrs['wavelength'])
        self.assertEqual(res.attrs['mode'], 'LA')
        self.assertEqual(res.attrs['resolution'], 333)


class TestAddBands(unittest.TestCase):
    """Test case for the `add_bands` function."""

    def test_add_bands(self):
        """Test adding bands."""
        from satpy.composites import add_bands
        import dask.array as da
        import numpy as np
        import xarray as xr

        # L + RGB -> RGB
        data = xr.DataArray(da.ones((1, 3, 3)), dims=('bands', 'y', 'x'),
                            coords={'bands': ['L']})
        new_bands = xr.DataArray(da.array(['R', 'G', 'B']), dims=('bands'),
                                 coords={'bands': ['R', 'G', 'B']})
        res = add_bands(data, new_bands)
        res_bands = ['R', 'G', 'B']
        self.assertEqual(res.mode, ''.join(res_bands))
        np.testing.assert_array_equal(res.bands, res_bands)
        np.testing.assert_array_equal(res.coords['bands'], res_bands)

        # L + RGBA -> RGBA
        data = xr.DataArray(da.ones((1, 3, 3)), dims=('bands', 'y', 'x'),
                            coords={'bands': ['L']}, attrs={'mode': 'L'})
        new_bands = xr.DataArray(da.array(['R', 'G', 'B', 'A']), dims=('bands'),
                                 coords={'bands': ['R', 'G', 'B', 'A']})
        res = add_bands(data, new_bands)
        res_bands = ['R', 'G', 'B', 'A']
        self.assertEqual(res.mode, ''.join(res_bands))
        np.testing.assert_array_equal(res.bands, res_bands)
        np.testing.assert_array_equal(res.coords['bands'], res_bands)

        # LA + RGB -> RGBA
        data = xr.DataArray(da.ones((2, 3, 3)), dims=('bands', 'y', 'x'),
                            coords={'bands': ['L', 'A']}, attrs={'mode': 'LA'})
        new_bands = xr.DataArray(da.array(['R', 'G', 'B']), dims=('bands'),
                                 coords={'bands': ['R', 'G', 'B']})
        res = add_bands(data, new_bands)
        res_bands = ['R', 'G', 'B', 'A']
        self.assertEqual(res.mode, ''.join(res_bands))
        np.testing.assert_array_equal(res.bands, res_bands)
        np.testing.assert_array_equal(res.coords['bands'], res_bands)

        # RGB + RGBA -> RGBA
        data = xr.DataArray(da.ones((3, 3, 3)), dims=('bands', 'y', 'x'),
                            coords={'bands': ['R', 'G', 'B']},
                            attrs={'mode': 'RGB'})
        new_bands = xr.DataArray(da.array(['R', 'G', 'B', 'A']), dims=('bands'),
                                 coords={'bands': ['R', 'G', 'B', 'A']})
        res = add_bands(data, new_bands)
        res_bands = ['R', 'G', 'B', 'A']
        self.assertEqual(res.mode, ''.join(res_bands))
        np.testing.assert_array_equal(res.bands, res_bands)
        np.testing.assert_array_equal(res.coords['bands'], res_bands)


class TestStaticImageCompositor(unittest.TestCase):
    """Test case for the static compositor."""

    @mock.patch('satpy.resample.get_area_def')
    def test_init(self, get_area_def):
        """Test the initializiation of static compositor."""
        from satpy.composites import StaticImageCompositor

        # No filename given raises ValueError
        with self.assertRaises(ValueError):
            comp = StaticImageCompositor("name")

        # No area defined
        comp = StaticImageCompositor("name", filename="foo.tif")
        self.assertEqual(comp.filename, "foo.tif")
        self.assertIsNone(comp.area)

        # Area defined
        get_area_def.return_value = "bar"
        comp = StaticImageCompositor("name", filename="foo.tif", area="euro4")
        self.assertEqual(comp.filename, "foo.tif")
        self.assertEqual(comp.area, "bar")
        get_area_def.assert_called_once_with("euro4")

    @mock.patch('satpy.Scene')
    def test_call(self, Scene):  # noqa
        """Test the static compositing."""
        from satpy.composites import StaticImageCompositor

        class MockScene(dict):
            def load(self, arg):
                pass

        img = mock.MagicMock()
        img.attrs = {}
        scn = MockScene()
        scn['image'] = img
        Scene.return_value = scn
        comp = StaticImageCompositor("name", filename="foo.tif", area="euro4")
        res = comp()
        Scene.assert_called_once_with(reader='generic_image',
                                      filenames=[comp.filename])
        self.assertTrue("start_time" in res.attrs)
        self.assertTrue("end_time" in res.attrs)
        self.assertIsNone(res.attrs['sensor'])
        self.assertTrue('modifiers' not in res.attrs)
        self.assertTrue('calibration' not in res.attrs)

        # Non-georeferenced image, no area given
        img.attrs.pop('area')
        comp = StaticImageCompositor("name", filename="foo.tif")
        with self.assertRaises(AttributeError):
            res = comp()

        # Non-georeferenced image, area given
        comp = StaticImageCompositor("name", filename="foo.tif", area='euro4')
        res = comp()
        self.assertEqual(res.attrs['area'].area_id, 'euro4')


def _enhance2dataset(dataset):
    """Mock the enhance2dataset to return the original data."""
    return dataset


class TestBackgroundCompositor(unittest.TestCase):
    """Test case for the background compositor."""

    @mock.patch('satpy.composites.enhance2dataset', _enhance2dataset)
    def test_call(self):
        """Test the background compositing."""
        from satpy.composites import BackgroundCompositor
        import numpy as np
        comp = BackgroundCompositor("name")

        # L mode images
        attrs = {'mode': 'L', 'area': 'foo'}
        foreground = xr.DataArray(np.array([[[1., 0.5],
                                             [0., np.nan]]]),
                                  dims=('bands', 'y', 'x'),
                                  coords={'bands': [c for c in attrs['mode']]},
                                  attrs=attrs)
        background = xr.DataArray(np.ones((1, 2, 2)), dims=('bands', 'y', 'x'),
                                  coords={'bands': [c for c in attrs['mode']]},
                                  attrs=attrs)
        res = comp([foreground, background])
        self.assertEqual(res.attrs['area'], 'foo')
        self.assertTrue(np.all(res == np.array([[1., 0.5], [0., 1.]])))
        self.assertEqual(res.attrs['mode'], 'L')

        # LA mode images
        attrs = {'mode': 'LA', 'area': 'foo'}
        foreground = xr.DataArray(np.array([[[1., 0.5],
                                             [0., np.nan]],
                                            [[0.5, 0.5],
                                             [0.5, 0.5]]]),
                                  dims=('bands', 'y', 'x'),
                                  coords={'bands': [c for c in attrs['mode']]},
                                  attrs=attrs)
        background = xr.DataArray(np.ones((2, 2, 2)), dims=('bands', 'y', 'x'),
                                  coords={'bands': [c for c in attrs['mode']]},
                                  attrs=attrs)
        res = comp([foreground, background])
        self.assertTrue(np.all(res == np.array([[1., 0.75], [0.5, 1.]])))
        self.assertEqual(res.attrs['mode'], 'LA')

        # RGB mode images
        attrs = {'mode': 'RGB', 'area': 'foo'}
        foreground = xr.DataArray(np.array([[[1., 0.5],
                                             [0., np.nan]],
                                            [[1., 0.5],
                                             [0., np.nan]],
                                            [[1., 0.5],
                                             [0., np.nan]]]),
                                  dims=('bands', 'y', 'x'),
                                  coords={'bands': [c for c in attrs['mode']]},
                                  attrs=attrs)
        background = xr.DataArray(np.ones((3, 2, 2)), dims=('bands', 'y', 'x'),
                                  coords={'bands': [c for c in attrs['mode']]},
                                  attrs=attrs)

        res = comp([foreground, background])
        self.assertTrue(np.all(res == np.array([[[1., 0.5], [0., 1.]],
                                                [[1., 0.5], [0., 1.]],
                                                [[1., 0.5], [0., 1.]]])))
        self.assertEqual(res.attrs['mode'], 'RGB')

        # RGBA mode images
        attrs = {'mode': 'RGBA', 'area': 'foo'}
        foreground = xr.DataArray(np.array([[[1., 0.5],
                                             [0., np.nan]],
                                            [[1., 0.5],
                                             [0., np.nan]],
                                            [[1., 0.5],
                                             [0., np.nan]],
                                            [[0.5, 0.5],
                                             [0.5, 0.5]]]),
                                  dims=('bands', 'y', 'x'),
                                  coords={'bands': [c for c in attrs['mode']]},
                                  attrs=attrs)
        background = xr.DataArray(np.ones((4, 2, 2)), dims=('bands', 'y', 'x'),
                                  coords={'bands': [c for c in attrs['mode']]},
                                  attrs=attrs)

        res = comp([foreground, background])
        self.assertTrue(np.all(res == np.array([[[1., 0.75], [0.5, 1.]],
                                                [[1., 0.75], [0.5, 1.]],
                                                [[1., 0.75], [0.5, 1.]]])))
        self.assertEqual(res.attrs['mode'], 'RGBA')

    @mock.patch('satpy.composites.enhance2dataset', _enhance2dataset)
    def test_multiple_sensors(self):
        """Test the background compositing from multiple sensor data."""
        from satpy.composites import BackgroundCompositor
        import numpy as np
        comp = BackgroundCompositor("name")

        # L mode images
        attrs = {'mode': 'L', 'area': 'foo'}
        foreground = xr.DataArray(np.array([[[1., 0.5],
                                             [0., np.nan]]]),
                                  dims=('bands', 'y', 'x'),
                                  coords={'bands': [c for c in attrs['mode']]},
                                  attrs=attrs.copy())
        foreground.attrs['sensor'] = 'abi'
        background = xr.DataArray(np.ones((1, 2, 2)), dims=('bands', 'y', 'x'),
                                  coords={'bands': [c for c in attrs['mode']]},
                                  attrs=attrs.copy())
        background.attrs['sensor'] = 'glm'
        res = comp([foreground, background])
        self.assertEqual(res.attrs['area'], 'foo')
        self.assertTrue(np.all(res == np.array([[1., 0.5], [0., 1.]])))
        self.assertEqual(res.attrs['mode'], 'L')
        self.assertEqual(res.attrs['sensor'], {'abi', 'glm'})


class TestPSPAtmosphericalCorrection(unittest.TestCase):
    """Test the pyspectral-based atmospheric correction modifier."""

    def setUp(self):
        """Patch in-class imports."""
        self.orbital = mock.MagicMock()
        modules = {
            'pyspectral.atm_correction_ir': mock.MagicMock(),
            'pyorbital.orbital': self.orbital,
        }
        self.module_patcher = mock.patch.dict('sys.modules', modules)
        self.module_patcher.start()

    def tearDown(self):
        """Unpatch in-class imports."""
        self.module_patcher.stop()

    @mock.patch('satpy.composites.PSPAtmosphericalCorrection.apply_modifier_info')
    @mock.patch('satpy.composites.get_satpos')
    def test_call(self, get_satpos, *mocks):
        """Test atmospherical correction."""
        from satpy.composites import PSPAtmosphericalCorrection

        # Patch methods
        get_satpos.return_value = 'sat_lon', 'sat_lat', 12345678
        self.orbital.get_observer_look.return_value = 0, 0
        area = mock.MagicMock()
        area.get_lonlats.return_value = 'lons', 'lats'
        band = mock.MagicMock(attrs={'area': area,
                                     'start_time': 'start_time',
                                     'name': 'name',
                                     'platform_name': 'platform',
                                     'sensor': 'sensor'})

        # Perform atmospherical correction
        psp = PSPAtmosphericalCorrection(name='dummy')
        psp(projectables=[band])

        # Check arguments of get_orbserver_look() call, especially the altitude
        # unit conversion from meters to kilometers
        self.orbital.get_observer_look.assert_called_with(
            'sat_lon', 'sat_lat', 12345.678, 'start_time', 'lons', 'lats', 0)


class TestPSPRayleighReflectance(unittest.TestCase):
    """Test the pyspectral-based rayleigh correction modifier."""

    def setUp(self):
        """Patch in-class imports."""
        self.astronomy = mock.MagicMock()
        self.orbital = mock.MagicMock()
        modules = {
            'pyorbital.astronomy': self.astronomy,
            'pyorbital.orbital': self.orbital,
        }
        self.module_patcher = mock.patch.dict('sys.modules', modules)
        self.module_patcher.start()

    def tearDown(self):
        """Unpatch in-class imports."""
        self.module_patcher.stop()

    @mock.patch('satpy.composites.get_satpos')
    def test_get_angles(self, get_satpos):
        """Test sun and satellite angle calculation."""
        from satpy.composites import PSPRayleighReflectance

        # Patch methods
        get_satpos.return_value = 'sat_lon', 'sat_lat', 12345678
        self.orbital.get_observer_look.return_value = 0, 0
        self.astronomy.get_alt_az.return_value = 0, 0
        area = mock.MagicMock()
        lons = np.zeros((5, 5))
        lons[1, 1] = np.inf
        lons = da.from_array(lons, chunks=5)
        lats = np.zeros((5, 5))
        lats[1, 1] = np.inf
        lats = da.from_array(lats, chunks=5)
        area.get_lonlats.return_value = (lons, lats)
        vis = mock.MagicMock(attrs={'area': area,
                                    'start_time': 'start_time'})

        # Compute angles
        psp = PSPRayleighReflectance(name='dummy')
        psp.get_angles(vis)

        # Check arguments of get_orbserver_look() call, especially the altitude
        # unit conversion from meters to kilometers
        self.orbital.get_observer_look.assert_called_once()
        args = self.orbital.get_observer_look.call_args[0]
        self.assertEqual(args[:4], ('sat_lon', 'sat_lat', 12345.678, 'start_time'))
        self.assertIsInstance(args[4], da.Array)
        self.assertIsInstance(args[5], da.Array)
        self.assertEqual(args[6], 0)


class TestMaskingCompositor(unittest.TestCase):
    """Test case for the simple masking compositor."""

    def test_init(self):
        """Test the initializiation of compositor."""
        from satpy.composites import MaskingCompositor

        # No transparency or conditions given raises ValueError
        with self.assertRaises(ValueError):
            comp = MaskingCompositor("name")

        # transparency defined
        transparency = {0: 100, 1: 50}
        conditions = [{'method': 'equal', 'value': 0, 'transparency': 100},
                      {'method': 'equal', 'value': 1, 'transparency': 50}]
        comp = MaskingCompositor("name", transparency=transparency.copy())
        assert not hasattr(comp, 'transparency')
        # Transparency should be converted to conditions
        assert comp.conditions == conditions

        # conditions defined
        comp = MaskingCompositor("name", conditions=conditions.copy())
        assert comp.conditions == conditions

    def test_get_flag_value(self):
        """Test reading flag value from attributes based on a name."""
        from satpy.composites import _get_flag_value

        flag_values = da.array([1, 2])
        mask = da.array([[1, 2, 2],
                         [2, 1, 2],
                         [2, 2, 1]])
        mask = xr.DataArray(mask, dims=['y', 'x'])
        flag_meanings = ['Cloud-free_land', 'Cloud-free_sea']
        mask.attrs['flag_meanings'] = flag_meanings
        mask.attrs['flag_values'] = flag_values

        assert _get_flag_value(mask, 'Cloud-free_land') == 1
        assert _get_flag_value(mask, 'Cloud-free_sea') == 2

        flag_meanings_str = 'Cloud-free_land Cloud-free_sea'
        mask.attrs['flag_meanings'] = flag_meanings_str
        assert _get_flag_value(mask, 'Cloud-free_land') == 1
        assert _get_flag_value(mask, 'Cloud-free_sea') == 2

    def test_call(self):
        """Test call the compositor."""
        from satpy.composites import MaskingCompositor
        from satpy.tests.utils import CustomScheduler

        flag_meanings = ['Cloud-free_land', 'Cloud-free_sea']
        flag_meanings_str = 'Cloud-free_land Cloud-free_sea'
        flag_values = da.array([1, 2])
        conditions_v1 = [{'method': 'equal',
                          'value': 'Cloud-free_land',
                          'transparency': 100},
                         {'method': 'equal',
                          'value': 'Cloud-free_sea',
                          'transparency': 50}]
        conditions_v2 = [{'method': 'equal',
                          'value': 1,
                          'transparency': 100},
                         {'method': 'equal',
                          'value': 2,
                          'transparency': 50}]
        conditions_v3 = [{'method': 'isnan',
                          'transparency': 100}]
        conditions_v4 = [{'method': 'absolute_import',
                          'transparency': 'satpy.resample'}]

        # 2D data array
        data = xr.DataArray(da.random.random((3, 3)), dims=['y', 'x'])

        # 2D CT data array
        ct_data = da.array([[1, 2, 2],
                            [2, 1, 2],
                            [2, 2, 1]])
        ct_data = xr.DataArray(ct_data, dims=['y', 'x'])
        ct_data.attrs['flag_meanings'] = flag_meanings
        ct_data.attrs['flag_values'] = flag_values

        reference_alpha = da.array([[0, 0.5, 0.5],
                                    [0.5, 0, 0.5],
                                    [0.5, 0.5, 0]])
        reference_alpha = xr.DataArray(reference_alpha, dims=['y', 'x'])
        # The data are set to NaN where ct is `1`
        reference_data = data.where(ct_data > 1)

        reference_alpha_v3 = da.array([[1., 0., 0.],
                                       [0., 1., 0.],
                                       [0., 0., 1.]])
        reference_alpha_v3 = xr.DataArray(reference_alpha_v3, dims=['y', 'x'])
        # The data are set to NaN where ct is NaN
        reference_data_v3 = data.where(ct_data == 1)

        # Test with numerical transparency data
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            comp = MaskingCompositor("name", conditions=conditions_v1)
            res = comp([data, ct_data])
        self.assertTrue(res.mode == 'LA')
        np.testing.assert_allclose(res.sel(bands='L'), reference_data)
        np.testing.assert_allclose(res.sel(bands='A'), reference_alpha)

        # Test with named fields
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            comp = MaskingCompositor("name", conditions=conditions_v2)
            res = comp([data, ct_data])
        self.assertTrue(res.mode == 'LA')
        np.testing.assert_allclose(res.sel(bands='L'), reference_data)
        np.testing.assert_allclose(res.sel(bands='A'), reference_alpha)

        # Test with named fields which are as a string in the mask attributes
        ct_data.attrs['flag_meanings'] = flag_meanings_str
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            comp = MaskingCompositor("name", conditions=conditions_v2)
            res = comp([data, ct_data])
        self.assertTrue(res.mode == 'LA')
        np.testing.assert_allclose(res.sel(bands='L'), reference_data)
        np.testing.assert_allclose(res.sel(bands='A'), reference_alpha)

        # Test "isnan" as method
        # Set ct data to NaN where it originally is 1
        ct_data_v3 = ct_data.where(ct_data == 1)
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            comp = MaskingCompositor("name", conditions=conditions_v3)
            res = comp([data, ct_data_v3])
        self.assertTrue(res.mode == 'LA')
        np.testing.assert_allclose(res.sel(bands='L'), reference_data_v3)
        np.testing.assert_allclose(res.sel(bands='A'), reference_alpha_v3)

        # Test "absolute_import" as method
        # This should raise AttributeError
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            comp = MaskingCompositor("name", conditions=conditions_v4)
            try:
                res = comp([data, ct_data_v3])
                raise ValueError("Tried to use 'np.absolute_import'")
            except AttributeError:
                pass

        # Test RGB dataset
        # 3D data array
        data = xr.DataArray(da.random.random((3, 3, 3)),
                            dims=['bands', 'y', 'x'],
                            coords={'bands': ['R', 'G', 'B'],
                                    'y': np.arange(3),
                                    'x': np.arange(3)})

        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            comp = MaskingCompositor("name", conditions=conditions_v1)
            res = comp([data, ct_data])
        self.assertTrue(res.mode == 'RGBA')
        np.testing.assert_allclose(res.sel(bands='R'),
                                   data.sel(bands='R').where(ct_data > 1))
        np.testing.assert_allclose(res.sel(bands='G'),
                                   data.sel(bands='G').where(ct_data > 1))
        np.testing.assert_allclose(res.sel(bands='B'),
                                   data.sel(bands='B').where(ct_data > 1))
        np.testing.assert_allclose(res.sel(bands='A'), reference_alpha)

        # Test RGBA dataset
        data = xr.DataArray(da.random.random((4, 3, 3)),
                            dims=['bands', 'y', 'x'],
                            coords={'bands': ['R', 'G', 'B', 'A'],
                                    'y': np.arange(3),
                                    'x': np.arange(3)})

        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            comp = MaskingCompositor("name", conditions=conditions_v2)
            res = comp([data, ct_data])
        self.assertTrue(res.mode == 'RGBA')
        np.testing.assert_allclose(res.sel(bands='R'),
                                   data.sel(bands='R').where(ct_data > 1))
        np.testing.assert_allclose(res.sel(bands='G'),
                                   data.sel(bands='G').where(ct_data > 1))
        np.testing.assert_allclose(res.sel(bands='B'),
                                   data.sel(bands='B').where(ct_data > 1))
        # The compositor should drop the original alpha band
        np.testing.assert_allclose(res.sel(bands='A'), reference_alpha)

        # incorrect method
        conditions = [{'method': 'foo', 'value': 0, 'transparency': 100}]
        comp = MaskingCompositor("name", conditions=conditions)
        with self.assertRaises(AttributeError):
            res = comp([data, ct_data])

        # too few projectables
        with self.assertRaises(ValueError):
            res = comp([data])


class TestNaturalEnhCompositor(unittest.TestCase):
    """Test NaturalEnh compositor."""

    def setUp(self):
        """Create channel data and set channel weights."""
        self.ch1 = xr.DataArray([1.0])
        self.ch2 = xr.DataArray([2.0])
        self.ch3 = xr.DataArray([3.0])
        self.ch16_w = 2.0
        self.ch08_w = 3.0
        self.ch06_w = 4.0

    @mock.patch('satpy.composites.NaturalEnh.__repr__')
    @mock.patch('satpy.composites.NaturalEnh.match_data_arrays')
    def test_natural_enh(self, match_data_arrays, repr_):
        """Test NaturalEnh compositor."""
        from satpy.composites import NaturalEnh
        repr_.return_value = ''
        projectables = [self.ch1, self.ch2, self.ch3]

        def temp_func(*args):
            return args[0]
        match_data_arrays.side_effect = temp_func
        comp = NaturalEnh("foo", ch16_w=self.ch16_w, ch08_w=self.ch08_w,
                          ch06_w=self.ch06_w)
        self.assertEqual(comp.ch16_w, self.ch16_w)
        self.assertEqual(comp.ch08_w, self.ch08_w)
        self.assertEqual(comp.ch06_w, self.ch06_w)
        res = comp(projectables)
        assert mock.call(projectables) in match_data_arrays.mock_calls
        correct = (self.ch16_w * projectables[0] +
                   self.ch08_w * projectables[1] +
                   self.ch06_w * projectables[2])
        self.assertEqual(res[0], correct)
        self.assertEqual(res[1], projectables[1])
        self.assertEqual(res[2], projectables[2])
