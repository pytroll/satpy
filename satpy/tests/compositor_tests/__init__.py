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

import xarray as xr
import dask.array as da
import numpy as np
from datetime import datetime
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

    def test_check_areas(self):
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
    def setUp(self):
        """Create test data."""
        from pyresample.geometry import AreaDefinition
        area = AreaDefinition('test', 'test', 'test',
                              {'proj': 'merc'}, 2, 2,
                              (-2000, -2000, 2000, 2000))
        attrs = {'area': area,
                 'start_time': datetime(2018, 1, 1, 18),
                 'modifiers': tuple(),
                 'name': 'test_vis'}
        ds1 = xr.DataArray(da.ones((2, 2), chunks=2, dtype=np.float64),
                           attrs=attrs, dims=('y', 'x'),
                           coords={'y': [0, 1], 'x': [0, 1]})
        self.ds1 = ds1
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


class TestDifferenceCompositor(unittest.TestCase):

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
        my_area.get_lonlats_dask.return_value = (lons, lats)
        self.data_a.attrs['area'] = my_area
        self.data_b.attrs['area'] = my_area
        # not used except to check that it matches the data arrays
        self.sza.attrs['area'] = my_area

    def test_basic_sza(self):
        """Test compositor when SZA data is included"""
        from satpy.composites import DayNightCompositor
        comp = DayNightCompositor(name='dn_test')
        res = comp((self.data_a, self.data_b, self.sza))
        res = res.compute()
        expected = np.array([[0., 0.2985455], [0.51680423, 1.]])
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

    def test_fill(self):
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


class TestColormapCompositor(unittest.TestCase):
    """Test the ColormapCompositor."""

    def test_build_colormap(self):
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
        from satpy.composites.cloud_products import CloudTopHeightCompositor
        cmap_comp = CloudTopHeightCompositor('test_cmap_compositor')
        palette = xr.DataArray(np.array([[0, 0, 0], [127, 127, 127], [255, 255, 255]]),
                               dims=['value', 'band'])
        palette.attrs['palette_meanings'] = [2, 3, 4]
        status = np.array([1, 0, 1])
        data = xr.DataArray(np.array([[4, 3, 2], [2, 3, 4]], dtype=np.uint8), dims=['y', 'x'])
        res = cmap_comp([data, palette, status])
        exp = np.array([[[0., 0.498039, 0.],
                         [0., 0.498039, 0.]],
                        [[0., 0.498039, 0.],
                         [0., 0.498039, 0.]],
                        [[0., 0.498039, 0.],
                         [0., 0.498039, 0.]]])
        self.assertTrue(np.allclose(res, exp))


def suite():
    """Test suite for all reader tests."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTests(test_abi.suite())
    mysuite.addTests(test_ahi.suite())
    mysuite.addTests(test_viirs.suite())
    mysuite.addTest(loader.loadTestsFromTestCase(TestCheckArea))
    mysuite.addTest(loader.loadTestsFromTestCase(TestRatioSharpenedCompositors))
    mysuite.addTest(loader.loadTestsFromTestCase(TestSunZenithCorrector))
    mysuite.addTest(loader.loadTestsFromTestCase(TestDifferenceCompositor))
    mysuite.addTest(loader.loadTestsFromTestCase(TestDayNightCompositor))
    mysuite.addTest(loader.loadTestsFromTestCase(TestFillingCompositor))
    mysuite.addTest(loader.loadTestsFromTestCase(TestSandwichCompositor))
    mysuite.addTest(loader.loadTestsFromTestCase(TestLuminanceSharpeningCompositor))
    mysuite.addTest(loader.loadTestsFromTestCase(TestInlineComposites))
    mysuite.addTest(loader.loadTestsFromTestCase(TestColormapCompositor))
    mysuite.addTest(loader.loadTestsFromTestCase(TestPaletteCompositor))
    mysuite.addTest(loader.loadTestsFromTestCase(TestCloudTopHeightCompositor))

    return mysuite


if __name__ == '__main__':
    unittest.main()
