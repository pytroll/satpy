#!/usr/bin/env python
# Copyright (c) 2018-2025 Satpy developers
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

"""Tests for compositors using lookup tables."""

import unittest

import dask.array as da
import numpy as np
import xarray as xr


class TestColormapCompositor(unittest.TestCase):
    """Test the ColormapCompositor."""

    def setUp(self):
        """Set up the test case."""
        from satpy.composites.lookup import ColormapCompositor
        self.colormap_compositor = ColormapCompositor("test_cmap_compositor")

    def test_build_colormap_with_int_data_and_without_meanings(self):
        """Test colormap building."""
        palette = np.array([[0, 0, 0], [127, 127, 127], [255, 255, 255]])
        colormap, squeezed_palette = self.colormap_compositor.build_colormap(palette, np.uint8, {})
        assert np.allclose(colormap.values, [0, 1])
        assert np.allclose(squeezed_palette, palette / 255.0)

    def test_build_colormap_with_int_data_and_with_meanings(self):
        """Test colormap building."""
        palette = xr.DataArray(np.array([[0, 0, 0], [127, 127, 127], [255, 255, 255]]),
                               dims=["value", "band"])
        palette.attrs["palette_meanings"] = [2, 3, 4]
        colormap, squeezed_palette = self.colormap_compositor.build_colormap(palette, np.uint8, {})
        assert np.allclose(colormap.values, [2, 3, 4])
        assert np.allclose(squeezed_palette, palette / 255.0)


class TestPaletteCompositor(unittest.TestCase):
    """Test the PaletteCompositor."""

    def test_call(self):
        """Test palette compositing."""
        from satpy.composites.lookup import PaletteCompositor
        cmap_comp = PaletteCompositor("test_cmap_compositor")
        palette = xr.DataArray(np.array([[0, 0, 0], [127, 127, 127], [255, 255, 255]]),
                               dims=["value", "band"])
        palette.attrs["palette_meanings"] = [2, 3, 4]

        data = xr.DataArray(da.from_array(np.array([[4, 3, 2], [2, 3, 4]], dtype=np.uint8)), dims=["y", "x"])
        res = cmap_comp([data, palette])
        exp = np.array([[[1., 0.498039, 0.],
                         [0., 0.498039, 1.]],
                        [[1., 0.498039, 0.],
                         [0., 0.498039, 1.]],
                        [[1., 0.498039, 0.],
                         [0., 0.498039, 1.]]])
        assert np.allclose(res, exp)


class TestColorizeCompositor(unittest.TestCase):
    """Test the ColorizeCompositor."""

    def test_colorize_no_fill(self):
        """Test colorizing."""
        from satpy.composites.lookup import ColorizeCompositor
        colormap_composite = ColorizeCompositor("test_color_compositor")
        palette = xr.DataArray(np.array([[0, 0, 0], [127, 127, 127], [255, 255, 255]]),
                               dims=["value", "band"])
        palette.attrs["palette_meanings"] = [2, 3, 4]

        data = xr.DataArray(np.array([[4, 3, 2],
                                      [2, 3, 4]],
                                     dtype=np.uint8),
                            dims=["y", "x"])
        res = colormap_composite([data, palette])
        exp = np.array([[[1., 0.498039, 0.],
                         [0., 0.498039, 1.]],
                        [[1., 0.498039, 0.],
                         [0., 0.498039, 1.]],
                        [[1., 0.498039, 0.],
                         [0., 0.498039, 1.]]])
        assert np.allclose(res, exp, atol=0.0001)

    def test_colorize_with_interpolation(self):
        """Test colorizing with interpolation."""
        from satpy.composites.lookup import ColorizeCompositor
        colormap_composite = ColorizeCompositor("test_color_compositor")
        palette = xr.DataArray(np.array([[0, 0, 0], [127, 127, 127], [255, 255, 255]]),
                               dims=["value", "band"])
        palette.attrs["palette_meanings"] = [2, 3, 4]

        data = xr.DataArray(da.from_array(np.array([[4, 3, 2.5],
                                                    [2, 3.2, 4]])),
                            dims=["y", "x"],
                            attrs={"valid_range": np.array([2, 4])})
        res = colormap_composite([data, palette])
        exp = np.array([[[1.0, 0.498039, 0.246575],
                         [0., 0.59309977, 1.0]],
                        [[1.0, 0.49803924, 0.24657543],
                         [0., 0.59309983, 1.0]],
                        [[1.0, 0.4980392, 0.24657541],
                         [0., 0.59309978, 1.0]]])
        np.testing.assert_allclose(res, exp, atol=1e-4)


class TestCategoricalDataCompositor(unittest.TestCase):
    """Test composiotor for recategorization of categorical data."""

    def setUp(self):
        """Create test data."""
        attrs = {"name": "foo"}
        data = xr.DataArray(da.from_array([[2., 1.], [3., 0.]]), attrs=attrs,
                            dims=("y", "x"), coords={"y": [0, 1], "x": [0, 1]})

        self.data = data

    def test_basic_recategorization(self):
        """Test general functionality of compositor incl. attributes."""
        from satpy.composites.lookup import CategoricalDataCompositor
        lut = [np.nan, 0, 1, 1]
        name = "bar"
        comp = CategoricalDataCompositor(name=name, lut=lut)
        res = comp([self.data])
        res = res.compute()
        expected = np.array([[1., 0.], [1., np.nan]])
        np.testing.assert_equal(res.values, expected)
        np.testing.assert_equal(res.attrs["name"], name)
        np.testing.assert_equal(res.attrs["composite_lut"], lut)

    def test_too_many_datasets(self):
        """Test that ValueError is raised if more than one dataset is provided."""
        from satpy.composites.lookup import CategoricalDataCompositor
        lut = [np.nan, 0, 1, 1]
        comp = CategoricalDataCompositor(name="foo", lut=lut)
        np.testing.assert_raises(ValueError, comp, [self.data, self.data])
