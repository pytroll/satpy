#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 Satpy developers
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
"""Unit testing the enhancements functions, e.g. cira_stretch."""

import os
import unittest
import numpy as np
import xarray as xr
import dask.array as da
from unittest import mock


class TestEnhancementStretch(unittest.TestCase):
    """Class for testing enhancements in satpy.enhancements."""

    def setUp(self):
        """Create test data used by every test."""
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
        """Perform basic checks that apply to multiple tests."""
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
        """Test applying the cira_stretch."""
        from satpy.enhancements import cira_stretch

        expected = np.array([[
            [np.nan, -7.04045974, -7.04045974, 0.79630132, 0.95947296],
            [1.05181359, 1.11651012, 1.16635571, 1.20691137, 1.24110186]]])
        self._test_enhancement(cira_stretch, self.ch1, expected)

    def test_lookup(self):
        """Test the lookup enhancement function."""
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
        """Test the colorize enhancement function."""
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
        """Test the palettize enhancement function."""
        from satpy.enhancements import palettize
        from trollimage.colormap import brbg
        expected = np.array([[[10, 0, 0, 10, 10], [10, 10, 10, 10, 10]]])
        self._test_enhancement(palettize, self.ch1, expected, palettes=brbg)

    def test_three_d_effect(self):
        """Test the three_d_effect enhancement function."""
        from satpy.enhancements import three_d_effect
        expected = np.array([[
            [np.nan, np.nan, -389.5, -294.5, 826.5],
            [np.nan, np.nan, 85.5, 180.5, 1301.5]]])
        self._test_enhancement(three_d_effect, self.ch1, expected)

    def test_crefl_scaling(self):
        """Test the crefl_scaling enhancement function."""
        from satpy.enhancements import crefl_scaling
        expected = np.array([[
            [np.nan, 0., 0., 0.44378, 0.631734],
            [0.737562, 0.825041, 0.912521, 1., 1.]]])
        self._test_enhancement(crefl_scaling, self.ch2, expected, idx=[0., 25., 55., 100., 255.],
                               sc=[0., 90., 140., 175., 255.])

    def test_btemp_threshold(self):
        """Test applying the cira_stretch."""
        from satpy.enhancements import btemp_threshold

        expected = np.array([[
            [np.nan, 0.946207, 0.892695, 0.839184, 0.785672],
            [0.73216, 0.595869, 0.158745, -0.278379, -0.715503]]])
        self._test_enhancement(btemp_threshold, self.ch1, expected,
                               min_in=-200, max_in=500, threshold=350)

    def test_merge_colormaps(self):
        """Test merging colormaps."""
        from trollimage.colormap import Colormap
        from satpy.enhancements import _merge_colormaps as mcp, create_colormap
        ret_map = mock.MagicMock()

        create_colormap_mock = mock.Mock(wraps=create_colormap)
        cmap1 = Colormap((1, (1., 1., 1.)))
        kwargs = {'palettes': cmap1}

        with mock.patch('satpy.enhancements.create_colormap', create_colormap_mock):
            res = mcp(kwargs)
        self.assertTrue(res is cmap1)
        create_colormap_mock.assert_not_called()
        create_colormap_mock.reset_mock()
        ret_map.reset_mock()

        cmap1 = {'colors': 'blues', 'min_value': 0,
                 'max_value': 1}
        kwargs = {'palettes': [cmap1]}
        with mock.patch('satpy.enhancements.create_colormap', create_colormap_mock),\
                mock.patch('trollimage.colormap.blues', ret_map):
            _ = mcp(kwargs)
        create_colormap_mock.assert_called_once()
        ret_map.reverse.assert_not_called()
        ret_map.set_range.assert_called_with(0, 1)
        create_colormap_mock.reset_mock()
        ret_map.reset_mock()

        cmap2 = {'colors': 'blues', 'min_value': 2,
                 'max_value': 3, 'reverse': True}
        kwargs = {'palettes': [cmap2]}
        with mock.patch('trollimage.colormap.blues', ret_map):
            _ = mcp(kwargs)
        ret_map.reverse.assert_called_once()
        ret_map.set_range.assert_called_with(2, 3)
        create_colormap_mock.reset_mock()
        ret_map.reset_mock()

        kwargs = {'palettes': [cmap1, cmap2]}
        with mock.patch('trollimage.colormap.blues', ret_map):
            _ = mcp(kwargs)
        ret_map.__add__.assert_called_once()

    def tearDown(self):
        """Clean up."""
        pass


class TestColormapLoading(unittest.TestCase):
    """Test utilities used with colormaps."""

    def test_cmap_from_file_rgb(self):
        """Test that colormaps can be loaded from a binary file."""
        from satpy.enhancements import create_colormap
        from tempfile import NamedTemporaryFile
        # create the colormap file on disk
        with NamedTemporaryFile(suffix='.npy', delete=False) as tmp_cmap:
            cmap_filename = tmp_cmap.name
            np.save(cmap_filename, np.array([
                [255, 0, 0],
                [255, 255, 0],
                [255, 255, 255],
                [0, 0, 255],
            ]))

        try:
            cmap = create_colormap({'filename': cmap_filename})
            self.assertEqual(cmap.colors.shape[0], 4)
            np.testing.assert_equal(cmap.colors[0], [1.0, 0, 0])
            self.assertEqual(cmap.values.shape[0], 4)
            self.assertEqual(cmap.values[0], 0)
            self.assertEqual(cmap.values[-1], 1.0)

            cmap = create_colormap({'filename': cmap_filename, 'min_value': 50, 'max_value': 100})
            self.assertEqual(cmap.colors.shape[0], 4)
            np.testing.assert_equal(cmap.colors[0], [1.0, 0, 0])
            self.assertEqual(cmap.values.shape[0], 4)
            self.assertEqual(cmap.values[0], 50)
            self.assertEqual(cmap.values[-1], 100)
        finally:
            os.remove(cmap_filename)

    def test_cmap_from_file_rgb_1(self):
        """Test that colormaps can be loaded from a binary file with 0-1 colors."""
        from satpy.enhancements import create_colormap
        from tempfile import NamedTemporaryFile
        # create the colormap file on disk
        with NamedTemporaryFile(suffix='.npy', delete=False) as tmp_cmap:
            cmap_filename = tmp_cmap.name
            np.save(cmap_filename, np.array([
                [1, 0, 0],
                [1, 1, 0],
                [1, 1, 1],
                [0, 0, 1],
            ]))

        try:
            cmap = create_colormap({'filename': cmap_filename,
                                    'color_scale': 1})
            self.assertEqual(cmap.colors.shape[0], 4)
            np.testing.assert_equal(cmap.colors[0], [1.0, 0, 0])
            self.assertEqual(cmap.values.shape[0], 4)
            self.assertEqual(cmap.values[0], 0)
            self.assertEqual(cmap.values[-1], 1.0)

            cmap = create_colormap({'filename': cmap_filename, 'color_scale': 1,
                                    'min_value': 50, 'max_value': 100})
            self.assertEqual(cmap.colors.shape[0], 4)
            np.testing.assert_equal(cmap.colors[0], [1.0, 0, 0])
            self.assertEqual(cmap.values.shape[0], 4)
            self.assertEqual(cmap.values[0], 50)
            self.assertEqual(cmap.values[-1], 100)
        finally:
            os.remove(cmap_filename)

    def test_cmap_from_file_vrgb(self):
        """Test that colormaps can be loaded from a binary file with values."""
        from satpy.enhancements import create_colormap
        from tempfile import NamedTemporaryFile
        # create the colormap file on disk
        with NamedTemporaryFile(suffix='.npy', delete=False) as tmp_cmap:
            cmap_filename = tmp_cmap.name
            np.save(cmap_filename, np.array([
                [128, 255, 0, 0],
                [130, 255, 255, 0],
                [132, 255, 255, 255],
                [134, 0, 0, 255],
            ]))

        try:
            # default mode of VRGB
            cmap = create_colormap({'filename': cmap_filename})
            self.assertEqual(cmap.colors.shape[0], 4)
            np.testing.assert_equal(cmap.colors[0], [1.0, 0, 0])
            self.assertEqual(cmap.values.shape[0], 4)
            self.assertEqual(cmap.values[0], 128)
            self.assertEqual(cmap.values[-1], 134)

            cmap = create_colormap({'filename': cmap_filename, 'colormap_mode': 'RGBA'})
            self.assertEqual(cmap.colors.shape[0], 4)
            self.assertEqual(cmap.colors.shape[1], 4)  # RGBA
            np.testing.assert_equal(cmap.colors[0], [128 / 255., 1.0, 0, 0])
            self.assertEqual(cmap.values.shape[0], 4)
            self.assertEqual(cmap.values[0], 0)
            self.assertEqual(cmap.values[-1], 1.0)

            cmap = create_colormap({'filename': cmap_filename, 'min_value': 50, 'max_value': 100})
            self.assertEqual(cmap.colors.shape[0], 4)
            np.testing.assert_equal(cmap.colors[0], [1.0, 0, 0])
            self.assertEqual(cmap.values.shape[0], 4)
            self.assertEqual(cmap.values[0], 50)
            self.assertEqual(cmap.values[-1], 100)

            self.assertRaises(ValueError, create_colormap,
                              {'filename': cmap_filename, 'colormap_mode': 'RGB',
                               'min_value': 50, 'max_value': 100})
        finally:
            os.remove(cmap_filename)

    def test_cmap_from_file_vrgba(self):
        """Test that colormaps can be loaded RGBA colors and values."""
        from satpy.enhancements import create_colormap
        from tempfile import NamedTemporaryFile
        # create the colormap file on disk
        with NamedTemporaryFile(suffix='.npy', delete=False) as tmp_cmap:
            cmap_filename = tmp_cmap.name
            np.save(cmap_filename, np.array([
                [128, 128, 255, 0, 0],  # value, R, G, B, A
                [130, 130, 255, 255, 0],
                [132, 132, 255, 255, 255],
                [134, 134, 0, 0, 255],
            ]))

        try:
            # default mode of VRGBA
            cmap = create_colormap({'filename': cmap_filename})
            self.assertEqual(cmap.colors.shape[0], 4)
            self.assertEqual(cmap.colors.shape[1], 4)  # RGBA
            np.testing.assert_equal(cmap.colors[0], [128 / 255.0, 1.0, 0, 0])
            self.assertEqual(cmap.values.shape[0], 4)
            self.assertEqual(cmap.values[0], 128)
            self.assertEqual(cmap.values[-1], 134)

            self.assertRaises(ValueError, create_colormap,
                              {'filename': cmap_filename, 'colormap_mode': 'RGBA'})

            cmap = create_colormap({'filename': cmap_filename, 'min_value': 50, 'max_value': 100})
            self.assertEqual(cmap.colors.shape[0], 4)
            self.assertEqual(cmap.colors.shape[1], 4)  # RGBA
            np.testing.assert_equal(cmap.colors[0], [128 / 255.0, 1.0, 0, 0])
            self.assertEqual(cmap.values.shape[0], 4)
            self.assertEqual(cmap.values[0], 50)
            self.assertEqual(cmap.values[-1], 100)
        finally:
            os.remove(cmap_filename)

    def test_cmap_from_file_bad_shape(self):
        """Test that unknown array shape causes an error."""
        from satpy.enhancements import create_colormap
        from tempfile import NamedTemporaryFile
        # create the colormap file on disk
        with NamedTemporaryFile(suffix='.npy', delete=False) as tmp_cmap:
            cmap_filename = tmp_cmap.name
            np.save(cmap_filename, np.array([
                [0],
                [64],
                [128],
                [255],
            ]))

        try:
            self.assertRaises(ValueError, create_colormap,
                              {'filename': cmap_filename})
        finally:
            os.remove(cmap_filename)

    def test_cmap_from_trollimage(self):
        """Test that colormaps in trollimage can be loaded."""
        from satpy.enhancements import create_colormap
        cmap = create_colormap({'colors': 'pubu'})
        from trollimage.colormap import pubu
        np.testing.assert_equal(cmap.colors, pubu.colors)
        np.testing.assert_equal(cmap.values, pubu.values)

    def test_cmap_no_colormap(self):
        """Test that being unable to create a colormap raises an error."""
        from satpy.enhancements import create_colormap
        self.assertRaises(ValueError, create_colormap, {})

    def test_cmap_list(self):
        """Test that colors can be a list/tuple."""
        from satpy.enhancements import create_colormap
        colors = [
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]
        values = [2, 4, 6, 8]
        cmap = create_colormap({'colors': colors, 'color_scale': 1})
        self.assertEqual(cmap.colors.shape[0], 4)
        np.testing.assert_equal(cmap.colors[0], [0.0, 0.0, 1.0])
        self.assertEqual(cmap.values.shape[0], 4)
        self.assertEqual(cmap.values[0], 0)
        self.assertEqual(cmap.values[-1], 1.0)

        cmap = create_colormap({'colors': colors, 'color_scale': 1, 'values': values})
        self.assertEqual(cmap.colors.shape[0], 4)
        np.testing.assert_equal(cmap.colors[0], [0.0, 0.0, 1.0])
        self.assertEqual(cmap.values.shape[0], 4)
        self.assertEqual(cmap.values[0], 2)
        self.assertEqual(cmap.values[-1], 8)
