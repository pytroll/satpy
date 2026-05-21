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

"""Tests for sharpening compositors."""

import datetime as dt
import unittest
from unittest import mock

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from satpy.tests.utils import RANDOM_GEN


class TestRatioSharpenedCompositors:
    """Test RatioSharpenedRGB and SelfSharpendRGB compositors."""

    def setup_method(self):
        """Create test data."""
        from pyresample.geometry import AreaDefinition
        area = AreaDefinition("test", "test", "test",
                              {"proj": "merc"}, 2, 2,
                              (-2000, -2000, 2000, 2000))
        attrs = {"area": area,
                 "start_time": dt.datetime(2018, 1, 1, 18),
                 "modifiers": tuple(),
                 "resolution": 1000,
                 "calibration": "reflectance",
                 "units": "%",
                 "name": "test_vis"}
        low_res_data = np.ones((2, 2), dtype=np.float64) + 4
        low_res_data[1, 1] = 0.0  # produces infinite ratio
        ds1 = xr.DataArray(da.from_array(low_res_data, chunks=2),
                           attrs=attrs, dims=("y", "x"),
                           coords={"y": [0, 1], "x": [0, 1]})
        self.ds1 = ds1

        ds2 = xr.DataArray(da.ones((2, 2), chunks=2, dtype=np.float64) + 2,
                           attrs=attrs, dims=("y", "x"),
                           coords={"y": [0, 1], "x": [0, 1]})
        ds2.attrs["name"] += "2"
        self.ds2 = ds2

        ds3 = xr.DataArray(da.ones((2, 2), chunks=2, dtype=np.float64) + 3,
                           attrs=attrs, dims=("y", "x"),
                           coords={"y": [0, 1], "x": [0, 1]})
        ds3.attrs["name"] += "3"
        self.ds3 = ds3

        # high resolution version
        high_res_data = np.ones((2, 2), dtype=np.float64)
        high_res_data[1, 0] = np.nan  # invalid value in one band
        ds4 = xr.DataArray(da.from_array(high_res_data, chunks=2),
                           attrs=attrs, dims=("y", "x"),
                           coords={"y": [0, 1], "x": [0, 1]})
        ds4.attrs["name"] += "4"
        ds4.attrs["resolution"] = 500
        self.ds4 = ds4

        # high resolution version - but too big
        ds4_big = xr.DataArray(da.ones((4, 4), chunks=2, dtype=np.float64),
                               attrs=attrs.copy(), dims=("y", "x"),
                               coords={"y": [0, 1, 2, 3], "x": [0, 1, 2, 3]})
        ds4_big.attrs["name"] += "4"
        ds4_big.attrs["resolution"] = 500
        ds4_big.attrs["rows_per_scan"] = 1
        ds4_big.attrs["area"] = AreaDefinition("test", "test", "test",
                                               {"proj": "merc"}, 4, 4,
                                               (-2000, -2000, 2000, 2000))
        self.ds4_big = ds4_big

    @pytest.mark.parametrize(
        "init_kwargs",
        [
            {"high_resolution_band": "bad", "neutral_resolution_band": "red"},
            {"high_resolution_band": "red", "neutral_resolution_band": "bad"}
        ]
    )
    def test_bad_colors(self, init_kwargs):
        """Test that only valid band colors can be provided."""
        from satpy.composites.resolution import RatioSharpenedRGB
        with pytest.raises(ValueError, match="RatioSharpenedRGB..*_band must be one of .*"):
            RatioSharpenedRGB(name="true_color", **init_kwargs)

    def test_match_data_arrays(self):
        """Test that all areas have to be the same resolution."""
        from satpy.composites.resolution import IncompatibleAreas, RatioSharpenedRGB
        comp = RatioSharpenedRGB(name="true_color")
        with pytest.raises(IncompatibleAreas):
            comp((self.ds1, self.ds2, self.ds3), optional_datasets=(self.ds4_big,))

    def test_more_than_three_datasets(self):
        """Test that only 3 datasets can be passed."""
        from satpy.composites.resolution import RatioSharpenedRGB
        comp = RatioSharpenedRGB(name="true_color")
        with pytest.raises(ValueError, match="Expected 3 datasets, got 4"):
            comp((self.ds1, self.ds2, self.ds3, self.ds1), optional_datasets=(self.ds4_big,))

    def test_self_sharpened_no_high_res(self):
        """Test for exception when no high_res band is specified."""
        from satpy.composites.resolution import SelfSharpenedRGB
        comp = SelfSharpenedRGB(name="true_color", high_resolution_band=None)
        with pytest.raises(ValueError, match="SelfSharpenedRGB requires at least one high resolution band, not 'None'"):
            comp((self.ds1, self.ds2, self.ds3))

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_basic_no_high_res(self, dtype):
        """Test that three datasets can be passed without optional high res."""
        from satpy.composites.resolution import RatioSharpenedRGB
        comp = RatioSharpenedRGB(name="true_color")
        res = comp((self.ds1.astype(dtype), self.ds2.astype(dtype), self.ds3.astype(dtype)))
        assert res.shape == (3, 2, 2)
        assert res.dtype == dtype
        assert res.values.dtype == dtype

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_basic_no_sharpen(self, dtype):
        """Test that color None does no sharpening."""
        from satpy.composites.resolution import RatioSharpenedRGB
        comp = RatioSharpenedRGB(name="true_color", high_resolution_band=None)
        res = comp((self.ds1.astype(dtype), self.ds2.astype(dtype), self.ds3.astype(dtype)),
                   optional_datasets=(self.ds4.astype(dtype),))
        assert res.shape == (3, 2, 2)
        assert res.dtype == dtype
        assert res.values.dtype == dtype

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize(
        ("high_resolution_band", "neutral_resolution_band", "exp_r", "exp_g", "exp_b"),
        [
            ("red", None,
             np.array([[1.0, 1.0], [np.nan, 1.0]], dtype=np.float64),
             np.array([[0.6, 0.6], [np.nan, 3.0]], dtype=np.float64),
             np.array([[0.8, 0.8], [np.nan, 4.0]], dtype=np.float64)),
            ("red", "green",
             np.array([[1.0, 1.0], [np.nan, 1.0]], dtype=np.float64),
             np.array([[3.0, 3.0], [np.nan, 3.0]], dtype=np.float64),
             np.array([[0.8, 0.8], [np.nan, 4.0]], dtype=np.float64)),
            ("green", None,
             np.array([[5 / 3, 5 / 3], [np.nan, 0.0]], dtype=np.float64),
             np.array([[1.0, 1.0], [np.nan, 1.0]], dtype=np.float64),
             np.array([[4 / 3, 4 / 3], [np.nan, 4 / 3]], dtype=np.float64)),
            ("green", "blue",
             np.array([[5 / 3, 5 / 3], [np.nan, 0.0]], dtype=np.float64),
             np.array([[1.0, 1.0], [np.nan, 1.0]], dtype=np.float64),
             np.array([[4.0, 4.0], [np.nan, 4.0]], dtype=np.float64)),
            ("blue", None,
             np.array([[1.25, 1.25], [np.nan, 0.0]], dtype=np.float64),
             np.array([[0.75, 0.75], [np.nan, 0.75]], dtype=np.float64),
             np.array([[1.0, 1.0], [np.nan, 1.0]], dtype=np.float64)),
            ("blue", "red",
             np.array([[5.0, 5.0], [np.nan, 0.0]], dtype=np.float64),
             np.array([[0.75, 0.75], [np.nan, 0.75]], dtype=np.float64),
             np.array([[1.0, 1.0], [np.nan, 1.0]], dtype=np.float64))
        ]
    )
    def test_ratio_sharpening(self, high_resolution_band, neutral_resolution_band, exp_r, exp_g, exp_b, dtype):
        """Test RatioSharpenedRGB by different groups of high_resolution_band and neutral_resolution_band."""
        from satpy.composites.resolution import RatioSharpenedRGB
        comp = RatioSharpenedRGB(name="true_color", high_resolution_band=high_resolution_band,
                                 neutral_resolution_band=neutral_resolution_band)
        res = comp((self.ds1.astype(dtype), self.ds2.astype(dtype), self.ds3.astype(dtype)),
                   optional_datasets=(self.ds4.astype(dtype),))

        assert "units" not in res.attrs
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        assert res.dtype == dtype

        data = res.values
        np.testing.assert_allclose(data[0], exp_r, rtol=1e-5)
        np.testing.assert_allclose(data[1], exp_g, rtol=1e-5)
        np.testing.assert_allclose(data[2], exp_b, rtol=1e-5)
        assert res.dtype == dtype

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize(
        ("exp_shape", "exp_r", "exp_g", "exp_b"),
        [
            ((3, 2, 2),
             np.array([[5.0, 5.0], [5.0, 0]], dtype=np.float64),
             np.array([[4.0, 4.0], [4.0, 0]], dtype=np.float64),
             np.array([[16 / 3, 16 / 3], [16 / 3, 0]], dtype=np.float64))
        ]
    )
    def test_self_sharpened_basic(self, exp_shape, exp_r, exp_g, exp_b, dtype):
        """Test that three datasets can be passed without optional high res."""
        from satpy.composites.resolution import SelfSharpenedRGB
        comp = SelfSharpenedRGB(name="true_color")
        res = comp((self.ds1.astype(dtype), self.ds2.astype(dtype), self.ds3.astype(dtype)))
        assert res.dtype == dtype

        data = res.values
        assert data.shape == exp_shape
        np.testing.assert_allclose(data[0], exp_r, rtol=1e-5)
        np.testing.assert_allclose(data[1], exp_g, rtol=1e-5)
        np.testing.assert_allclose(data[2], exp_b, rtol=1e-5)
        assert data.dtype == dtype


class TestLuminanceSharpeningCompositor(unittest.TestCase):
    """Test luminance sharpening compositor."""

    def test_compositor(self):
        """Test luminance sharpening compositor."""
        from satpy.composites.resolution import LuminanceSharpeningCompositor
        comp = LuminanceSharpeningCompositor(name="test")
        # Three shades of grey
        rgb_arr = np.array([1, 50, 100, 200, 1, 50, 100, 200, 1, 50, 100, 200])
        rgb = xr.DataArray(rgb_arr.reshape((3, 2, 2)),
                           dims=["bands", "y", "x"], coords={"bands": ["R", "G", "B"]})
        # 100 % luminance -> all result values ~1.0
        lum = xr.DataArray(np.array([[100., 100.], [100., 100.]]),
                           dims=["y", "x"])
        res = comp([lum, rgb])
        np.testing.assert_allclose(res.data, 1., atol=1e-9)
        # 50 % luminance, all result values ~0.5
        lum = xr.DataArray(np.array([[50., 50.], [50., 50.]]),
                           dims=["y", "x"])
        res = comp([lum, rgb])
        np.testing.assert_allclose(res.data, 0.5, atol=1e-9)
        # 30 % luminance, all result values ~0.3
        lum = xr.DataArray(np.array([[30., 30.], [30., 30.]]),
                           dims=["y", "x"])
        res = comp([lum, rgb])
        np.testing.assert_allclose(res.data, 0.3, atol=1e-9)
        # 0 % luminance, all values ~0.0
        lum = xr.DataArray(np.array([[0., 0.], [0., 0.]]),
                           dims=["y", "x"])
        res = comp([lum, rgb])
        np.testing.assert_allclose(res.data, 0.0, atol=1e-9)


class TestSandwichCompositor:
    """Test sandwich compositor."""

    # Test RGB and RGBA
    @pytest.mark.parametrize(
        ("input_shape", "bands"),
        [
            ((3, 2, 2), ["R", "G", "B"]),
            ((4, 2, 2), ["R", "G", "B", "A"])
        ]
    )
    @mock.patch("satpy.composites.resolution.enhance2dataset")
    def test_compositor(self, e2d, input_shape, bands):
        """Test luminance sharpening compositor."""
        from satpy.composites.resolution import SandwichCompositor

        rgb_arr = da.from_array(RANDOM_GEN.random(input_shape), chunks=2)
        rgb = xr.DataArray(rgb_arr, dims=["bands", "y", "x"],
                           coords={"bands": bands})
        lum_arr = da.from_array(100 * RANDOM_GEN.random((2, 2)), chunks=2)
        lum = xr.DataArray(lum_arr, dims=["y", "x"])

        # Make enhance2dataset return unmodified dataset
        e2d.return_value = rgb
        comp = SandwichCompositor(name="test")

        res = comp([lum, rgb])

        for band in rgb:
            if band.bands != "A":
                # Check compositor has modified this band
                np.testing.assert_allclose(res.loc[band.bands].to_numpy(),
                                           band.to_numpy() * lum_arr / 100.)
            else:
                # Check Alpha band remains intact
                np.testing.assert_allclose(res.loc[band.bands].to_numpy(),
                                           band.to_numpy())
        # make sure the compositor doesn't modify the input data
        np.testing.assert_allclose(lum.values, lum_arr.compute())
