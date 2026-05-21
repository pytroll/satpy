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

"""Tests for cloud product compositors."""

import unittest

import dask.array as da
import numpy as np
import xarray as xr


class TestCloudCompositorWithoutCloudfree:
    """Test the CloudCompositorWithoutCloudfree."""

    def setup_method(self):
        """Set up the test case."""
        from satpy.composites.cloud_products import CloudCompositorWithoutCloudfree
        self.colormap_composite = CloudCompositorWithoutCloudfree("test_cmap_compositor")

        self.exp = np.array([[4, 3, 2], [2, 3, np.nan], [8, 7, 655350]])
        self.exp_bad_oc = np.array([[4, 3, 2],
                                    [2, np.nan, 4],
                                    [np.nan, 7, 255]])

    def test_call_numpy_with_invalid_value_in_status(self):
        """Test the CloudCompositorWithoutCloudfree composite generation."""
        status = xr.DataArray(np.array([[0, 0, 0], [0, 0, 65535], [0, 0, 1]]), dims=["y", "x"],
                              attrs={"_FillValue": 65535})
        data = xr.DataArray(np.array([[4, 3, 2], [2, 3, np.nan], [8, 7, np.nan]], dtype=np.float32),
                            dims=["y", "x"],
                            attrs={"_FillValue": 65535,
                                   "scaled_FillValue": 655350})
        res = self.colormap_composite([data, status])
        np.testing.assert_allclose(res, self.exp, atol=1e-4)

    def test_call_dask_with_invalid_value_in_status(self):
        """Test the CloudCompositorWithoutCloudfree composite generation."""
        status = xr.DataArray(da.from_array(np.array([[0, 0, 0], [0, 0, 65535], [0, 0, 1]])), dims=["y", "x"],
                              attrs={"_FillValue": 65535})
        data = xr.DataArray(da.from_array(np.array([[4, 3, 2], [2, 3, np.nan], [8, 7, np.nan]], dtype=np.float32)),
                            dims=["y", "x"],
                            attrs={"_FillValue": 99,
                                   "scaled_FillValue": 655350})
        res = self.colormap_composite([data, status])
        np.testing.assert_allclose(res, self.exp, atol=1e-4)

    def test_call_bad_optical_conditions(self):
        """Test the CloudCompositorWithoutCloudfree composite generation."""
        status = xr.DataArray(da.from_array(np.array([[0, 0, 0], [3, 3, 3], [0, 0, 1]])), dims=["y", "x"],
                              attrs={"_FillValue": 65535,
                                     "flag_meanings": "bad_optical_conditions"})
        data = xr.DataArray(np.array([[4, 3, 2], [2, 255, 4], [255, 7, 255]], dtype=np.uint8),
                            dims=["y", "x"],
                            name="cmic_cre",
                            attrs={"_FillValue": 255,
                                   "scaled_FillValue": 255})
        res = self.colormap_composite([data, status])
        np.testing.assert_allclose(res, self.exp_bad_oc, atol=1e-4)

    def test_bad_indata(self):
        """Test the CloudCompositorWithoutCloudfree composite generation without status."""
        data = xr.DataArray(np.array([[4, 3, 2], [2, 3, 4], [255, 7, 255]], dtype=np.uint8),
                            dims=["y", "x"],
                            attrs={"_FillValue": 255,
                                   "scaled_FillValue": 255})
        np.testing.assert_raises(ValueError, self.colormap_composite, [data])


class TestCloudCompositorCommonMask:
    """Test the CloudCompositorCommonMask."""

    def setup_method(self):
        """Set up the test case."""
        from satpy.composites.cloud_products import CloudCompositorCommonMask

        self.exp_a = np.array([[4, 3, 2],
                               [2, 3, 655350],
                               [np.nan, np.nan, np.nan]])
        self.exp_b = np.array([[4, 3, 2],
                               [2, 3, 255],
                               [np.nan, np.nan, np.nan]])
        self.colormap_composite = CloudCompositorCommonMask("test_cmap_compositor")

    def test_call_numpy(self):
        """Test the CloudCompositorCommonMask with numpy."""
        mask = xr.DataArray(np.array([[0, 0, 0], [1, 1, 1], [255, 255, 255]]), dims=["y", "x"],
                            attrs={"_FillValue": 255})
        data = xr.DataArray(np.array([[4, 3, 2], [2, 3, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32),
                            dims=["y", "x"],
                            attrs={"_FillValue": 65535,
                                   "scaled_FillValue": 655350})
        res = self.colormap_composite([data, mask])
        np.testing.assert_allclose(res, self.exp_a, atol=1e-4)

    def test_call_dask(self):
        """Test the CloudCompositorCommonMask with dask."""
        mask = xr.DataArray(da.from_array(np.array([[0, 0, 0], [1, 1, 1], [255, 255, 255]])), dims=["y", "x"],
                            attrs={"_FillValue": 255})
        data = xr.DataArray(da.from_array(np.array([[4, 3, 2], [2, 3, 255], [255, 255, 255]], dtype=np.int16)),
                            dims=["y", "x"],
                            attrs={"_FillValue": 255,
                                   "scaled_FillValue": 255})
        res = self.colormap_composite([data, mask])
        np.testing.assert_allclose(res, self.exp_b, atol=1e-4)

    def test_bad_call(self):
        """Test the CloudCompositorCommonMask without mask."""
        data = xr.DataArray(np.array([[4, 3, 2], [2, 3, 255], [255, 255, 255]], dtype=np.int16),
                            dims=["y", "x"],
                            attrs={"_FillValue": 255,
                                   "scaled_FillValue": 255})
        np.testing.assert_raises(ValueError, self.colormap_composite, [data])


class TestPrecipCloudsCompositor(unittest.TestCase):
    """Test the PrecipClouds compositor."""

    def test_call(self):
        """Test the precip composite generation."""
        from satpy.composites.cloud_products import PrecipCloudsRGB
        colormap_compositor = PrecipCloudsRGB("test_precip_compositor")

        data_light = xr.DataArray(np.array([[80, 70, 60, 0], [20, 30, 40, 255]], dtype=np.uint8),
                                  dims=["y", "x"], attrs={"_FillValue": 255})
        data_moderate = xr.DataArray(np.array([[60, 50, 40, 0], [20, 30, 40, 255]], dtype=np.uint8),
                                     dims=["y", "x"], attrs={"_FillValue": 255})
        data_intense = xr.DataArray(np.array([[40, 30, 20, 0], [20, 30, 40, 255]], dtype=np.uint8),
                                    dims=["y", "x"], attrs={"_FillValue": 255})
        data_flags = xr.DataArray(np.array([[0, 0, 4, 0], [0, 0, 0, 0]], dtype=np.uint8),
                                  dims=["y", "x"])
        res = colormap_compositor([data_light, data_moderate, data_intense, data_flags])

        exp = np.array([[[0.24313725, 0.18235294, 0.12156863, np.nan],
                         [0.12156863, 0.18235294, 0.24313725, np.nan]],
                        [[0.62184874, 0.51820728, 0.41456583, np.nan],
                         [0.20728291, 0.31092437, 0.41456583, np.nan]],
                        [[0.82913165, 0.7254902, 0.62184874, np.nan],
                         [0.20728291, 0.31092437, 0.41456583, np.nan]]])

        np.testing.assert_allclose(res, exp)
