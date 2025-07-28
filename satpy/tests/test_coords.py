#!/usr/bin/python
# Copyright (c) 2016 Satpy developers
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
"""Unittests for coordinate utilities."""

import unittest

import dask.array as da
import numpy as np
import xarray as xr
from pyproj import CRS


class TestCoordinateHelpers(unittest.TestCase):
    """Test various utility functions for working with coordinates."""

    def test_area_def_coordinates(self):
        """Test coordinates being added with an AreaDefinition."""
        from pyresample.geometry import AreaDefinition

        from satpy.coords import add_crs_xy_coords
        area_def = AreaDefinition(
            "test", "test", "test", {"proj": "lcc", "lat_1": 25, "lat_0": 25},
            100, 200, [-100, -100, 100, 100]
        )
        data_arr = xr.DataArray(
            da.zeros((200, 100), chunks=50),
            attrs={"area": area_def},
            dims=("y", "x"),
        )
        new_data_arr = add_crs_xy_coords(data_arr, area_def)
        assert "y" in new_data_arr.coords
        assert "x" in new_data_arr.coords

        assert "units" in new_data_arr.coords["y"].attrs
        assert new_data_arr.coords["y"].attrs["units"] == "meter"
        assert "units" in new_data_arr.coords["x"].attrs
        assert new_data_arr.coords["x"].attrs["units"] == "meter"
        assert "crs" in new_data_arr.coords
        assert isinstance(new_data_arr.coords["crs"].item(), CRS)
        assert area_def.crs == new_data_arr.coords["crs"].item()

        # already has coords
        data_arr = xr.DataArray(
            da.zeros((200, 100), chunks=50),
            attrs={"area": area_def},
            dims=("y", "x"),
            coords={"y": np.arange(2, 202), "x": np.arange(100)}
        )
        new_data_arr = add_crs_xy_coords(data_arr, area_def)
        assert "y" in new_data_arr.coords
        assert "units" not in new_data_arr.coords["y"].attrs
        assert "x" in new_data_arr.coords
        assert "units" not in new_data_arr.coords["x"].attrs
        np.testing.assert_equal(new_data_arr.coords["y"], np.arange(2, 202))

        assert "crs" in new_data_arr.coords
        assert isinstance(new_data_arr.coords["crs"].item(), CRS)
        assert area_def.crs == new_data_arr.coords["crs"].item()

        # lat/lon area
        area_def = AreaDefinition(
            "test", "test", "test", {"proj": "latlong"},
            100, 200, [-100, -100, 100, 100]
        )
        data_arr = xr.DataArray(
            da.zeros((200, 100), chunks=50),
            attrs={"area": area_def},
            dims=("y", "x"),
        )
        new_data_arr = add_crs_xy_coords(data_arr, area_def)
        assert "y" in new_data_arr.coords
        assert "x" in new_data_arr.coords

        assert "units" in new_data_arr.coords["y"].attrs
        assert new_data_arr.coords["y"].attrs["units"] == "degrees_north"
        assert "units" in new_data_arr.coords["x"].attrs
        assert new_data_arr.coords["x"].attrs["units"] == "degrees_east"
        assert "crs" in new_data_arr.coords
        assert isinstance(new_data_arr.coords["crs"].item(), CRS)
        assert area_def.crs == new_data_arr.coords["crs"].item()

    def test_swath_def_coordinates(self):
        """Test coordinates being added with an SwathDefinition."""
        from pyresample.geometry import SwathDefinition

        from satpy.coords import add_crs_xy_coords
        lons_data = da.random.random((200, 100), chunks=50)
        lats_data = da.random.random((200, 100), chunks=50)
        lons = xr.DataArray(lons_data, attrs={"units": "degrees_east"},
                            dims=("y", "x"))
        lats = xr.DataArray(lats_data, attrs={"units": "degrees_north"},
                            dims=("y", "x"))
        area_def = SwathDefinition(lons, lats)
        data_arr = xr.DataArray(
            da.zeros((200, 100), chunks=50),
            attrs={"area": area_def},
            dims=("y", "x"),
        )
        new_data_arr = add_crs_xy_coords(data_arr, area_def)
        # See https://github.com/pydata/xarray/issues/3068
        # self.assertIn('longitude', new_data_arr.coords)
        # self.assertIn('units', new_data_arr.coords['longitude'].attrs)
        # self.assertEqual(
        #     new_data_arr.coords['longitude'].attrs['units'], 'degrees_east')
        # self.assertIsInstance(new_data_arr.coords['longitude'].data, da.Array)
        # self.assertIn('latitude', new_data_arr.coords)
        # self.assertIn('units', new_data_arr.coords['latitude'].attrs)
        # self.assertEqual(
        #     new_data_arr.coords['latitude'].attrs['units'], 'degrees_north')
        # self.assertIsInstance(new_data_arr.coords['latitude'].data, da.Array)

        assert "crs" in new_data_arr.coords
        crs = new_data_arr.coords["crs"].item()
        assert isinstance(crs, CRS)
        assert crs.is_geographic
        assert isinstance(new_data_arr.coords["crs"].item(), CRS)
