#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2023 Satpy developers
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
"""CF processing of time information (coordinates and dimensions)."""
import logging

import numpy as np
import pytest
import xarray as xr
from pyresample import AreaDefinition

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - caplog


class TestCFtime:
    """Test cases for CF time dimension and coordinates."""

    def test_add_time_bounds_dimension(self):
        """Test addition of CF-compliant time attributes."""
        from satpy.cf.coords import add_time_bounds_dimension

        test_array = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        times = np.array(["2018-05-30T10:05:00", "2018-05-30T10:05:01",
                          "2018-05-30T10:05:02", "2018-05-30T10:05:03"], dtype=np.datetime64)
        dataarray = xr.DataArray(test_array,
                                 dims=["y", "x"],
                                 coords={"time": ("y", times)},
                                 attrs=dict(start_time=times[0], end_time=times[-1]))
        ds = dataarray.to_dataset(name="test-array")
        ds = add_time_bounds_dimension(ds)

        assert "bnds_1d" in ds.dims
        assert ds.dims["bnds_1d"] == 2
        assert "time_bnds" in list(ds.data_vars)
        assert "bounds" in ds["time"].attrs
        assert "standard_name" in ds["time"].attrs

    # set_cf_time_info


class TestCFcoords:
    """Test cases for CF spatial dimension and coordinates."""

    def test_check_unique_projection_coords(self):
        """Test that the x and y coordinates are unique."""
        from satpy.cf.coords import check_unique_projection_coords

        dummy = [[1, 2], [3, 4]]
        datas = {"a": xr.DataArray(data=dummy, dims=("y", "x"), coords={"y": [1, 2], "x": [3, 4]}),
                 "b": xr.DataArray(data=dummy, dims=("y", "x"), coords={"y": [1, 2], "x": [3, 4]}),
                 "n": xr.DataArray(data=dummy, dims=("v", "w"), coords={"v": [1, 2], "w": [3, 4]})}
        check_unique_projection_coords(datas)

        datas["c"] = xr.DataArray(data=dummy, dims=("y", "x"), coords={"y": [1, 3], "x": [3, 4]})
        with pytest.raises(ValueError, match="must have identical projection coordinates"):
            check_unique_projection_coords(datas)

    def test_add_coordinates_attrs_coords(self):
        """Check that coordinates link has been established correctly."""
        from satpy.cf.coords import add_coordinates_attrs_coords

        data = [[1, 2], [3, 4]]
        lon = np.zeros((2, 2))
        lon2 = np.zeros((1, 2, 2))
        lat = np.ones((2, 2))
        datasets = {
            "var1": xr.DataArray(data=data, dims=("y", "x"), attrs={"coordinates": "lon lat"}),
            "var2": xr.DataArray(data=data, dims=("y", "x")),
            "var3": xr.DataArray(data=data, dims=("y", "x"), attrs={"coordinates": "lon2 lat"}),
            "var4": xr.DataArray(data=data, dims=("y", "x"), attrs={"coordinates": "not_exist lon lat"}),
            "lon": xr.DataArray(data=lon, dims=("y", "x")),
            "lon2": xr.DataArray(data=lon2, dims=("time", "y", "x")),
            "lat": xr.DataArray(data=lat, dims=("y", "x"))
        }

        datasets = add_coordinates_attrs_coords(datasets)

        # Check that link has been established correctly and 'coordinate' atrribute has been dropped
        assert "lon" in datasets["var1"].coords
        assert "lat" in datasets["var1"].coords
        np.testing.assert_array_equal(datasets["var1"]["lon"].data, lon)
        np.testing.assert_array_equal(datasets["var1"]["lat"].data, lat)
        assert "coordinates" not in datasets["var1"].attrs

        # There should be no link if there was no 'coordinate' attribute
        assert "lon" not in datasets["var2"].coords
        assert "lat" not in datasets["var2"].coords

        # The non-existent dimension or coordinate should be dropped
        assert "time" not in datasets["var3"].coords
        assert "not_exist" not in datasets["var4"].coords

    def test_ensure_unique_nondimensional_coords(self):
        """Test that created coordinate variables are unique."""
        from satpy.cf.coords import ensure_unique_nondimensional_coords

        data = [[1, 2], [3, 4]]
        y = [1, 2]
        x = [1, 2]
        time1 = [1, 2]
        time2 = [3, 4]
        datasets = {"var1": xr.DataArray(data=data,
                                         dims=("y", "x"),
                                         coords={"y": y, "x": x, "acq_time": ("y", time1)}),
                    "var2": xr.DataArray(data=data,
                                         dims=("y", "x"),
                                         coords={"y": y, "x": x, "acq_time": ("y", time2)})}

        # Test that dataset names are prepended to alternative coordinates
        res = ensure_unique_nondimensional_coords(datasets)
        np.testing.assert_array_equal(res["var1"]["var1_acq_time"], time1)
        np.testing.assert_array_equal(res["var2"]["var2_acq_time"], time2)
        assert "acq_time" not in res["var1"].coords
        assert "acq_time" not in res["var2"].coords

        # Make sure nothing else is modified
        np.testing.assert_array_equal(res["var1"]["x"], x)
        np.testing.assert_array_equal(res["var1"]["y"], y)
        np.testing.assert_array_equal(res["var2"]["x"], x)
        np.testing.assert_array_equal(res["var2"]["y"], y)

        # Coords not unique -> Dataset names must be prepended, even if pretty=True
        with pytest.warns(UserWarning, match='Cannot pretty-format "acq_time"'):
            res = ensure_unique_nondimensional_coords(datasets, pretty=True)
        np.testing.assert_array_equal(res["var1"]["var1_acq_time"], time1)
        np.testing.assert_array_equal(res["var2"]["var2_acq_time"], time2)
        assert "acq_time" not in res["var1"].coords
        assert "acq_time" not in res["var2"].coords

        # Coords unique and pretty=True -> Don't modify coordinate names
        datasets["var2"]["acq_time"] = ("y", time1)
        res = ensure_unique_nondimensional_coords(datasets, pretty=True)
        np.testing.assert_array_equal(res["var1"]["acq_time"], time1)
        np.testing.assert_array_equal(res["var2"]["acq_time"], time1)
        assert "var1_acq_time" not in res["var1"].coords
        assert "var2_acq_time" not in res["var2"].coords

    def test_is_projected(self, caplog):
        """Tests for private _is_projected function."""
        from satpy.cf.coords import _is_projected

        # test case with units but no area
        da = xr.DataArray(
            np.arange(25).reshape(5, 5),
            dims=("y", "x"),
            coords={"x": xr.DataArray(np.arange(5), dims=("x",), attrs={"units": "m"}),
                    "y": xr.DataArray(np.arange(5), dims=("y",), attrs={"units": "m"})})
        assert _is_projected(da)

        da = xr.DataArray(
            np.arange(25).reshape(5, 5),
            dims=("y", "x"),
            coords={"x": xr.DataArray(np.arange(5), dims=("x",), attrs={"units": "degrees_east"}),
                    "y": xr.DataArray(np.arange(5), dims=("y",), attrs={"units": "degrees_north"})})
        assert not _is_projected(da)

        da = xr.DataArray(
            np.arange(25).reshape(5, 5),
            dims=("y", "x"))
        with caplog.at_level(logging.WARNING):
            assert _is_projected(da)
        assert "Failed to tell if data are projected." in caplog.text

    @pytest.fixture
    def datasets(self):
        """Create test dataset."""
        data = [[75, 2], [3, 4]]
        y = [1, 2]
        x = [1, 2]
        geos = AreaDefinition(
            area_id="geos",
            description="geos",
            proj_id="geos",
            projection={"proj": "geos", "h": 35785831., "a": 6378169., "b": 6356583.8},
            width=2, height=2,
            area_extent=[-1, -1, 1, 1])
        datasets = {
            "var1": xr.DataArray(data=data,
                                 dims=("y", "x"),
                                 coords={"y": y, "x": x}),
            "var2": xr.DataArray(data=data,
                                 dims=("y", "x"),
                                 coords={"y": y, "x": x}),
            "lat": xr.DataArray(data=data,
                                dims=("y", "x"),
                                coords={"y": y, "x": x}),
            "lon": xr.DataArray(data=data,
                                dims=("y", "x"),
                                coords={"y": y, "x": x})}
        datasets["lat"].attrs["standard_name"] = "latitude"
        datasets["var1"].attrs["standard_name"] = "dummy"
        datasets["var2"].attrs["standard_name"] = "dummy"
        datasets["var2"].attrs["area"] = geos
        datasets["var1"].attrs["area"] = geos
        datasets["lat"].attrs["name"] = "lat"
        datasets["var1"].attrs["name"] = "var1"
        datasets["var2"].attrs["name"] = "var2"
        datasets["lon"].attrs["name"] = "lon"
        return datasets

    def test__is_lon_or_lat_dataarray(self, datasets):
        """Test the _is_lon_or_lat_dataarray function."""
        from satpy.cf.coords import _is_lon_or_lat_dataarray

        assert _is_lon_or_lat_dataarray(datasets["lat"])
        assert not _is_lon_or_lat_dataarray(datasets["var1"])

    def test_has_projection_coords(self, datasets):
        """Test the has_projection_coords function."""
        from satpy.cf.coords import has_projection_coords

        assert has_projection_coords(datasets)
        datasets["lat"].attrs["standard_name"] = "dummy"
        assert not has_projection_coords(datasets)
