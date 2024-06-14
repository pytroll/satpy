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
"""Tests CF-compliant DataArray creation."""
import numpy as np
import xarray as xr

from satpy.tests.utils import make_dsq


def test_preprocess_dataarray_name():
    """Test saving an array to netcdf/cf where dataset name starting with a digit with prefix include orig name."""
    from satpy import Scene
    from satpy.cf.data_array import _preprocess_data_array_name

    scn = Scene()
    scn["1"] = xr.DataArray([1, 2, 3])
    dataarray = scn["1"]
    # If numeric_name_prefix is a string, test add the original_name attributes
    out_da = _preprocess_data_array_name(dataarray, numeric_name_prefix="TEST", include_orig_name=True)
    assert out_da.attrs["original_name"] == "1"

    # If numeric_name_prefix is empty string, False or None, test do not add original_name attributes
    out_da = _preprocess_data_array_name(dataarray, numeric_name_prefix="", include_orig_name=True)
    assert "original_name" not in out_da.attrs

    out_da = _preprocess_data_array_name(dataarray, numeric_name_prefix=False, include_orig_name=True)
    assert "original_name" not in out_da.attrs

    out_da = _preprocess_data_array_name(dataarray, numeric_name_prefix=None, include_orig_name=True)
    assert "original_name" not in out_da.attrs


def test_make_cf_dataarray_lonlat():
    """Test correct CF encoding for area with lon/lat units."""
    from pyresample import create_area_def

    from satpy.cf.data_array import make_cf_data_array
    from satpy.resample import add_crs_xy_coords

    area = create_area_def("mavas", 4326, shape=(5, 5),
                           center=(0, 0), resolution=(1, 1))
    da = xr.DataArray(
        np.arange(25).reshape(5, 5),
        dims=("y", "x"),
        attrs={"area": area})
    da = add_crs_xy_coords(da, area)
    new_da = make_cf_data_array(da)
    assert new_da["x"].attrs["units"] == "degrees_east"
    assert new_da["y"].attrs["units"] == "degrees_north"


class TestCfDataArray:
    """Test creation of CF DataArray."""

    def test_make_cf_dataarray(self):
        """Test the conversion of a DataArray to a CF-compatible DataArray."""
        from satpy.cf.data_array import make_cf_data_array
        from satpy.tests.cf_tests._test_data import get_test_attrs
        from satpy.tests.utils import assert_dict_array_equality

        # Create set of test attributes
        attrs, attrs_expected, attrs_expected_flat = get_test_attrs()
        attrs["area"] = "some_area"
        attrs["prerequisites"] = [make_dsq(name="hej")]
        attrs["_satpy_id_name"] = "myname"

        # Adjust expected attributes
        expected_prereq = ("DataQuery(name='hej')")
        update = {"prerequisites": [expected_prereq], "long_name": attrs["name"]}

        attrs_expected.update(update)
        attrs_expected_flat.update(update)

        attrs_expected.pop("name")
        attrs_expected_flat.pop("name")

        # Create test data array
        arr = xr.DataArray(np.array([[1, 2], [3, 4]]), attrs=attrs, dims=("y", "x"),
                           coords={"y": [0, 1], "x": [1, 2], "acq_time": ("y", [3, 4])})

        # Test conversion to something cf-compliant
        res = make_cf_data_array(arr)
        np.testing.assert_array_equal(res["x"], arr["x"])
        np.testing.assert_array_equal(res["y"], arr["y"])
        np.testing.assert_array_equal(res["acq_time"], arr["acq_time"])
        assert res["x"].attrs == {"units": "m", "standard_name": "projection_x_coordinate"}
        assert res["y"].attrs == {"units": "m", "standard_name": "projection_y_coordinate"}
        assert_dict_array_equality(res.attrs, attrs_expected)

        # Test attribute kwargs
        res_flat = make_cf_data_array(arr, flatten_attrs=True, exclude_attrs=["int"])
        attrs_expected_flat.pop("int")
        assert_dict_array_equality(res_flat.attrs, attrs_expected_flat)

    def test_make_cf_dataarray_one_dimensional_array(self):
        """Test the conversion of an 1d DataArray to a CF-compatible DataArray."""
        from satpy.cf.data_array import make_cf_data_array

        arr = xr.DataArray(np.array([1, 2, 3, 4]), attrs={}, dims=("y",),
                           coords={"y": [0, 1, 2, 3], "acq_time": ("y", [0, 1, 2, 3])})
        _ = make_cf_data_array(arr)
