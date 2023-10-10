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
import xarray as xr

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - caplog


class TestCFtime:
    """Test cases for CF time dimension and coordinates."""

    def test_add_time_bounds_dimension(self):
        """Test addition of CF-compliant time attributes."""
        from satpy.cf.coords import add_time_bounds_dimension

        test_array = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        times = np.array(['2018-05-30T10:05:00', '2018-05-30T10:05:01',
                          '2018-05-30T10:05:02', '2018-05-30T10:05:03'], dtype=np.datetime64)
        dataarray = xr.DataArray(test_array,
                                 dims=['y', 'x'],
                                 coords={'time': ('y', times)},
                                 attrs=dict(start_time=times[0], end_time=times[-1]))
        ds = dataarray.to_dataset(name='test-array')
        ds = add_time_bounds_dimension(ds)

        assert "bnds_1d" in ds.dims
        assert ds.dims['bnds_1d'] == 2
        assert "time_bnds" in list(ds.data_vars)
        assert "bounds" in ds["time"].attrs
        assert "standard_name" in ds["time"].attrs


class TestCFcoords:
    """Test cases for CF spatial dimension and coordinates."""

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

    # add_xy_coords_attrs
    # process_time_coord
