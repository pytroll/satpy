"""Test the flash age compositor."""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 Satpy developers
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


import datetime
import logging

import dask.array as da
import numpy as np
import xarray as xr

from satpy.composites.lightning import LightningTimeCompositor


def test_flash_age_compositor():
    """Test the flash_age compsitor by comparing two xarrays object."""
    comp = LightningTimeCompositor("flash_age",prerequisites=["flash_time"],
                                   standard_name="lightning_time",
                                   time_range=60,
                                   reference_time="end_time")
    attrs_flash_age = {"variable_name": "flash_time","name": "flash_time",
                       "start_time": datetime.datetime(2024, 8, 1, 10, 50, 0),
                       "end_time": datetime.datetime(2024, 8, 1, 11, 0, 0),"reader": "li_l2_nc"}
    flash_age_value = da.array(["2024-08-01T09:00:00",
            "2024-08-01T10:00:00", "2024-08-01T10:30:00","2024-08-01T11:00:00"], dtype="datetime64[ns]")
    flash_age = xr.DataArray(
        flash_age_value,
        dims=["y"],
        coords={
            "crs": "8B +proj=longlat +ellps=WGS84 +type=crs"},
        attrs = attrs_flash_age,
        name="flash_time")
    res = comp([flash_age])
    expected_attrs = {"variable_name": "flash_time","name": "lightning_time",
                       "start_time": datetime.datetime(2024, 8, 1, 10, 50, 0),
                       "end_time": datetime.datetime(2024, 8, 1, 11, 0, 0),"reader": "li_l2_nc",
                       "standard_name": "lightning_time"
                       }
    expected_array = xr.DataArray(da.array([np.nan, 0.0,0.5,1.0]),
                                  dims=["y"],
                                  coords={
                                      "crs": "8B +proj=longlat +ellps=WGS84 +type=crs"},
                                  attrs = expected_attrs,
                                  name="flash_time")
    xr.testing.assert_equal(res,expected_array)

def test_empty_array_error(caplog):
    """Test when the filtered array is empty."""
    comp = LightningTimeCompositor("flash_age",prerequisites=["flash_time"],
                                   standard_name="lightning_time",
                                   time_range=60,
                                   reference_time="end_time")
    attrs_flash_age = {"variable_name": "flash_time","name": "flash_time",
                       "start_time": np.datetime64(datetime.datetime(2024, 8, 1, 10, 0, 0)),
                       "end_time": datetime.datetime(2024, 8, 1, 11, 0, 0),
                       "reader": "li_l2_nc"}
    flash_age_value = da.array(["2024-08-01T09:00:00"], dtype="datetime64[ns]")
    flash_age = xr.DataArray(flash_age_value,
                             dims=["y"],
                             coords={
                                 "crs": "8B +proj=longlat +ellps=WGS84 +type=crs"},
                             attrs = attrs_flash_age,
                             name="flash_time")
    with caplog.at_level(logging.WARNING):
        _ = comp([flash_age])
    # Assert that the log contains the expected warning message
    assert "All the flash_age events happened before" in caplog.text

def test_update_missing_metadata():
    """Test the _update_missing_metadata method."""
    existing_attrs = {
        "standard_name": "lightning_event_time",
        "time_range": 30
    }

    # New metadata to be merged
    new_attrs = {
        "standard_name": None,  # Should not overwrite since it's None
        "reference_time": "2023-09-20T00:00:00Z",  # Should be added
        "units": "seconds"  # Should be added
    }

    # Expected result after merging
    expected_attrs = {
        "standard_name": "lightning_event_time",  # Should remain the same
        "time_range": 30,  # Should remain the same
        "reference_time": "2023-09-20T00:00:00Z",  # Should be added
        "units": "seconds"  # Should be added
    }

    # Call the static method
    LightningTimeCompositor._update_missing_metadata(existing_attrs, new_attrs)

    # Assert the final state of existing_attrs is as expected
    assert existing_attrs == expected_attrs
