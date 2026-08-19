#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Satpy developers
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
"""Tests for dropping negligible compositor coordinates."""

import numpy as np
import xarray as xr

from satpy.composites.core import CompositeBase


def test_drop_coordinates_drops_non_dimension_time_coordinate():
    """Test that a time coordinate on another dimension is dropped."""
    data = xr.DataArray(
        np.arange(12).reshape(3, 4),
        dims=("y", "x"),
        coords={"time": xr.DataArray(np.arange(3), dims=("y",))},
    )

    result = CompositeBase.drop_coordinates([data])[0]

    assert "time" not in result.coords


def test_drop_coordinates_drops_mismatched_same_named_coordinate():
    """Test that a same-named coordinate on the wrong dimension is dropped."""
    data = xr.DataArray(
        np.arange(24).reshape(2, 3, 4),
        dims=("time", "y", "x"),
        coords={"time": xr.DataArray(np.arange(3), dims=("y",))},
    )

    result = CompositeBase.drop_coordinates([data])[0]

    assert "time" not in result.coords


def test_drop_coordinates_keeps_dimension_time_coordinate():
    """Test that a true time dimension coordinate is retained."""
    data = xr.DataArray(
        np.arange(24).reshape(2, 3, 4),
        dims=("time", "y", "x"),
        coords={"time": xr.DataArray(np.arange(2), dims=("time",))},
    )

    result = CompositeBase.drop_coordinates([data])[0]

    assert result.coords["time"].dims == ("time",)
