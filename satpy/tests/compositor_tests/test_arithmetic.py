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

"""Tests for arithmetic compositors."""

import datetime as dt
import unittest

import dask.array as da
import numpy as np
import pytest
import xarray as xr


class TestDifferenceCompositor(unittest.TestCase):
    """Test case for the difference compositor."""

    def setUp(self):
        """Create test data."""
        from pyresample.geometry import AreaDefinition
        area = AreaDefinition("test", "test", "test",
                              {"proj": "merc"}, 2, 2,
                              (-2000, -2000, 2000, 2000))
        attrs = {"area": area,
                 "start_time": dt.datetime(2018, 1, 1, 18),
                 "modifiers": tuple(),
                 "resolution": 1000,
                 "name": "test_vis"}
        ds1 = xr.DataArray(da.ones((2, 2), chunks=2, dtype=np.float64),
                           attrs=attrs, dims=("y", "x"),
                           coords={"y": [0, 1], "x": [0, 1]})
        self.ds1 = ds1
        ds2 = xr.DataArray(da.ones((2, 2), chunks=2, dtype=np.float64) + 2,
                           attrs=attrs, dims=("y", "x"),
                           coords={"y": [0, 1], "x": [0, 1]})
        ds2.attrs["name"] += "2"
        self.ds2 = ds2

        # high res version
        ds2 = xr.DataArray(da.ones((4, 4), chunks=2, dtype=np.float64) + 4,
                           attrs=attrs.copy(), dims=("y", "x"),
                           coords={"y": [0, 1, 2, 3], "x": [0, 1, 2, 3]})
        ds2.attrs["name"] += "2"
        ds2.attrs["resolution"] = 500
        ds2.attrs["rows_per_scan"] = 1
        ds2.attrs["area"] = AreaDefinition("test", "test", "test",
                                           {"proj": "merc"}, 4, 4,
                                           (-2000, -2000, 2000, 2000))
        self.ds2_big = ds2

    def test_basic_diff(self):
        """Test that a basic difference composite works."""
        from satpy.composites.arithmetic import DifferenceCompositor
        comp = DifferenceCompositor(name="diff", standard_name="temperature_difference")
        res = comp((self.ds1, self.ds2))
        np.testing.assert_allclose(res.values, -2)
        assert res.attrs.get("standard_name") == "temperature_difference"

    def test_bad_areas_diff(self):
        """Test that a difference where resolutions are different fails."""
        from satpy.composites.arithmetic import DifferenceCompositor
        from satpy.composites.core import IncompatibleAreas
        comp = DifferenceCompositor(name="diff")
        # too many arguments
        with pytest.raises(ValueError, match="Expected 2 datasets, got 3"):
            comp((self.ds1, self.ds2, self.ds2_big))
        # different resolution
        with pytest.raises(IncompatibleAreas):
            comp((self.ds1, self.ds2_big))


@pytest.fixture
def fake_area():
    """Return a fake 2×2 area."""
    from pyresample.geometry import create_area_def
    return create_area_def("skierffe", 4087, area_extent=[-5_000, -5_000, 5_000, 5_000], shape=(2, 2))


@pytest.fixture
def fake_dataset_pair(fake_area):
    """Return a fake pair of 2×2 datasets."""
    ds1 = xr.DataArray(da.full((2, 2), 8, chunks=2, dtype=np.float32),
                       attrs={"area": fake_area, "standard_name": "toa_bidirectional_reflectance"})
    ds2 = xr.DataArray(da.full((2, 2), 4, chunks=2, dtype=np.float32),
                       attrs={"area": fake_area, "standard_name": "toa_bidirectional_reflectance"})
    return (ds1, ds2)


@pytest.mark.parametrize("kwargs", [{}, {"standard_name": "channel_ratio", "foo": "bar"}])
def test_ratio_compositor(fake_dataset_pair, kwargs):
    """Test the ratio compositor."""
    from satpy.composites.arithmetic import RatioCompositor
    comp = RatioCompositor("ratio", **kwargs)
    res = comp(fake_dataset_pair)
    np.testing.assert_allclose(res.values, 2)

    assert res.attrs["name"] == "ratio"

    if "standard_name" in kwargs:
        # See that the kwargs have been updated to the attrs
        assert res.attrs["standard_name"] == "channel_ratio"
        assert res.attrs["foo"] == "bar"
    else:
        assert res.attrs["standard_name"] == "toa_bidirectional_reflectance"


def test_sum_compositor(fake_dataset_pair):
    """Test the sum compositor."""
    from satpy.composites.arithmetic import SumCompositor
    comp = SumCompositor(name="sum", standard_name="channel_sum")
    res = comp(fake_dataset_pair)
    np.testing.assert_allclose(res.values, 12)
