#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2024, 2025 Satpy developers

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Tests for the EPS-Sterna MWR l1b filehandlers."""

from enum import Enum

import numpy as np
import pytest

from satpy.tests.reader_tests.conftest import make_fake_mwr_lonlats

geo_dims = ["n_scans", "n_fovs", "n_feedhorns"]
geo_size = 10*145*4
shape = (10, 145, 4)
fake_lon_data, fake_lat_data = make_fake_mwr_lonlats(geo_size, geo_dims, shape)


@pytest.mark.parametrize(("id_name", "file_key", "fake_array"),
                         [("longitude", "data/navigation/longitude", fake_lon_data * 1e-4),
                          ("latitude", "data/navigation/latitude", fake_lat_data),
                          ])
def test_get_navigation_data(eps_sterna_mwr_handler, id_name, file_key, fake_array):
    """Test retrieving the geolocation (lon-lat) data."""
    Horn = Enum("Horn", ["1", "2", "3", "4"])
    did = dict(name=id_name, horn=Horn["1"])
    dataset_info = dict(file_key=file_key, standard_name=id_name)
    res = eps_sterna_mwr_handler.get_dataset(did, dataset_info)
    if id_name == "longitude":
        fake_array = fake_array.where(fake_array <= 180, fake_array - 360)

    np.testing.assert_allclose(res, fake_array.isel(n_feedhorns=0))
    assert "x" in res.dims
    assert "y" in res.dims
    assert "orbital_parameters" in res.attrs
    assert res.dims == ("y", "x")
    assert "standard_name" in res.attrs
    assert "n_feedhorns" not in res.coords
    if id_name == "longitude":
        assert res.max() <= 180


def test_try_get_data_not_in_file(eps_sterna_mwr_handler):
    """Test retrieving a data field that is not available in the file."""
    did = dict(name="aws_toa_brightness_temperature")
    dataset_info = dict(file_key="data/calibration/aws_toa_brightness_temperature")

    match_str = "Dataset aws_toa_brightness_temperature not available or not supported yet!"
    with pytest.raises(NotImplementedError, match=match_str):
        _ = eps_sterna_mwr_handler.get_dataset(did, dataset_info)

def test_metadata(eps_sterna_mwr_handler):
    """Test that the metadata is read correctly."""
    assert eps_sterna_mwr_handler.sensor == "mwr"
    assert eps_sterna_mwr_handler.platform_name == "ST01"
