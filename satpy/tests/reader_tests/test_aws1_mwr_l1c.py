#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2024-2025 Satpy developers

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

"""Tests for ESA Arctic Weather Satellite (AWS) level-1c file reading."""


import numpy as np
import pytest

from satpy.tests.reader_tests.conftest import make_fake_angles, make_fake_mwr_l1c_lonlats

PLATFORM_NAME = "AWS1"

geo_dims = ["n_scans", "n_fovs"]
geo_size = 10 * 145
fake_lon_data, fake_lat_data = make_fake_mwr_l1c_lonlats(geo_size, geo_dims)
fake_sun_azi_data = make_fake_angles(geo_size, geo_dims, shape=(10, 145))
fake_sun_zen_data = make_fake_angles(geo_size, geo_dims, shape=(10, 145))
fake_sat_azi_data = make_fake_angles(geo_size, geo_dims, shape=(10, 145))
fake_sat_zen_data = make_fake_angles(geo_size, geo_dims, shape=(10, 145))


def test_get_channel_data(aws_mwr_l1c_handler, fake_mwr_data_array):
    """Test retrieving the channel data."""
    did = dict(name="1")
    dataset_info = dict(file_key="data/calibration/aws_toa_brightness_temperature")
    expected = fake_mwr_data_array.isel(n_channels=0)
    # mask no_data value
    expected = expected.where(expected != -2147483648)
    # mask outside the valid range
    expected = expected.where(expected <= 700000)
    expected = expected.where(expected >= 0)
    # "calibrate"
    expected = expected * 0.001
    res = aws_mwr_l1c_handler.get_dataset(did, dataset_info)
    np.testing.assert_allclose(res, expected)
    assert "x" in res.dims
    assert "y" in res.dims
    assert res.dims == ("y", "x")
    assert "n_channels" not in res.coords
    assert res.attrs["sensor"] == "mwr"
    assert res.attrs["platform_name"] == PLATFORM_NAME


@pytest.mark.parametrize(("id_name", "file_key", "fake_array"),
                         [("longitude", "data/navigation/aws_lon", fake_lon_data * 1e-4),
                          ("latitude", "data/navigation/aws_lat", fake_lat_data),
                          ])
def test_get_navigation_data(aws_mwr_l1c_handler, id_name, file_key, fake_array):
    """Test retrieving the geolocation (lon, lat) data."""
    did = dict(name=id_name)
    dataset_info = dict(file_key=file_key, standard_name=id_name)
    res = aws_mwr_l1c_handler.get_dataset(did, dataset_info)
    if id_name == "longitude":
        fake_array = fake_array.where(fake_array <= 180, fake_array - 360)

    np.testing.assert_allclose(res, fake_array)
    assert "x" in res.dims
    assert "y" in res.dims
    assert res.dims == ("y", "x")
    assert "standard_name" in res.attrs
    if id_name == "longitude":
        assert res.max() <= 180


@pytest.mark.parametrize(("id_name", "file_key", "fake_array"),
                         [("solar_azimuth_angle", "data/navigation/aws_solar_azimuth_angle", fake_sun_azi_data),
                          ("solar_zenith_angle", "data/navigation/aws_solar_zenith_angle", fake_sun_zen_data),
                          ("satellite_azimuth_angle", "data/navigation/aws_satellite_azimuth_angle", fake_sat_azi_data),
                          ("satellite_zenith_angle", "data/navigation/aws_satellite_zenith_angle", fake_sat_zen_data)])
def test_get_viewing_geometry_data(aws_mwr_l1c_handler, id_name, file_key, fake_array):
    """Test retrieving the angles_data."""
    dset_id = dict(name=id_name)
    dataset_info = dict(file_key=file_key, standard_name=id_name)
    res = aws_mwr_l1c_handler.get_dataset(dset_id, dataset_info)
    np.testing.assert_allclose(res, fake_array)
    assert "x" in res.dims
    assert "y" in res.dims
    assert res.dims == ("y", "x")
    assert "standard_name" in res.attrs
