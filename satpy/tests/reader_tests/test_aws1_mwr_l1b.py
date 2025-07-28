"""Tests for aws l1b filehandlers."""

import datetime as dt
from enum import Enum

import numpy as np
import pytest

from satpy.tests.reader_tests.conftest import make_fake_angles, make_fake_mwr_lonlats

PLATFORM_NAME = "AWS1"


geo_dims = ["n_scans", "n_fovs", "n_geo_groups"]
geo_size = 10*145*4
shape = (10, 145, 4)
fake_lon_data, fake_lat_data = make_fake_mwr_lonlats(geo_size, geo_dims, shape)
fake_sun_azi_data = make_fake_angles(geo_size, geo_dims, shape)
fake_sun_zen_data = make_fake_angles(geo_size, geo_dims, shape)
fake_sat_azi_data = make_fake_angles(geo_size, geo_dims, shape)
fake_sat_zen_data = make_fake_angles(geo_size, geo_dims, shape)



def test_start_end_time(aws_mwr_handler):
    """Test that start and end times are read correctly."""
    assert aws_mwr_handler.start_time == dt.datetime(2024, 9, 1, 12, 0)
    assert aws_mwr_handler.end_time == dt.datetime(2024, 9, 1, 12, 15)


def test_orbit_number_start_end(aws_mwr_handler):
    """Test that start and end orbit number is read correctly."""
    assert aws_mwr_handler.orbit_start == 9991
    assert aws_mwr_handler.orbit_end == 9992


def test_metadata(aws_mwr_handler):
    """Test that the metadata is read correctly."""
    assert aws_mwr_handler.sensor == "mwr"
    assert aws_mwr_handler.platform_name == PLATFORM_NAME


def test_get_channel_data(aws_mwr_handler, fake_mwr_data_array):
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
    res = aws_mwr_handler.get_dataset(did, dataset_info)
    np.testing.assert_allclose(res, expected)
    assert "x" in res.dims
    assert "y" in res.dims
    assert "orbital_parameters" in res.attrs
    assert res.attrs["orbital_parameters"]["sub_satellite_longitude_end"] == 296.79
    assert res.dims == ("y", "x")
    assert "n_channels" not in res.coords
    assert res.attrs["sensor"] == "mwr"
    assert res.attrs["platform_name"] == PLATFORM_NAME


@pytest.mark.parametrize(("id_name", "file_key", "fake_array"),
                         [("longitude", "data/navigation/aws_lon", fake_lon_data * 1e-4),
                          ("latitude", "data/navigation/aws_lat", fake_lat_data),
                          ])
def test_get_navigation_data(aws_mwr_handler, id_name, file_key, fake_array):
    """Test retrieving the geolocation (lon-lat) data."""
    Horn = Enum("Horn", ["1", "2", "3", "4"])
    did = dict(name=id_name, horn=Horn["1"])
    dataset_info = dict(file_key=file_key, standard_name=id_name)
    res = aws_mwr_handler.get_dataset(did, dataset_info)
    if id_name == "longitude":
        fake_array = fake_array.where(fake_array <= 180, fake_array - 360)

    np.testing.assert_allclose(res, fake_array.isel(n_geo_groups=0))
    assert "x" in res.dims
    assert "y" in res.dims
    assert "orbital_parameters" in res.attrs
    assert res.dims == ("y", "x")
    assert "standard_name" in res.attrs
    assert "n_geo_groups" not in res.coords
    if id_name == "longitude":
        assert res.max() <= 180


@pytest.mark.parametrize(("id_name", "file_key", "fake_array"),
                         [("solar_azimuth_horn1", "data/navigation/aws_solar_azimuth_angle", fake_sun_azi_data),
                          ("solar_zenith_horn1", "data/navigation/aws_solar_zenith_angle", fake_sun_zen_data),
                          ("satellite_azimuth_horn1", "data/navigation/aws_satellite_azimuth_angle", fake_sat_azi_data),
                          ("satellite_zenith_horn1", "data/navigation/aws_satellite_zenith_angle", fake_sat_zen_data)])
def test_get_viewing_geometry_data(aws_mwr_handler, id_name, file_key, fake_array):
    """Test retrieving the angles_data."""
    Horn = Enum("Horn", ["1", "2", "3", "4"])
    dset_id = dict(name=id_name, horn=Horn["1"])

    dataset_info = dict(file_key=file_key, standard_name=id_name)
    res = aws_mwr_handler.get_dataset(dset_id, dataset_info)

    np.testing.assert_allclose(res, fake_array.isel(n_geo_groups=0))
    assert "x" in res.dims
    assert "y" in res.dims
    assert "orbital_parameters" in res.attrs
    assert res.dims == ("y", "x")
    assert "standard_name" in res.attrs
    assert "n_geo_groups" not in res.coords


def test_try_get_data_not_in_file(aws_mwr_handler):
    """Test retrieving a data field that is not available in the file."""
    did = dict(name="toa_brightness_temperature")
    dataset_info = dict(file_key="data/calibration/toa_brightness_temperature")

    match_str = "Dataset toa_brightness_temperature not available or not supported yet!"
    with pytest.raises(NotImplementedError, match=match_str):
        _ = aws_mwr_handler.get_dataset(did, dataset_info)
