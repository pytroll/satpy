"""Tests for aws l1b filehandlers."""

import os
from datetime import datetime, timedelta
from enum import Enum
from random import randrange

import numpy as np
import pytest
import xarray as xr
from datatree import DataTree
from trollsift import compose, parse

from satpy.readers.aws_l1b import DATETIME_FORMAT, AWSL1BFile

platform_name = "AWS1"
file_pattern = "W_XX-OHB-Stockholm,SAT,{platform_name}-MWR-1B-RAD_C_OHB_{processing_time:%Y%m%d%H%M%S}_G_D_{start_time:%Y%m%d%H%M%S}_{end_time:%Y%m%d%H%M%S}_T_B____.nc"  # noqa

rng = np.random.default_rng()

fake_data_np = rng.integers(0, 700000, size=10*145*19).reshape((10, 145, 19))
fake_data_np[0, 0, 0] = -2147483648
fake_data_np[1, 0, 0] = 700000 + 10
fake_data_np[2, 0, 0] = -10

ARRAY_DIMS = ["n_scans", "n_fovs", "n_channels"]
fake_data = xr.DataArray(fake_data_np, dims=ARRAY_DIMS)

GEO_DIMS = ["n_scans", "n_fovs", "n_geo_groups"]
GEO_SIZE = 10*145*4
fake_lon_data = xr.DataArray(rng.integers(0, 3599999, size=GEO_SIZE).reshape((10, 145, 4)), dims=GEO_DIMS)
fake_lat_data = xr.DataArray(rng.integers(-900000, 900000, size=GEO_SIZE).reshape((10, 145, 4)), dims=GEO_DIMS)
fake_sun_azi_data = xr.DataArray(rng.integers(0, 36000, size=GEO_SIZE).reshape((10, 145, 4)), dims=GEO_DIMS)
fake_sun_zen_data = xr.DataArray(rng.integers(0, 36000, size=GEO_SIZE).reshape((10, 145, 4)), dims=GEO_DIMS)
fake_sat_azi_data = xr.DataArray(rng.integers(0, 36000, size=GEO_SIZE).reshape((10, 145, 4)), dims=GEO_DIMS)
fake_sat_zen_data = xr.DataArray(rng.integers(0, 36000, size=GEO_SIZE).reshape((10, 145, 4)), dims=GEO_DIMS)


def random_date(start, end):
    """Create a random datetime between two datetimes."""
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = randrange(int_delta)
    return start + timedelta(seconds=random_second)


@pytest.fixture(scope="session")
def aws_file(tmp_path_factory):
    """Create an AWS file."""
    ds = DataTree()
    start_time = datetime(2024, 9, 1, 12, 0)
    ds.attrs["sensing_start_time_utc"] = start_time.strftime(DATETIME_FORMAT)
    end_time = datetime(2024, 9, 1, 12, 15)
    ds.attrs["sensing_end_time_utc"] = end_time.strftime(DATETIME_FORMAT)
    processing_time = random_date(datetime(2024, 6, 1), datetime(2030, 6, 1))

    instrument = "AWS"
    ds.attrs["instrument"] = instrument
    ds.attrs["orbit_start"] = 9991
    ds.attrs["orbit_end"] = 9992
    ds["data/calibration/aws_toa_brightness_temperature"] = fake_data
    ds["data/calibration/aws_toa_brightness_temperature"].attrs["scale_factor"] = 0.001
    ds["data/calibration/aws_toa_brightness_temperature"].attrs["add_offset"] = 0.0
    ds["data/calibration/aws_toa_brightness_temperature"].attrs["missing_value"] = -2147483648
    ds["data/calibration/aws_toa_brightness_temperature"].attrs["valid_min"] = 0
    ds["data/calibration/aws_toa_brightness_temperature"].attrs["valid_max"] = 700000

    ds["data/navigation/aws_lon"] = fake_lon_data
    ds["data/navigation/aws_lon"].attrs["scale_factor"] = 1e-4
    ds["data/navigation/aws_lon"].attrs["add_offset"] = 0.0
    ds["data/navigation/aws_lat"] = fake_lat_data
    ds["data/navigation/aws_solar_azimuth_angle"] = fake_sun_azi_data
    ds["data/navigation/aws_solar_zenith_angle"] = fake_sun_zen_data
    ds["data/navigation/aws_satellite_azimuth_angle"] = fake_sat_azi_data
    ds["data/navigation/aws_satellite_zenith_angle"] = fake_sat_zen_data
    ds["status/satellite/subsat_latitude_end"] = np.array(22.39)
    ds["status/satellite/subsat_longitude_start"] = np.array(304.79)
    ds["status/satellite/subsat_latitude_start"] = np.array(55.41)
    ds["status/satellite/subsat_longitude_end"] = np.array(296.79)

    tmp_dir = tmp_path_factory.mktemp("aws_l1b_tests")
    filename = tmp_dir / compose(file_pattern, dict(start_time=start_time, end_time=end_time,
                                                    processing_time=processing_time, platform_name=platform_name))

    ds.to_netcdf(filename)
    return filename


@pytest.fixture
def aws_handler(aws_file):
    """Create an aws filehandler."""
    filename_info = parse(file_pattern, os.path.basename(aws_file))
    filetype_info = dict()
    filetype_info["file_type"] = "aws_l1b"
    return AWSL1BFile(aws_file, filename_info, filetype_info)


def test_start_end_time(aws_handler):
    """Test that start and end times are read correctly."""
    assert aws_handler.start_time == datetime(2024, 9, 1, 12, 0)
    assert aws_handler.end_time == datetime(2024, 9, 1, 12, 15)


def test_metadata(aws_handler):
    """Test that the metadata is read correctly."""
    assert aws_handler.sensor == "AWS"
    assert aws_handler.platform_name == platform_name


def test_get_channel_data(aws_handler):
    """Test retrieving the channel data."""
    did = dict(name="1")
    dataset_info = dict(file_key="data/calibration/aws_toa_brightness_temperature")
    expected = fake_data.isel(n_channels=0)
    # mask no_data value
    expected = expected.where(expected != -2147483648)
    # mask outside the valid range
    expected = expected.where(expected <= 700000)
    expected = expected.where(expected >= 0)
    # "calibrate"
    expected = expected * 0.001
    res = aws_handler.get_dataset(did, dataset_info)
    np.testing.assert_allclose(res, expected)
    assert "x" in res.dims
    assert "y" in res.dims
    assert "orbital_parameters" in res.attrs
    assert res.attrs["orbital_parameters"]["sub_satellite_longitude_end"] == 296.79
    assert res.dims == ("y", "x")
    assert "n_channels" not in res.coords
    assert res.attrs["sensor"] == "AWS"
    assert res.attrs["platform_name"] == "AWS1"


@pytest.mark.parametrize(("id_name", "file_key", "fake_array"),
                         [("longitude", "data/navigation/aws_lon", fake_lon_data * 1e-4),
                          ("latitude", "data/navigation/aws_lat", fake_lat_data),
                          ])
def test_get_navigation_data(aws_handler, id_name, file_key, fake_array):
    """Test retrieving the geolocation (lon-lat) data."""
    Horn = Enum("Horn", ["1", "2", "3", "4"])
    did = dict(name=id_name, horn=Horn["1"])
    dataset_info = dict(file_key=file_key, standard_name=id_name)
    res = aws_handler.get_dataset(did, dataset_info)
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
def test_get_viewing_geometry_data(aws_handler, id_name, file_key, fake_array):
    """Test retrieving the angles_data."""
    Horn = Enum("Horn", ["1", "2", "3", "4"])
    dset_id = dict(name=id_name, horn=Horn["1"])

    dataset_info = dict(file_key=file_key, standard_name=id_name)
    res = aws_handler.get_dataset(dset_id, dataset_info)

    np.testing.assert_allclose(res, fake_array.isel(n_geo_groups=0))
    assert "x" in res.dims
    assert "y" in res.dims
    assert "orbital_parameters" in res.attrs
    assert res.dims == ("y", "x")
    assert "standard_name" in res.attrs
    assert "n_geo_groups" not in res.coords
    if id_name == "longitude":
        assert res.max() <= 180
