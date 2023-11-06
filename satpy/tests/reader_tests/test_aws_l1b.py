"""Tests for aws l1b filehandlers."""

import os
from datetime import datetime, timedelta
from random import randrange

import numpy as np
import pytest
import xarray as xr
from datatree import DataTree
from trollsift import compose, parse

from satpy.readers.aws_l1b import DATETIME_FORMAT, AWSL1BFile

platform_name = "AWS1"
file_pattern = "W_XX-OHB-Stockholm,SAT,{platform_name}-MWR-1B-RAD_C_OHB_{processing_time:%Y%m%d%H%M%S}_G_D_{start_time:%Y%m%d%H%M%S}_{end_time:%Y%m%d%H%M%S}_T_B____.nc"  # noqa
fake_data = xr.DataArray(np.random.randint(0, 700000, size=19*5*5).reshape((19, 5, 5)),
                         dims=["n_channels", "n_fovs", "n_scans"])
fake_lon_data = xr.DataArray(np.random.randint(0, 3599999, size=25 * 4).reshape((4, 5, 5)),
                             dims=["n_geo_groups", "n_fovs", "n_scans"])
fake_lat_data = xr.DataArray(np.random.randint(-900000, 900000, size=25 * 4).reshape((4, 5, 5)),
                             dims=["n_geo_groups", "n_fovs", "n_scans"])
fake_sun_azi_data = xr.DataArray(np.random.randint(0, 36000, size=25 * 4).reshape((4, 5, 5)),
                                 dims=["n_geo_groups", "n_fovs", "n_scans"])
fake_sun_zen_data = xr.DataArray(np.random.randint(0, 36000, size=25 * 4).reshape((4, 5, 5)),
                                 dims=["n_geo_groups", "n_fovs", "n_scans"])
fake_sat_azi_data = xr.DataArray(np.random.randint(0, 36000, size=25 * 4).reshape((4, 5, 5)),
                                 dims=["n_geo_groups", "n_fovs", "n_scans"])
fake_sat_zen_data = xr.DataArray(np.random.randint(0, 36000, size=25 * 4).reshape((4, 5, 5)),
                                 dims=["n_geo_groups", "n_fovs", "n_scans"])


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
    start_time = random_date(datetime(2024, 6, 1), datetime(2030, 6, 1))
    ds.attrs["sensing_start_time_utc"] = start_time.strftime(DATETIME_FORMAT)
    end_time = random_date(datetime(2024, 6, 1), datetime(2030, 6, 1))
    ds.attrs["sensing_end_time_utc"] = end_time.strftime(DATETIME_FORMAT)
    processing_time = random_date(datetime(2024, 6, 1), datetime(2030, 6, 1))

    instrument = "AWS"
    ds.attrs["instrument"] = instrument
    ds["data/calibration/aws_toa_brightness_temperature"] = fake_data
    ds["data/calibration/aws_toa_brightness_temperature"].attrs["scale_factor"] = 0.001
    ds["data/calibration/aws_toa_brightness_temperature"].attrs["add_offset"] = 0.0
    ds["data/navigation/aws_lon"] = fake_lon_data
    ds["data/navigation/aws_lat"] = fake_lat_data
    ds["data/navigation/aws_solar_azimuth_angle"] = fake_sun_azi_data
    ds["data/navigation/aws_solar_zenith_angle"] = fake_sun_zen_data
    ds["data/navigation/aws_satellite_azimuth_angle"] = fake_sat_azi_data
    ds["data/navigation/aws_satellite_zenith_angle"] = fake_sat_zen_data

    tmp_dir = tmp_path_factory.mktemp("aws_l1b_tests")
    filename = tmp_dir / compose(file_pattern, dict(start_time=start_time, end_time=end_time,
                                                    processing_time=processing_time, platform_name=platform_name))

    ds.to_netcdf(filename)
    return filename


@pytest.fixture
def aws_handler(aws_file):
    """Create an aws filehandler."""
    filename_info = parse(file_pattern, os.path.basename(aws_file))
    return AWSL1BFile(aws_file, filename_info, dict())


def test_start_end_time(aws_file):
    """Test that start and end times are read correctly."""
    filename_info = parse(file_pattern, os.path.basename(aws_file))
    handler = AWSL1BFile(aws_file, filename_info, dict())

    assert handler.start_time == filename_info["start_time"]
    assert handler.end_time == filename_info["end_time"]


def test_metadata(aws_handler):
    """Test that the metadata is read correctly."""
    assert aws_handler.sensor == "AWS"
    assert aws_handler.platform_name == platform_name


def test_get_channel_data(aws_handler):
    """Test retrieving the channel data."""
    did = dict(name="1")
    dataset_info = dict(file_key="data/calibration/aws_toa_brightness_temperature")
    np.testing.assert_allclose(aws_handler.get_dataset(did, dataset_info), fake_data.isel(n_channels=0) * 0.001)


@pytest.mark.parametrize(["id_name", "file_key", "fake_array"],
                         [("lon_horn_1", "data/navigation/aws_lon", fake_lon_data),
                          ("lat_horn_1", "data/navigation/aws_lat", fake_lat_data),
                          ("solar_azimuth_horn_1", "data/navigation/aws_solar_azimuth_angle", fake_sun_azi_data),
                          ("solar_zenith_horn_1", "data/navigation/aws_solar_zenith_angle", fake_sun_zen_data),
                          ("satellite_azimuth_horn_1", "data/navigation/aws_satellite_azimuth_angle",
                           fake_sat_azi_data),
                          ("satellite_zenith_horn_1", "data/navigation/aws_satellite_zenith_angle",
                           fake_sat_zen_data)])
def test_get_navigation_data(aws_handler, id_name, file_key, fake_array):
    """Test retrieving the angles_data."""
    did = dict(name=id_name)
    dataset_info = dict(file_key=file_key, n_horns=0)
    np.testing.assert_allclose(aws_handler.get_dataset(did, dataset_info), fake_array.isel(n_geo_groups=0))


# def test_channel_is_masked_and_scaled():
#     pass

# def test_navigation_is_scaled_and_scaled():
#     pass


# def test_orbital_parameters_are_provided():
#     pass


# def test_coords_contain_xy():
#     pass
