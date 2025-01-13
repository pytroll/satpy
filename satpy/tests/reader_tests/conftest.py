#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2021, 2024, 2025 Satpy developers
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

"""Setup and configuration for all reader tests."""

import datetime as dt
import os
from random import randrange

import numpy as np
import pytest
import xarray as xr
from trollsift import compose, parse
from xarray import DataTree

from satpy.readers.mwr_l1b import AWS_EPS_Sterna_MWR_L1BFile
from satpy.readers.mwr_l1c import AWS_MWR_L1CFile

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

platform_name = "AWS1"
# W_XX-EUMETSAT-Darmstadt,SAT,AWS1-MWR-1B-RAD_C_EUMT_20241121085911_G_D_20241109234502_20241110004559_T_N____.nc
file_pattern = "W_{country:2s}-{organisation:s}-{location:s},SAT,{platform_name}-MWR-{processing_level}-RAD_C_{originator:4s}_{processing_time:%Y%m%d%H%M%S}_G_D_{start_time:%Y%m%d%H%M%S}_{end_time:%Y%m%d%H%M%S}_T_B____.nc"  # noqa


rng = np.random.default_rng()

def random_date(start, end):
    """Create a random datetime between two datetimes."""
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = randrange(int_delta)
    return start + dt.timedelta(seconds=random_second)


@pytest.fixture(scope="module")
def fake_mwr_data_array():
    """Return a fake AWS/EPS-Sterna MWR l1b data array."""
    fake_data_np = rng.integers(0, 700000, size=10*145*19).reshape((10, 145, 19))
    fake_data_np[0, 0, 0] = -2147483648
    fake_data_np[1, 0, 0] = 700000 + 10
    fake_data_np[2, 0, 0] = -10
    array_dims = ["n_scans", "n_fovs", "n_channels"]
    return xr.DataArray(fake_data_np, dims=array_dims)


def make_fake_angles(geo_size, geo_dims, shape):
    """Return fake sun-satellite angle array."""
    maxval = 36000
    dummy_array = (np.arange(0, geo_size) * maxval/geo_size).astype("int32")
    return xr.DataArray(dummy_array.reshape(shape), dims=geo_dims)


def make_fake_mwr_lonlats(geo_size, geo_dims, shape):
    """Return fake geolocation data arrays for all 4 MWR horns."""
    maxval = 3600000
    dummy_array = (np.arange(0, geo_size) * maxval/geo_size).astype("int32")
    fake_lon_data = xr.DataArray(dummy_array.reshape(shape), dims=geo_dims)
    maxval = 1800000
    dummy_array = (np.arange(0, geo_size) * maxval/geo_size - maxval/2).astype("int32")
    fake_lat_data = xr.DataArray(dummy_array.reshape(shape), dims=geo_dims)
    return (fake_lon_data, fake_lat_data)


def make_fake_mwr_l1c_lonlats(geo_size, geo_dims):
    """Return fake level-1c geolocation data arrays."""
    maxval = 3600000
    dummy_array = (np.arange(0, geo_size) * maxval/geo_size).astype("int32")
    fake_lon_data = xr.DataArray(dummy_array.reshape((10, 145)), dims=geo_dims)
    maxval = 1800000
    dummy_array = (np.arange(0, geo_size) * maxval/geo_size - maxval/2).astype("int32")
    fake_lat_data = xr.DataArray(dummy_array.reshape((10, 145)), dims=geo_dims)
    return (fake_lon_data, fake_lat_data)


def aws_eps_sterna_mwr_level1_file(fake_mwr_data_array, eps_sterna=True, l1b=True):
    """Create an AWS and EPS-Sterna MWR l1b file."""
    if eps_sterna:
        n_feedhorns="n_feedhorns"
        prefix = ""
        longitude_attr = "longitude"
        latitude_attr = "latitude"
    else:
        n_feedhorns="n_geo_groups"
        prefix = "aws_"
        longitude_attr = "aws_lon"
        latitude_attr = "aws_lat"

    if l1b:
        geo_dims = ["n_scans", "n_fovs", n_feedhorns]
        geo_size = 10 * 145 * 4
        shape = (10, 145, 4)
    else:
        geo_dims = ["n_scans", "n_fovs"]
        geo_size = 10 * 145
        shape = (10, 145)

    ds = DataTree()
    start_time = dt.datetime(2024, 9, 1, 12, 0)
    ds.attrs["sensing_start_time_utc"] = start_time.strftime(DATETIME_FORMAT)
    end_time = dt.datetime(2024, 9, 1, 12, 15)
    ds.attrs["sensing_end_time_utc"] = end_time.strftime(DATETIME_FORMAT)

    ds.attrs["instrument"] = "MWR"
    ds.attrs["orbit_start"] = 9991
    ds.attrs["orbit_end"] = 9992
    dset_name = f"data/calibration/{prefix}toa_brightness_temperature"
    ds[dset_name] = fake_mwr_data_array
    ds[dset_name].attrs["scale_factor"] = 0.001
    ds[dset_name].attrs["add_offset"] = 0.0
    ds[dset_name].attrs["missing_value"] = -2147483648
    ds[dset_name].attrs["valid_min"] = 0
    ds[dset_name].attrs["valid_max"] = 700000

    fake_lon_data, fake_lat_data = make_fake_mwr_lonlats(geo_size, geo_dims, shape)

    ds[f"data/navigation/{longitude_attr}"] = fake_lon_data
    ds[f"data/navigation/{longitude_attr}"].attrs["scale_factor"] = 1e-4
    ds[f"data/navigation/{longitude_attr}"].attrs["add_offset"] = 0.0
    ds[f"data/navigation/{latitude_attr}"] = fake_lat_data
    ds[f"data/navigation/{prefix}solar_azimuth_angle"] = make_fake_angles(geo_size, geo_dims, shape)
    ds[f"data/navigation/{prefix}solar_zenith_angle"] = make_fake_angles(geo_size, geo_dims, shape)
    ds[f"data/navigation/{prefix}satellite_azimuth_angle"] = make_fake_angles(geo_size, geo_dims, shape)
    ds[f"data/navigation/{prefix}satellite_zenith_angle"] = make_fake_angles(geo_size, geo_dims, shape)
    if l1b:
        ds["status/satellite/subsat_latitude_end"] = np.array(22.39)
        ds["status/satellite/subsat_longitude_start"] = np.array(304.79)
        ds["status/satellite/subsat_latitude_start"] = np.array(55.41)
        ds["status/satellite/subsat_longitude_end"] = np.array(296.79)

    return ds


def create_mwr_file(tmpdir, data_array, eps_sterna=False, l1b=True):
    """Create an AWS or EPS-Sterna MWR l1b (or level-1c) file."""
    ds = aws_eps_sterna_mwr_level1_file(data_array, eps_sterna=eps_sterna, l1b=l1b)
    start_time = dt.datetime.fromisoformat(ds.attrs["sensing_start_time_utc"])
    end_time = dt.datetime.fromisoformat(ds.attrs["sensing_end_time_utc"])

    platform_name = "ST01" if eps_sterna else "AWS1"
    processing_level = "1B" if l1b else "1C"

    processing_time = random_date(dt.datetime(2024, 9, 1, 13), dt.datetime(2030, 6, 1))
    filename = tmpdir / compose(file_pattern, dict(country="XX",
                                                   organisation="EUMETSAT",
                                                   location="Darmstadt",
                                                   processing_level=processing_level,
                                                   originator="EUMT",
                                                   start_time=start_time, end_time=end_time,
                                                   processing_time=processing_time,
                                                   platform_name=platform_name))
    ds.to_netcdf(filename)
    return filename

@pytest.fixture(scope="module")
def eps_sterna_mwr_file(tmp_path_factory, fake_mwr_data_array):
    """Create an EPS-Sterna MWR l1b file."""
    tmpdir = tmp_path_factory.mktemp("eps_sterna_mwr_l1b_tests")
    return create_mwr_file(tmpdir, fake_mwr_data_array, eps_sterna=True)


@pytest.fixture(scope="module")
def aws_mwr_file(tmp_path_factory, fake_mwr_data_array):
    """Create an AWS MWR l1b file."""
    tmpdir = tmp_path_factory.mktemp("aws_l1b_tests")
    return create_mwr_file(tmpdir, fake_mwr_data_array, eps_sterna=False)


@pytest.fixture(scope="module")
def aws_mwr_l1c_file(tmp_path_factory, fake_mwr_data_array):
    """Create an AWS MWR l1c file."""
    tmpdir = tmp_path_factory.mktemp("aws_l1c_tests")
    return create_mwr_file(tmpdir, fake_mwr_data_array, eps_sterna=False, l1b=False)


@pytest.fixture(scope="module")
def eps_sterna_mwr_handler(eps_sterna_mwr_file):
    """Create an EPS-Sterna MWR filehandler."""
    filename_info = parse(file_pattern, os.path.basename(eps_sterna_mwr_file))
    filetype_info = dict()
    filetype_info["file_type"] = "eps_sterna_mwr_l1b"
    filetype_info["feed_horn_group_name"] = "n_feedhorns"
    return AWS_EPS_Sterna_MWR_L1BFile(eps_sterna_mwr_file, filename_info, filetype_info)


@pytest.fixture(scope="module")
def aws_mwr_handler(aws_mwr_file):
    """Create an AWS MWR filehandler."""
    filename_info = parse(file_pattern, os.path.basename(aws_mwr_file))
    filetype_info = dict()
    filetype_info["file_type"] = "aws1_mwr_l1b"
    filetype_info["feed_horn_group_name"] = "n_geo_groups"
    return AWS_EPS_Sterna_MWR_L1BFile(aws_mwr_file, filename_info, filetype_info)


@pytest.fixture(scope="module")
def aws_mwr_l1c_handler(aws_mwr_l1c_file):
    """Create an AWS MWR level-1c filehandler."""
    filename_info = parse(file_pattern, os.path.basename(aws_mwr_l1c_file))
    filetype_info = dict()
    filetype_info["file_type"] = "aws1_mwr_l1c"
    filetype_info["feed_horn_group_name"] = None
    return AWS_MWR_L1CFile(aws_mwr_l1c_file, filename_info, filetype_info)
