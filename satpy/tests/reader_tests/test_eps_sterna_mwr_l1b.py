#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2024 Satpy developers

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

import os
from datetime import datetime
from enum import Enum

import numpy as np
import pytest
import xarray as xr
from trollsift import compose, parse
from xarray import DataTree

from satpy.readers.mwr_l1b import DATETIME_FORMAT, AWS_EPS_Sterna_MWR_L1BFile
from satpy.tests.reader_tests.test_aws1_mwr_l1b import random_date

platform_name = "AWS1"
# W_XX-EUMETSAT-Darmstadt,SAT,AWS1-MWR-1B-RAD_C_EUMT_20241121085911_G_D_20241109234502_20241110004559_T_N____.nc
file_pattern = "W_XX-EUMETSAT-Darmstadt,SAT,{platform_name}-MWR-1B-RAD_C_OHB_{processing_time:%Y%m%d%H%M%S}_G_D_{start_time:%Y%m%d%H%M%S}_{end_time:%Y%m%d%H%M%S}_T_B____.nc"  # noqa

rng = np.random.default_rng()

fake_data_np = rng.integers(0, 700000, size=10*145*19).reshape((10, 145, 19))
fake_data_np[0, 0, 0] = -2147483648
fake_data_np[1, 0, 0] = 700000 + 10
fake_data_np[2, 0, 0] = -10

ARRAY_DIMS = ["n_scans", "n_fovs", "n_channels"]
fake_data = xr.DataArray(fake_data_np, dims=ARRAY_DIMS)

GEO_DIMS = ["n_scans", "n_fovs", "n_feedhorns"]
GEO_SIZE = 10*145*4
fake_lon_data = xr.DataArray(rng.integers(0, 3599999, size=GEO_SIZE).reshape((10, 145, 4)), dims=GEO_DIMS)
fake_lat_data = xr.DataArray(rng.integers(-900000, 900000, size=GEO_SIZE).reshape((10, 145, 4)), dims=GEO_DIMS)
fake_sun_azi_data = xr.DataArray(rng.integers(0, 36000, size=GEO_SIZE).reshape((10, 145, 4)), dims=GEO_DIMS)
fake_sun_zen_data = xr.DataArray(rng.integers(0, 36000, size=GEO_SIZE).reshape((10, 145, 4)), dims=GEO_DIMS)
fake_sat_azi_data = xr.DataArray(rng.integers(0, 36000, size=GEO_SIZE).reshape((10, 145, 4)), dims=GEO_DIMS)
fake_sat_zen_data = xr.DataArray(rng.integers(0, 36000, size=GEO_SIZE).reshape((10, 145, 4)), dims=GEO_DIMS)


@pytest.fixture(scope="session")
def eps_sterna_mwr_file(tmp_path_factory):
    """Create an EPS-Sterna MWR l1b file."""
    ds = DataTree()
    start_time = datetime(2024, 9, 1, 12, 0)
    ds.attrs["sensing_start_time_utc"] = start_time.strftime(DATETIME_FORMAT)
    end_time = datetime(2024, 9, 1, 12, 15)
    ds.attrs["sensing_end_time_utc"] = end_time.strftime(DATETIME_FORMAT)
    processing_time = random_date(datetime(2024, 6, 1), datetime(2030, 6, 1))

    instrument = "MWR"
    ds.attrs["instrument"] = instrument
    ds.attrs["orbit_start"] = 9991
    ds.attrs["orbit_end"] = 9992
    ds["data/calibration/toa_brightness_temperature"] = fake_data
    ds["data/calibration/toa_brightness_temperature"].attrs["scale_factor"] = 0.001
    ds["data/calibration/toa_brightness_temperature"].attrs["add_offset"] = 0.0
    ds["data/calibration/toa_brightness_temperature"].attrs["missing_value"] = -2147483648
    ds["data/calibration/toa_brightness_temperature"].attrs["valid_min"] = 0
    ds["data/calibration/toa_brightness_temperature"].attrs["valid_max"] = 700000

    ds["data/navigation/longitude"] = fake_lon_data
    ds["data/navigation/longitude"].attrs["scale_factor"] = 1e-4
    ds["data/navigation/longitude"].attrs["add_offset"] = 0.0
    ds["data/navigation/latitude"] = fake_lat_data
    ds["data/navigation/solar_azimuth_angle"] = fake_sun_azi_data
    ds["data/navigation/solar_zenith_angle"] = fake_sun_zen_data
    ds["data/navigation/satellite_azimuth_angle"] = fake_sat_azi_data
    ds["data/navigation/satellite_zenith_angle"] = fake_sat_zen_data
    ds["status/satellite/subsat_latitude_end"] = np.array(22.39)
    ds["status/satellite/subsat_longitude_start"] = np.array(304.79)
    ds["status/satellite/subsat_latitude_start"] = np.array(55.41)
    ds["status/satellite/subsat_longitude_end"] = np.array(296.79)

    tmp_dir = tmp_path_factory.mktemp("eps_sterna_mwr_l1b_tests")
    filename = tmp_dir / compose(file_pattern, dict(start_time=start_time, end_time=end_time,
                                                    processing_time=processing_time, platform_name=platform_name))

    ds.to_netcdf(filename)
    return filename


@pytest.fixture
def mwr_handler(eps_sterna_mwr_file):
    """Create an EPS-Sterna MWR filehandler."""
    filename_info = parse(file_pattern, os.path.basename(eps_sterna_mwr_file))
    filetype_info = dict()
    filetype_info["file_type"] = "eps_sterna_mwr_l1b"
    return AWS_EPS_Sterna_MWR_L1BFile(eps_sterna_mwr_file, filename_info, filetype_info)


@pytest.mark.parametrize(("id_name", "file_key", "fake_array"),
                         [("longitude", "data/navigation/longitude", fake_lon_data * 1e-4),
                          ("latitude", "data/navigation/latitude", fake_lat_data),
                          ])
def test_get_navigation_data(mwr_handler, id_name, file_key, fake_array):
    """Test retrieving the geolocation (lon-lat) data."""
    Horn = Enum("Horn", ["1", "2", "3", "4"])
    did = dict(name=id_name, horn=Horn["1"])
    dataset_info = dict(file_key=file_key, standard_name=id_name)
    res = mwr_handler.get_dataset(did, dataset_info)
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
