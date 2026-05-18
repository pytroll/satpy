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
"""Module for testing the satpy.readers.omps_edr module."""

import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pytest
import xarray as xr

START_TIME1 = datetime.datetime(2025, 1, 1, 0, 0, 0)
SINGLE_GRAN_SHAPE = (30, 240)


def _fake_filename(
    start_time: datetime.datetime, end_time: datetime.datetime | None = None, prefix: str = "V8TOZ"
) -> str:
    stime_str = f"{start_time:%Y%m%d%H%M%S}0"
    if end_time is None:
        end_time = start_time + datetime.timedelta(seconds=90)
    etime_str = f"{end_time:%Y%m%d%H%M%S}0"
    ctime_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S0")
    return f"{prefix}-EDR_v4r3_j01_s{stime_str}_e{etime_str}_c{ctime_str}.nc"


def create_v8toz_file(tmp_path: Path, start_time: datetime.datetime) -> Path:
    """Create a fake file for testing."""
    from netCDF4 import Dataset

    end_time = start_time + datetime.timedelta(seconds=90)
    filename = tmp_path / _fake_filename(start_time, end_time=end_time, prefix="V8TOZ")
    rng = np.random.default_rng(12345)

    with Dataset(filename, "w") as nc:
        nc.platform = "NOAA20"
        nc.platform_name = "J01"
        nc.instrument = "OMPS"
        nc.instrument_name = "OMPS"
        nc.time_coverage_start = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        nc.time_coverage_end = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        nc.start_orbit_number = 43448
        nc.end_orbit_number = 43448

        shape = SINGLE_GRAN_SHAPE
        ntimes = nc.createDimension("nTimes", shape[0])
        nifov = nc.createDimension("nIFOV", shape[1])
        lon_var = nc.createVariable("Longitude", np.float32, dimensions=(ntimes, nifov), fill_value=-9999.0)
        lon_var[:] = rng.random(shape).astype(np.float32) * 45.0
        lat_var = nc.createVariable("Latitude", np.float32, dimensions=(ntimes, nifov), fill_value=-9999.0)
        lat_var[:] = rng.random(shape).astype(np.float32) * 45.0

        amount_o3 = nc.createVariable("ColumnAmountO3", np.float32, dimensions=(ntimes, nifov), fill_value=-9999.0)
        amount_o3.valid_range = (0, 1000)
        amount_o3.units = "0.01mm"
        amount_o3[:] = rng.random(shape).astype(np.float32) * 1000

        aerosol_idx = nc.createVariable("AerosolIndex", np.float32, dimensions=(ntimes, nifov), fill_value=-9999.0)
        aerosol_idx.valid_range = (-100, 100)
        aerosol_idx.units = "1"
        aerosol_idx[:] = rng.random(shape).astype(np.float32) * 200 - 100

        refl331 = nc.createVariable("Reflectivity331", np.float32, dimensions=(ntimes, nifov), fill_value=-9999.0)
        refl331.valid_range = (0, 1000)
        refl331.units = "1"
        refl331[:] = rng.random(shape).astype(np.float32) * 100

        err_flag = nc.createVariable("ErrorFlag", np.int32, dimensions=(ntimes, nifov), fill_value=-9999)
        err_flag.valid_range = (0, 10)
        err_flag.units = "1"
        err_data = np.zeros(shape, dtype=np.int32)
        err_data[0, :10] = np.arange(10, dtype=np.int32)
        err_flag[:] = err_data

    return filename


def omps_reader_gen(file_paths: Iterable[Path], reader_kwargs: dict[str, Any] | None = None):
    """Create a reader instance with provided files loaded."""
    from satpy._config import config_search_paths
    from satpy.readers.core.loading import load_reader

    if reader_kwargs is None:
        reader_kwargs = {}

    reader_configs = config_search_paths("readers/omps_edr.yaml")
    reader = load_reader(reader_configs, **reader_kwargs)
    loadable_files = reader.select_files_from_pathnames(file_paths)
    reader.create_filehandlers(loadable_files, fh_kwargs=reader_kwargs)
    return reader


def test_available_datasets(tmp_path):
    """Test available datasets dynamically generated from file contents."""
    one_file = create_v8toz_file(tmp_path, START_TIME1)
    reader = omps_reader_gen([one_file])
    # make sure we have some files
    avail_datasets = list(data_id["name"] for data_id in reader.available_dataset_ids)
    assert "Reflectivity331" in avail_datasets
    assert "AerosolIndex" in avail_datasets
    assert "ColumnAmountO3" in avail_datasets


@pytest.mark.parametrize(
    "vars_to_load",
    [
        ["Reflectivity331", "AerosolIndex", "ColumnAmountO3"],
        ["Reflectivity331"],
    ],
)
@pytest.mark.parametrize(
    "filter_by_error_flag",
    [
        None,
        [0, 1],
        [0, 1, 2, 3],
    ],
)
def test_basic_load(tmp_path, vars_to_load, filter_by_error_flag):
    """Test basic load from multiple files."""
    one_file = create_v8toz_file(tmp_path, START_TIME1)
    two_file = create_v8toz_file(tmp_path, START_TIME1 + datetime.timedelta(seconds=90))
    reader = omps_reader_gen([one_file, two_file], reader_kwargs={"filter_by_error_flag": filter_by_error_flag})
    loaded_dict = reader.load(vars_to_load)
    assert len(loaded_dict) == len(vars_to_load)

    for var_name in vars_to_load:
        _check_expected_array(loaded_dict[var_name], num_granules=2, filter_by_error_flag=filter_by_error_flag)


def _check_expected_array(
    data_arr: xr.DataArray, num_granules: int = 2, filter_by_error_flag: None | Iterable[int] = None
) -> None:
    from pyresample.geometry import SwathDefinition

    assert data_arr.dims == ("y", "x")
    assert data_arr.shape == (SINGLE_GRAN_SHAPE[0] * num_granules, SINGLE_GRAN_SHAPE[1])
    assert data_arr.dtype.type == np.float32
    assert "units" in data_arr.attrs

    data_np = data_arr.data.compute()
    assert data_np.dtype == data_arr.dtype
    if filter_by_error_flag is None:
        assert not np.isnan(data_np).any()
    else:
        assert not np.isnan(data_np[1:SINGLE_GRAN_SHAPE[0]]).any()
        assert not np.isnan(data_np[0, 10:]).any()
        for filt_val in range(10):
            if filt_val in filter_by_error_flag:
                assert not np.isnan(data_np[0, filt_val])
            else:
                assert np.isnan(data_np[0, filt_val])

    area = data_arr.attrs["area"]
    assert isinstance(area, SwathDefinition)
    assert area.shape == (SINGLE_GRAN_SHAPE[0] * num_granules, SINGLE_GRAN_SHAPE[1])
