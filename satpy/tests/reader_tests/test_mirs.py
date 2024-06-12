#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Satpy developers
#
# This file is part of Satpy.
#
# Satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# Satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# Satpy.  If not, see <http://www.gnu.org/licenses/>.

"""Module for testing the satpy.readers.mirs module."""

from __future__ import annotations

import datetime as dt
import os
from unittest import mock

import numpy as np
import pytest
import xarray as xr

from satpy._config import config_search_paths
from satpy.dataset import DataID
from satpy.readers import load_reader
from satpy.readers.yaml_reader import FileYAMLReader
from satpy.tests.utils import RANDOM_GEN

METOP_FILE = "IMG_SX.M2.D17037.S1601.E1607.B0000001.WE.HR.ORB.nc"
NPP_MIRS_L2_SWATH = "NPR-MIRS-IMG_v11r6_npp_s201702061601000_e201702061607000_c202012201658410.nc"
N20_MIRS_L2_SWATH = "NPR-MIRS-IMG_v11r4_n20_s201702061601000_e201702061607000_c202012201658410.nc"
N21_MIRS_L2_SWATH = "NPR-MIRS-IMG_v11r4_n21_s201702061601000_e201702061607000_c202012201658410.nc"
OTHER_MIRS_L2_SWATH = "NPR-MIRS-IMG_v11r4_gpm_s201702061601000_e201702061607000_c202010080001310.nc"

EXAMPLE_FILES = [METOP_FILE, NPP_MIRS_L2_SWATH, OTHER_MIRS_L2_SWATH]

N_CHANNEL = 22
N_FOV = 96
N_SCANLINE = 100
DEFAULT_FILE_DTYPE = np.float32
DEFAULT_2D_SHAPE = (N_SCANLINE, N_FOV)
DEFAULT_DATE = dt.datetime(2019, 6, 19, 13, 0)
DEFAULT_LAT = np.linspace(23.09356, 36.42844, N_SCANLINE * N_FOV,
                          dtype=DEFAULT_FILE_DTYPE)
DEFAULT_LON = np.linspace(127.6879, 144.5284, N_SCANLINE * N_FOV,
                          dtype=DEFAULT_FILE_DTYPE)
FREQ = xr.DataArray(
    np.array([23.8, 31.4, 50.3, 51.76, 52.8, 53.596, 54.4, 54.94, 55.5,
              57.29, 57.29, 57.29, 57.29, 57.29, 57.29, 88.2, 165.5,
              183.31, 183.31, 183.31, 183.31, 183.31][:N_CHANNEL], dtype=np.float32),
    dims="Channel",
    attrs={"description": "Central Frequencies (GHz)"},
)
POLO = xr.DataArray(
    np.array([2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3][:N_CHANNEL], dtype=np.int16),
    dims="Channel",
    attrs={"description": "Polarizations"},
)

DS_IDS = ["RR", "longitude", "latitude"]
TEST_VARS = ["btemp_88v", "btemp_165h",
             "btemp_23v", "RR", "Sfc_type"]
DEFAULT_UNITS = {"btemp_88v": "K", "btemp_165h": "K",
                 "btemp_23v": "K", "RR": "mm/hr", "Sfc_type": "1"}
PLATFORM = {"M2": "metop-a", "NPP": "npp", "GPM": "gpm"}
SENSOR = {"m2": "amsu-mhs", "npp": "atms", "gpm": "GPI"}

START_TIME = dt.datetime(2017, 2, 6, 16, 1, 0)
END_TIME = dt.datetime(2017, 2, 6, 16, 7, 0)


def fake_coeff_from_fn(fn):
    """Create Fake Coefficients."""
    ameans = RANDOM_GEN.uniform(261, 267, N_CHANNEL)
    locations = [
        [1, 2],
        [1, 2],
        [3, 4, 5],
        [3, 4, 5],
        [4, 5, 6],
        [5, 6, 7],
        [6, 7, 8],
        [7, 8],
        [9, 10, 11],
        [10, 11],
        [10, 11, 12],
        [11, 12, 13],
        [12, 13],
        [12, 13, 14],
        [14, 15],
        [1, 16],
        [17, 18],
        [18, 19],
        [18, 19, 20],
        [19, 20, 21],
        [20, 21, 22],
        [21, 22],
    ]
    all_nchx = [len(loc) for loc in locations]

    coeff_str = []
    for idx in range(1, N_CHANNEL + 1):
        nx = idx - 1
        coeff_str.append("\n")
        next_line = "   {}  {} {}\n".format(idx, all_nchx[nx], ameans[nx])
        coeff_str.append(next_line)
        next_line = "   {}\n".format("   ".join([str(x) for x in locations[idx - 1]]))
        coeff_str.append(next_line)
        for fov in range(1, N_FOV+1):
            random_coeff = np.ones(all_nchx[nx])
            str_coeff = "  ".join([str(x) for x in random_coeff])
            random_means = np.zeros(all_nchx[nx])
            str_means = " ".join([str(x) for x in random_means])
            error_val = RANDOM_GEN.uniform(0, 4)
            coeffs_line = " {:>2} {:>2}  {} {}   {}\n".format(idx, fov,
                                                              str_coeff,
                                                              str_means,
                                                              error_val)
            coeff_str.append(coeffs_line)

    return coeff_str


def _get_datasets_with_attributes(**kwargs):
    """Represent files with two resolution of variables in them (ex. OCEAN)."""
    bt = xr.DataArray(np.linspace(1830, 3930, N_SCANLINE * N_FOV * N_CHANNEL, dtype=np.int16).
                      reshape(N_SCANLINE, N_FOV, N_CHANNEL),
                      attrs={"long_name": "Channel Temperature (K)",
                             "units": "Kelvin",
                             "coordinates": "Longitude Latitude Freq",
                             "scale_factor": 0.01,
                             "_FillValue": -999,
                             "valid_range": [0, 50000]},
                      dims=("Scanline", "Field_of_view", "Channel"))
    rr = xr.DataArray(RANDOM_GEN.integers(100, 500, size=(N_SCANLINE, N_FOV), dtype=np.int16),
                      attrs={"long_name": "Rain Rate (mm/hr)",
                             "units": "mm/hr",
                             "coordinates": "Longitude Latitude",
                             "scale_factor": 0.1,
                             "_FillValue": -999,
                             "valid_range": [0, 1000]},
                      dims=("Scanline", "Field_of_view"))
    sfc_type = xr.DataArray(RANDOM_GEN.integers(0, 4, size=(N_SCANLINE, N_FOV), dtype=np.int16),
                            attrs={"description": "type of surface:0-ocean," +
                                                  "1-sea ice,2-land,3-snow",
                                   "units": "1",
                                   "coordinates": "Longitude Latitude",
                                   "_FillValue": -999,
                                   "valid_range": [0, 3]
                                   },
                            dims=("Scanline", "Field_of_view"))
    latitude = xr.DataArray(DEFAULT_LAT.reshape(DEFAULT_2D_SHAPE),
                            attrs={"long_name":
                                   "Latitude of the view (-90,90)"},
                            dims=("Scanline", "Field_of_view"))
    longitude = xr.DataArray(DEFAULT_LON.reshape(DEFAULT_2D_SHAPE),
                             attrs={"long_name":
                                    "Longitude of the view (-180,180)"},
                             dims=("Scanline", "Field_of_view"))

    ds_vars = {
        "Freq": FREQ,
        "Polo": POLO,
        "BT": bt,
        "RR": rr,
        "Sfc_type": sfc_type,
        "Latitude": latitude,
        "Longitude": longitude
    }

    attrs = {"missing_value": -999}
    ds = xr.Dataset(ds_vars, attrs=attrs)
    ds = ds.assign_coords({"Freq": FREQ, "Latitude": latitude, "Longitude": longitude})
    return ds


def _get_datasets_with_less_attributes():
    """Represent files with two resolution of variables in them (ex. OCEAN)."""
    bt = xr.DataArray(np.linspace(1830, 3930, N_SCANLINE * N_FOV * N_CHANNEL, dtype=np.int16).
                      reshape(N_SCANLINE, N_FOV, N_CHANNEL),
                      attrs={"long_name": "Channel Temperature (K)",
                             "scale_factor": 0.01},
                      dims=("Scanline", "Field_of_view", "Channel"))
    rr = xr.DataArray(RANDOM_GEN.integers(100, 500, size=(N_SCANLINE, N_FOV), dtype=np.int16),
                      attrs={"long_name": "Rain Rate (mm/hr)",
                             "scale_factor": 0.1},
                      dims=("Scanline", "Field_of_view"))

    sfc_type = xr.DataArray(RANDOM_GEN.integers(0, 4, size=(N_SCANLINE, N_FOV), dtype=np.int16),
                            attrs={"description": "type of surface:0-ocean," +
                                                  "1-sea ice,2-land,3-snow"},
                            dims=("Scanline", "Field_of_view"))
    latitude = xr.DataArray(DEFAULT_LAT.reshape(DEFAULT_2D_SHAPE),
                            attrs={"long_name":
                                   "Latitude of the view (-90,90)"},
                            dims=("Scanline", "Field_of_view"))
    longitude = xr.DataArray(DEFAULT_LON.reshape(DEFAULT_2D_SHAPE),
                             attrs={"long_name":
                                    "Longitude of the view (-180,180)"},
                             dims=("Scanline", "Field_of_view"))

    ds_vars = {
        "Freq": FREQ,
        "Polo": POLO,
        "BT": bt,
        "RR": rr,
        "Sfc_type": sfc_type,
        "Longitude": longitude,
        "Latitude": latitude
    }

    attrs = {"missing_value": -999.}
    ds = xr.Dataset(ds_vars, attrs=attrs)
    ds = ds.assign_coords({"Freq": FREQ, "Latitude": latitude, "Longitude": longitude})
    return ds


def fake_open_dataset(filename, **kwargs):
    """Create a Dataset similar to reading an actual file with xarray.open_dataset."""
    if filename == METOP_FILE:
        return _get_datasets_with_less_attributes()
    return _get_datasets_with_attributes()


@pytest.mark.parametrize(
    ("filenames", "expected_datasets"),
    [
        ([METOP_FILE], DS_IDS),
        ([NPP_MIRS_L2_SWATH], DS_IDS),
        ([OTHER_MIRS_L2_SWATH], DS_IDS),
    ]
)
def test_available_datasets(filenames, expected_datasets):
    """Test that variables are dynamically discovered."""
    r = _create_fake_reader(filenames, {})
    avails = list(r.available_dataset_names)
    for var_name in expected_datasets:
        assert var_name in avails


@pytest.mark.parametrize(
    ("filenames", "loadable_ids", "platform_name"),
    [
        ([METOP_FILE], TEST_VARS, "metop-a"),
        ([NPP_MIRS_L2_SWATH], TEST_VARS, "npp"),
        ([N20_MIRS_L2_SWATH], TEST_VARS, "noaa-20"),
        ([N21_MIRS_L2_SWATH], TEST_VARS, "noaa-21"),
        ([OTHER_MIRS_L2_SWATH], TEST_VARS, "gpm"),
    ]
)
@pytest.mark.parametrize("reader_kw", [{}, {"limb_correction": False}])
def test_basic_load(filenames, loadable_ids, platform_name, reader_kw):
    """Test that variables are loaded properly."""
    r = _create_fake_reader(filenames, reader_kw)

    test_data = fake_open_dataset(filenames[0])
    exp_limb_corr = reader_kw.get("limb_correction", True) and platform_name in ("npp", "noaa-20", "noaa-21")
    loaded_data_arrs = _load_and_check_limb_correction_variables(r, loadable_ids, platform_name, exp_limb_corr)
    for _data_id, data_arr_dask in loaded_data_arrs.items():
        data_arr = data_arr_dask.compute()
        assert data_arr.dtype == data_arr_dask.dtype
        if np.issubdtype(data_arr.dtype, np.floating):
            # we started with float32, it should stay that way
            # NOTE: Sfc_type does not have enough metadata to dynamically force integer type
            #   even though it is a mask/category product
            assert data_arr.dtype.type == np.float32
        _check_metadata(data_arr, test_data, platform_name)


def _create_fake_reader(
        filenames: list[str],
        reader_kwargs: dict,
        exp_loadable_files: int | None = None
) -> FileYAMLReader:
    exp_loadable_files = exp_loadable_files if exp_loadable_files is not None else len(filenames)
    reader_configs = config_search_paths(os.path.join("readers", "mirs.yaml"))
    with mock.patch("satpy.readers.mirs.xr.open_dataset") as od:
        od.side_effect = fake_open_dataset
        r = load_reader(reader_configs)
        loadables = r.select_files_from_pathnames(filenames)
        r.create_filehandlers(loadables, fh_kwargs=reader_kwargs)

        assert isinstance(r, FileYAMLReader)
        assert len(loadables) == exp_loadable_files
        assert r.file_handlers
    return r


def _load_and_check_limb_correction_variables(
        reader: FileYAMLReader,
        loadable_ids: list[str],
        platform_name: str,
        exp_limb_corr: bool
) -> dict[DataID, xr.DataArray]:
    with mock.patch("satpy.readers.mirs.read_atms_coeff_to_string") as \
            fd, mock.patch("satpy.readers.mirs.retrieve") as rtv:
        fd.side_effect = fake_coeff_from_fn
        loaded_data_arrs = reader.load(loadable_ids)
    if exp_limb_corr:
        fd.assert_called()
        suffix = f"noaa{platform_name[-2:]}" if platform_name.startswith("noaa") else "snpp"
        assert rtv.call_count == 2 * len([var_name for var_name in loadable_ids if "btemp" in var_name])
        for calls_args in rtv.call_args_list:
            assert calls_args[0][0].endswith(f"_{suffix}.txt")
    else:
        fd.assert_not_called()
        rtv.assert_not_called()
    assert len(loaded_data_arrs) == len(loadable_ids)
    return loaded_data_arrs


def _check_metadata(data_arr: xr.DataArray, test_data: xr.Dataset, platform_name: str) -> None:
    var_name = data_arr.attrs["name"]
    if var_name not in ["latitude", "longitude"]:
        _check_area(data_arr)
    assert "_FillValue" not in data_arr.attrs
    _check_attrs(data_arr, platform_name)

    input_fake_data = test_data["BT"] if "btemp" in var_name else test_data[var_name]
    if "valid_range" in input_fake_data.attrs:
        valid_range = input_fake_data.attrs["valid_range"]
        _check_valid_range(data_arr, valid_range)
    if "_FillValue" in input_fake_data.attrs:
        fill_value = input_fake_data.attrs["_FillValue"]
        _check_fill_value(data_arr, fill_value)

    assert data_arr.attrs["units"] == DEFAULT_UNITS[var_name]


def _check_area(data_arr):
    from pyresample.geometry import SwathDefinition
    area = data_arr.attrs["area"]
    assert isinstance(area, SwathDefinition)


def _check_valid_range(data_arr, test_valid_range):
    # valid_range is popped out of data_arr.attrs when it is applied
    assert "valid_range" not in data_arr.attrs
    assert data_arr.data.min() >= test_valid_range[0]
    assert data_arr.data.max() <= test_valid_range[1]


def _check_fill_value(data_arr, test_fill_value):
    assert "_FillValue" not in data_arr.attrs
    assert not (data_arr.data == test_fill_value).any()


def _check_attrs(data_arr, platform_name):
    attrs = data_arr.attrs
    assert "scale_factor" not in attrs
    assert "platform_name" in attrs
    assert attrs["platform_name"] == platform_name
    assert attrs["start_time"] == START_TIME
    assert attrs["end_time"] == END_TIME
