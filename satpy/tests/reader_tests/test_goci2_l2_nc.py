#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016-2018 Satpy developers
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

"""Module for testing the satpy.readers.goci2_l2_nc module."""

import datetime as dt

import numpy as np
import pytest
import xarray as xr
from pytest_lazy_fixtures import lf as lazy_fixture

from satpy import Scene
from satpy.tests.utils import RANDOM_GEN

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - tmp_path_factory


start_time = dt.datetime(2024, 2, 14, 2, 32, 27)
end_time = dt.datetime(2024, 2, 14, 2, 33, 31)

global_attrs = {
    "observation_start_time": start_time.strftime("%Y%m%d_%H%M%S"),
    "observation_end_time": end_time.strftime("%Y%m%d_%H%M%S"),
    "instrument": "GOCI-II",
    "platform": "GK-2B",
}

badarea_attrs = global_attrs.copy()
badarea_attrs["cdm_data_type"] = "bad_area"


def _create_lonlat():
    """Create a fake navigation dataset with lon/lat."""
    lon, lat = np.meshgrid(np.linspace(120, 130, 10), np.linspace(30, 40, 10))
    lon = xr.DataArray(
        lon,
        dims=("number_of_lines", "pixels_per_line"),
        attrs={"standard_name": "longitude", "units": "degrees_east"},
    )
    lat = xr.DataArray(
        lat,
        dims=("number_of_lines", "pixels_per_line"),
        attrs={"standard_name": "latitude", "units": "degrees_north"},
    )
    ds = xr.Dataset()
    ds["longitude"] = lon
    ds["latitude"] = lat
    return ds


def _create_bad_lon_lat():
    """Create a fake navigation dataset with lon/lat base name missing."""
    lon, lat = np.meshgrid(np.linspace(120, 130, 10), np.linspace(30, 40, 10))
    ds = xr.Dataset(
        {
            "longitude": (["number_of_lines", "pixels_per_line"], lon),
            "latitude": (["number_of_lines", "pixels_per_line"], lat),
        }
    )
    return ds


@pytest.fixture(scope="session")
def ac_file(tmp_path_factory):
    """Create a fake atmospheric correction product."""
    data = RANDOM_GEN.random((10, 10))
    RhoC = xr.Dataset(
        {"RhoC_555": (["number_of_lines", "pixels_per_line"], data)},
        coords={"number_of_lines": np.arange(10), "pixels_per_line": np.arange(10)},
    )
    Rrs = xr.Dataset(
        {"Rrs_555": (["number_of_lines", "pixels_per_line"], data)},
        coords={"number_of_lines": np.arange(10), "pixels_per_line": np.arange(10)},
    )
    navigation = _create_lonlat()
    ds = xr.Dataset(attrs=global_attrs)
    fname = (
        f'{tmp_path_factory.mktemp("data")}/GK2B_GOCI2_L2_20240214_021530_LA_S010_AC.nc'
    )
    ds.to_netcdf(fname)
    navigation.to_netcdf(fname, group="navigation_data", mode="a")
    RhoC.to_netcdf(fname, group="geophysical_data/RhoC", mode="a")
    Rrs.to_netcdf(fname, group="geophysical_data/Rrs", mode="a")
    return fname


@pytest.fixture(scope="module")
def iop_file(tmp_path_factory):
    """Create a fake IOP product."""
    data = RANDOM_GEN.random((10, 10))
    a = xr.Dataset(
        {"a_total_555": (["number_of_lines", "pixels_per_line"], data)},
        coords={"number_of_lines": np.arange(10), "pixels_per_line": np.arange(10)},
    )
    bb = xr.Dataset(
        {"bb_total_555": (["number_of_lines", "pixels_per_line"], data)},
        coords={"number_of_lines": np.arange(10), "pixels_per_line": np.arange(10)},
    )
    navigation = _create_lonlat()
    ds = xr.Dataset(attrs=global_attrs)
    fname = f'{tmp_path_factory.mktemp("data")}/GK2B_GOCI2_L2_20240214_021530_LA_S010_IOP.nc'
    ds.to_netcdf(fname)
    navigation.to_netcdf(fname, group="navigation_data", mode="a")
    a.to_netcdf(fname, group="geophysical_data/a_total", mode="a")
    bb.to_netcdf(fname, group="geophysical_data/bb_total", mode="a")
    return fname


@pytest.fixture(scope="module")
def generic_file(tmp_path_factory):
    """Create a fake ouput product like Chl, Zsd etc."""
    data = RANDOM_GEN.random((10, 10))
    geophysical_data = xr.Dataset(
        {"Chl": (["number_of_lines", "pixels_per_line"], data)},
        coords={"number_of_lines": np.arange(10), "pixels_per_line": np.arange(10)},
    )
    navigation = _create_lonlat()
    ds = xr.Dataset(attrs=global_attrs)
    fname = f'{tmp_path_factory.mktemp("data")}/GK2B_GOCI2_L2_20240214_021530_LA_S010_Chl.nc'
    ds.to_netcdf(fname)
    navigation.to_netcdf(fname, group="navigation_data", mode="a")
    geophysical_data.to_netcdf(fname, group="geophysical_data", mode="a")
    return fname


@pytest.fixture(scope="module")
def generic_bad_file(tmp_path_factory):
    """Create a PP product with lon/lat base name missing."""
    data = RANDOM_GEN.random((10, 10))
    geophysical_data = xr.Dataset(
        {"PP": (["number_of_lines", "pixels_per_line"], data)},
        coords={"number_of_lines": np.arange(10), "pixels_per_line": np.arange(10)},
    )
    navigation = _create_bad_lon_lat()
    ds = xr.Dataset(attrs=global_attrs)
    fname = (
        f'{tmp_path_factory.mktemp("data")}/GK2B_GOCI2_L2_20240214_021530_LA_S010_PP.nc'
    )
    ds.to_netcdf(fname)
    navigation.to_netcdf(fname, group="navigation_data", mode="a")
    geophysical_data.to_netcdf(fname, group="geophysical_data", mode="a")
    return fname


class TestGOCI2Reader:
    """Test the GOCI-II L2 netcdf file reader."""

    @pytest.mark.parametrize(
        "test_files",
        [
            lazy_fixture("ac_file"),
            lazy_fixture("iop_file"),
            lazy_fixture("generic_file"),
            lazy_fixture("generic_bad_file"),
        ],
    )
    def test_scene_available_datasets(self, test_files):
        """Test that datasets are available."""
        scene = Scene(filenames=[test_files], reader="goci2_l2_nc")
        available_datasets = scene.all_dataset_names()
        assert len(available_datasets) > 0
        assert "longitude" in available_datasets
        assert "latitude" in available_datasets

    @pytest.mark.parametrize(
        "test_files",
        [
            lazy_fixture("ac_file"),
            lazy_fixture("iop_file"),
            lazy_fixture("generic_file"),
            lazy_fixture("generic_bad_file"),
        ],
    )
    def test_start_end_time(self, test_files):
        """Test dataset start_time and end_time."""
        scene = Scene(filenames=[test_files], reader="goci2_l2_nc")
        assert scene.start_time == start_time
        assert scene.end_time == end_time

    @pytest.mark.parametrize(
        ("test_files", "datasets"),
        [
            (lazy_fixture("ac_file"), ["RhoC_555", "Rrs_555"]),
            (lazy_fixture("iop_file"), ["a_total_555", "bb_total_555"]),
            (lazy_fixture("generic_file"), ["Chl"]),
            (lazy_fixture("generic_bad_file"), ["PP"]),
        ],
    )
    def test_load_dataset(self, test_files, datasets):
        """Test dataset loading."""
        scene = Scene(filenames=[test_files], reader="goci2_l2_nc")
        scene.load(datasets)
        for dataset in datasets:
            data_arr = scene[dataset]
            assert data_arr.dims == ("y", "x")
