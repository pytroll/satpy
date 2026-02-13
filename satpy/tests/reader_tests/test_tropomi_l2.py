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

"""Module for testing the satpy.readers.tropomi_l2 module."""

import datetime as dt
import os
from pathlib import Path
from typing import final

import numpy as np
import pytest
import xarray as xr

DEFAULT_FILE_DTYPE = np.int16
DEFAULT_FILE_SHAPE = (3246, 450)
DEFAULT_FILE_DATA = np.arange(DEFAULT_FILE_SHAPE[0] * DEFAULT_FILE_SHAPE[1],
                              dtype=DEFAULT_FILE_DTYPE).reshape(DEFAULT_FILE_SHAPE)
DEFAULT_BOUND_DATA = np.arange(DEFAULT_FILE_SHAPE[0] * DEFAULT_FILE_SHAPE[1] * 4,
                               dtype=DEFAULT_FILE_DTYPE).reshape(DEFAULT_FILE_SHAPE+(4,))


def tropomi_base_data() -> xr.DataTree:
    """Create the base datatree for tropomi data."""
    dt_s = dt.datetime(2018, 7, 9, 17, 3, 34)
    dt_e = dt.datetime(2018, 7, 9, 18, 45, 4)
    ds = xr.DataTree()
    ds.attrs["time_coverage_start"] = (dt_s+dt.timedelta(minutes=22)).strftime("%Y-%m-%dT%H:%M:%SZ")
    ds.attrs["time_coverage_end"] = (dt_e-dt.timedelta(minutes=22)).strftime("%Y-%m-%dT%H:%M:%SZ")
    ds.attrs["platform_shortname"] = "S5P"
    ds.attrs["sensor"] = "TROPOMI"

    ds["/PRODUCT/latitude"] = xr.DataArray(DEFAULT_FILE_DATA, dims=("scanline", "ground_pixel"))
    ds["/PRODUCT/latitude"].attrs["_FillValue"] = -999.0

    ds["/PRODUCT/longitude"] = xr.DataArray(DEFAULT_FILE_DATA, dims=("scanline", "ground_pixel"))
    ds["/PRODUCT/longitude"].attrs["_FillValue"] = -999.0

    ds["/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds"] = xr.DataArray(DEFAULT_BOUND_DATA,
                                                                            dims=("scanline", "ground_pixel", "corner"))
    ds["/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds"].attrs["_FillValue"] = -999.0
    ds["/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds"] = xr.DataArray(DEFAULT_BOUND_DATA,
                                                                             dims=("scanline", "ground_pixel",
                                                                                   "corner"))
    ds["/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds"].attrs["_FillValue"] = -999.0

    return ds


@pytest.fixture(scope="module")
def tropomi_no2_file(module_tmp_path: Path) -> Path:
    """Create a NO2 tropomi file."""
    fn = module_tmp_path / "S5P_OFFL_L2__NO2____20180709T170334_20180709T184504_03821_01_010002_20180715T184729.nc"
    ds = tropomi_base_data()
    ds["/PRODUCT/nitrogen_dioxide_total_column"] = xr.DataArray(DEFAULT_FILE_DATA, dims=("scanline", "ground_pixel"))
    ds["/PRODUCT/nitrogen_dioxide_total_column"].attrs["_FillValue"] = -999.0

    ds.to_netcdf(fn)
    return fn


@pytest.fixture(scope="module")
def tropomi_so2_file(module_tmp_path: Path) -> Path:
    """Create a SO2 tropomi file."""
    fn = module_tmp_path / "S5P_OFFL_L2__SO2____20180709T170334_20180709T184504_03821_01_010002_20180715T184729.nc"
    ds = tropomi_base_data()
    ds["/PRODUCT/sulfurdioxide_total_vertical_column"] = xr.DataArray(DEFAULT_FILE_DATA,
                                                                      dims=("scanline", "ground_pixel"))
    ds["/PRODUCT/sulfurdioxide_total_vertical_column"].attrs["_FillValue"] = -999.0

    ds.to_netcdf(fn)
    return fn


@final
class TestTROPOMIL2Reader:
    """Test TROPOMI L2 Reader."""

    yaml_file = "tropomi_l2.yaml"

    def setup_method(self):
        """Fetch reader configs."""
        from satpy._config import config_search_paths
        self.reader_configs = config_search_paths(os.path.join("readers", self.yaml_file))

    def test_init(self, tropomi_no2_file):
        """Test basic initialization of this reader."""
        from satpy.readers.core.loading import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([tropomi_no2_file])
        assert len(loadables) == 1
        r.create_filehandlers(loadables)
        # make sure we have some files
        assert r.file_handlers

    def test_load_no2(self, tropomi_no2_file: Path):
        """Load NO2 dataset."""
        from satpy.readers.core.loading import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([tropomi_no2_file])
        r.create_filehandlers(loadables)
        ds = r.load(["nitrogen_dioxide_total_column"])
        assert len(ds) == 1
        for d in ds.values():
            assert d.attrs["platform_shortname"] == "S5P"
            assert d.attrs["sensor"] == "tropomi"
            assert d.attrs["time_coverage_start"] == dt.datetime(2018, 7, 9, 17, 25, 34)
            assert d.attrs["time_coverage_end"] == dt.datetime(2018, 7, 9, 18, 23, 4)
            assert "area" in d.attrs
            assert d.attrs["area"] is not None
            assert "y" in d.dims
            assert "x" in d.dims

    def test_load_so2(self, tropomi_so2_file: Path):
        """Load SO2 dataset."""
        from satpy.readers.core.loading import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([tropomi_so2_file])
        r.create_filehandlers(loadables)
        ds = r.load(["sulfurdioxide_total_vertical_column"])
        assert len(ds) == 1
        for d in ds.values():
            assert d.attrs["platform_shortname"] == "S5P"
            assert "area" in d.attrs
            assert d.attrs["area"] is not None
            assert "y" in d.dims
            assert "x" in d.dims

    def test_load_bounds(self, tropomi_no2_file: Path):
        """Load bounds dataset."""
        from satpy.readers.core.loading import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([tropomi_no2_file])
        r.create_filehandlers(loadables)
        keys = ["latitude_bounds", "longitude_bounds"]
        ds = r.load(keys)
        assert len(ds) == 2
        for key in keys:
            assert ds[key].attrs["platform_shortname"] == "S5P"
            assert "y" in ds[key].dims
            assert "x" in ds[key].dims
            assert "corner" in ds[key].dims
            # check assembled bounds
            left = np.vstack([ds[key][:, :, 0], ds[key][-1:, :, 3]])
            right = np.vstack([ds[key][:, -1:, 1], ds[key][-1:, -1:, 2]])
            dest = np.hstack([left, right])
            dest = xr.DataArray(dest, dims=("y", "x"))
            dest.attrs = ds[key].attrs
            assert dest.attrs["platform_shortname"] == "S5P"
            assert "y" in dest.dims
            assert "x" in dest.dims
            assert DEFAULT_FILE_SHAPE[0] + 1 == dest.shape[0]
            assert DEFAULT_FILE_SHAPE[1] + 1 == dest.shape[1]
            np.testing.assert_array_equal(dest[:-1, :-1], ds[key][:, :, 0])
            np.testing.assert_array_equal(dest[-1, :-1], ds[key][-1, :, 3])
            np.testing.assert_array_equal(dest[:, -1], np.append(ds[key][:, -1, 1], ds[key][-1:, -1:, 2]))
