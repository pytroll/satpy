#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Satpy developers
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
"""Module for testing the satpy.readers.smos_l2_wind module."""

import os
from datetime import datetime

import numpy as np
import pytest
import xarray as xr


@pytest.fixture(scope="module")
def smos_l2_file(module_tmp_path):
    """Create a real dummy file for testing."""
    filename = module_tmp_path / "SM_OPER_MIR_SCNFSW_20200420T021649_20200420T035013_110_001_7.nc"
    file_content = xr.DataTree()


    dt_s = datetime(2020, 4, 22, 12, 0, 0)
    dt_e = datetime(2020, 4, 22, 12, 0, 0)

    file_content.attrs["time_coverage_start"] = dt_s.strftime("%Y-%m-%dT%H:%M:%S Z")
    file_content.attrs["time_coverage_end"] = dt_e.strftime("%Y-%m-%dT%H:%M:%S Z")
    file_content.attrs["platform_shortname"] = "SM"
    file_content.attrs["platform"] = "SMOS"
    file_content.attrs["instrument"] = "MIRAS"
    file_content.attrs["processing_level"] = "L2"
    file_content.attrs["geospatial_bounds_vertical_crs"] = "EPSG:4623"

    file_content["lat"] = xr.DataArray(np.arange(-90., 90.25, 0.25), dims=("lat"))
    file_content["lat"].attrs["_FillValue"] = -999.0

    file_content["lon"] = xr.DataArray(np.arange(0., 360., 0.25), dims=("lon"))
    file_content["lon"].attrs["_FillValue"] = -999.0

    file_content["wind_speed"] = xr.DataArray(np.ndarray(shape=(1,  # Time dimension
                                                         len(file_content["lat"]),
                                                         len(file_content["lon"]))),
                                              dims=("time", "lat", "lon"),
                                              coords=[[1], file_content["lat"], file_content["lon"]])
    file_content["wind_speed"].attrs["_FillValue"] = -999.0
    file_content.to_netcdf(filename)

    return filename


class TestSMOSL2WINDReader:
    """Test SMOS L2 WINDReader."""

    yaml_file = "smos_l2_wind.yaml"

    def setup_method(self):
        """Wrap NetCDF4 file handler with our own fake handler."""
        from satpy._config import config_search_paths
        self.reader_configs = config_search_paths(os.path.join("readers", self.yaml_file))

    def test_init(self, smos_l2_file):
        """Test basic initialization of this reader."""
        from satpy.readers.core.loading import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([smos_l2_file])
        assert len(loadables) == 1
        r.create_filehandlers(loadables)
        # make sure we have some files
        assert r.file_handlers

    def test_load_wind_speed(self, smos_l2_file):
        """Load wind_speed dataset."""
        from satpy.readers.core.loading import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([smos_l2_file])
        r.create_filehandlers(loadables)
        ds = r.load(["wind_speed"])
        assert len(ds) == 1
        for d in ds.values():
            assert d.attrs["platform_shortname"] == "SM"
            assert d.attrs["sensor"] == "MIRAS"
            assert "area" in d.attrs
            assert d.attrs["area"] is not None
            assert "y" in d.dims
            assert "x" in d.dims
            assert d.shape == (719, 1440)
            assert d.y[0].data == -89.75
            assert d.y[d.shape[0] - 1].data == 89.75

    def test_load_lat(self, smos_l2_file):
        """Load lat dataset."""
        from satpy.readers.core.loading import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([smos_l2_file])
        r.create_filehandlers(loadables)
        ds = r.load(["lat"])
        assert len(ds) == 1
        for d in ds.values():
            assert "y" in d.dims
            assert d.shape == (719,)
            assert d.data[0] == -89.75
            assert d.data[d.shape[0] - 1] == 89.75

    def test_load_lon(self, smos_l2_file):
        """Load lon dataset."""
        from satpy.readers.core.loading import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([smos_l2_file])
        r.create_filehandlers(loadables)
        ds = r.load(["lon"])
        assert len(ds) == 1
        for d in ds.values():
            assert "x" in d.dims
            assert d.shape == (1440,)
            assert d.data[0] == -180.0
            assert d.data[d.shape[0] - 1] == 179.75

    def test_adjust_lon(self, smos_l2_file):
        """Load adjust longitude dataset."""
        from xarray import DataArray

        from satpy.readers.smos_l2_wind import SMOSL2WINDFileHandler
        smos_l2_wind_fh = SMOSL2WINDFileHandler(smos_l2_file,
                                                {}, filetype_info={"file_type": "smos_l2_wind"})
        lon = DataArray(np.arange(0., 360., 0.25), dims=("lon"))
        data = DataArray(np.empty_like(lon.data), dims=("lon"), coords=dict(lon=lon))
        adjusted = smos_l2_wind_fh._normalize_lon_coord(data)
        expected = np.concatenate((np.arange(0, 180., 0.25),
                                   np.arange(-180.0, 0, 0.25)))
        assert adjusted.lon.data.tolist() == expected.tolist()

    def test_roll_dataset(self, smos_l2_file):
        """Load roll of dataset along the lon coordinate."""
        from xarray import DataArray

        from satpy.readers.smos_l2_wind import SMOSL2WINDFileHandler
        smos_l2_wind_fh = SMOSL2WINDFileHandler(smos_l2_file,
                                                {}, filetype_info={"file_type": "smos_l2_wind"})
        lon = DataArray(np.arange(0., 360., 0.25), dims=("lon"))
        data = DataArray(np.empty_like(lon.data), dims=("lon"), coords=dict(lon=lon))
        data = smos_l2_wind_fh._normalize_lon_coord(data)
        adjusted = smos_l2_wind_fh._roll_dataset_lon_coord(data)
        expected = np.arange(-180., 180., 0.25)
        assert adjusted.lon.data.tolist() == expected.tolist()
