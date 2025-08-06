#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 Satpy developers
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
"""Module for testing the satpy.readers.acspo module."""

from __future__ import annotations

import datetime as dt
import os
from typing import Any
from unittest import mock

import numpy as np
import pytest

from satpy.tests.reader_tests.test_netcdf_utils import FakeNetCDF4FileHandler
from satpy.tests.utils import convert_file_content_to_data_array

DEFAULT_FILE_SHAPE = (10, 300)


class FakeNetCDF4FileHandler2(FakeNetCDF4FileHandler):
    """Swap-in NetCDF4 File Handler."""

    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content."""
        DEFAULT_FILE_DTYPE = np.uint16
        DEFAULT_FILE_DATA = np.arange(DEFAULT_FILE_SHAPE[0] * DEFAULT_FILE_SHAPE[1],
                                      dtype=DEFAULT_FILE_DTYPE).reshape(DEFAULT_FILE_SHAPE)
        DEFAULT_LAT_DATA = np.linspace(45, 65, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
        DEFAULT_LAT_DATA = np.repeat([DEFAULT_LAT_DATA], DEFAULT_FILE_SHAPE[0], axis=0)
        DEFAULT_LON_DATA = np.linspace(5, 45, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
        DEFAULT_LON_DATA = np.repeat([DEFAULT_LON_DATA], DEFAULT_FILE_SHAPE[0], axis=0)

        date = filename_info.get("start_time", dt.datetime(2016, 1, 1, 12, 0, 0))
        sat, inst = {
            "VIIRS_NPP": ("NPP", "VIIRS"),
            "VIIRS_N20": ("N20", "VIIRS"),
        }[filename_info["sensor_id"]]

        file_content: dict[str, Any] = {
            "/attr/platform": sat,
            "/attr/sensor": inst,
            "/attr/spatial_resolution": "742 m at nadir",
            "/attr/time_coverage_start": date.strftime("%Y%m%dT%H%M%SZ"),
            "/attr/time_coverage_end": (date + dt.timedelta(minutes=6)).strftime("%Y%m%dT%H%M%SZ"),
        }

        file_content["lat"] = DEFAULT_LAT_DATA
        file_content["lat/attr/comment"] = "Latitude of retrievals"
        file_content["lat/attr/long_name"] = "latitude"
        file_content["lat/attr/standard_name"] = "latitude"
        file_content["lat/attr/units"] = "degrees_north"
        file_content["lat/attr/valid_min"] = -90.
        file_content["lat/attr/valid_max"] = 90.
        file_content["lat/shape"] = DEFAULT_FILE_SHAPE

        file_content["lon"] = DEFAULT_LON_DATA
        file_content["lon/attr/comment"] = "Longitude of retrievals"
        file_content["lon/attr/long_name"] = "longitude"
        file_content["lon/attr/standard_name"] = "longitude"
        file_content["lon/attr/units"] = "degrees_east"
        file_content["lon/attr/valid_min"] = -180.
        file_content["lon/attr/valid_max"] = 180.
        file_content["lon/shape"] = DEFAULT_FILE_SHAPE

        for k in ["sea_surface_temperature",
                  "satellite_zenith_angle",
                  "sea_ice_fraction",
                  "wind_speed"]:
            file_content[k] = DEFAULT_FILE_DATA[None, ...]
            file_content[k + "/attr/scale_factor"] = 1.1
            file_content[k + "/attr/add_offset"] = 0.1
            file_content[k + "/attr/units"] = "some_units"
            file_content[k + "/attr/comment"] = "comment"
            file_content[k + "/attr/standard_name"] = "standard_name"
            file_content[k + "/attr/long_name"] = "long_name"
            file_content[k + "/attr/valid_min"] = 0
            file_content[k + "/attr/valid_max"] = 65534
            file_content[k + "/attr/_FillValue"] = 65534
            file_content[k + "/shape"] = (1, DEFAULT_FILE_SHAPE[0], DEFAULT_FILE_SHAPE[1])

        l2p_flags_data = np.zeros(
            (1, DEFAULT_FILE_SHAPE[0], DEFAULT_FILE_SHAPE[1]),
            dtype=np.int16)
        l2p_flags_data[0, -1, -1] = np.int16(np.uint16(0b1100000000000000))  # cloud
        file_content["l2p_flags"] = l2p_flags_data

        convert_file_content_to_data_array(file_content, dims=("time", "nj", "ni"))
        return file_content


class TestACSPOReader:
    """Test ACSPO Reader."""

    yaml_file = "acspo.yaml"

    def setup_method(self):
        """Wrap NetCDF4 file handler with our own fake handler."""
        from satpy._config import config_search_paths
        from satpy.readers.acspo import ACSPOFileHandler
        self.reader_configs = config_search_paths(os.path.join("readers", self.yaml_file))
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.p = mock.patch.object(ACSPOFileHandler, "__bases__", (FakeNetCDF4FileHandler2,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

    def teardown_method(self):
        """Stop wrapping the NetCDF4 file handler."""
        self.p.stop()

    @pytest.mark.parametrize(
        "filename",
        [
            ("20170401174600-STAR-L2P_GHRSST-SSTskin-VIIRS_NPP-ACSPO_V2.40-v02.0-fv01.0.nc"),
            ("20210916161708-STAR-L2P_GHRSST-SSTsubskin-VIIRS_N20-ACSPO_V2.80-v02.0-fv01.0.nc"),
        ]
    )
    def test_init(self, filename):
        """Test basic init with no extra parameters."""
        from satpy.readers.core.loading import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([filename])
        assert len(loadables) == 1
        r.create_filehandlers(loadables)
        # make sure we have some files
        assert r.file_handlers

    @pytest.mark.parametrize(
        ("var_name", "cloud_clearable", "cloud_clear"),
        [
            ("sst", True, False),
            ("sst", True, True),
            ("satellite_zenith_angle", False, False),
            ("satellite_zenith_angle", False, True),
            ("sea_ice_fraction", False, False),
            ("wind_speed", False, False),
        ])
    def test_load_every_dataset(self, var_name, cloud_clearable, cloud_clear):
        """Test loading all datasets."""
        from satpy.readers.core.loading import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            "20170401174600-STAR-L2P_GHRSST-SSTskin-VIIRS_NPP-ACSPO_V2.40-v02.0-fv01.0.nc",
        ])
        fh_kwargs = {}
        if cloud_clear:
            fh_kwargs["filters"] = ["cloud_clear"]
        r.create_filehandlers(loadables, fh_kwargs=fh_kwargs)
        datasets = r.load([var_name])
        assert len(datasets) == 1
        d = datasets[var_name]
        assert d.shape == DEFAULT_FILE_SHAPE
        assert d.dims == ("y", "x")
        assert d.attrs["sensor"] == "viirs"
        assert d.attrs["rows_per_scan"] == 16
        dask_data = d.data
        np_data = dask_data.compute()
        assert dask_data.dtype == np.float32
        assert np_data.dtype == np.float32

        exp_data = np.arange(DEFAULT_FILE_SHAPE[0] * DEFAULT_FILE_SHAPE[1], dtype=np.uint16).reshape(DEFAULT_FILE_SHAPE)
        exp_data = exp_data * np.float32(1.1) + 0.1
        if cloud_clearable and cloud_clear:
            exp_data[-1, -1] = np.nan
        np.testing.assert_array_equal(np_data, exp_data, strict=True)
