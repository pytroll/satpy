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

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

DEFAULT_FILE_DTYPE = np.float32
DEFAULT_FILE_SHAPE = (721, 1440)
DEFAULT_DATE = datetime(2019, 6, 19, 13, 0)
DEFAULT_LAT = np.linspace(-90, 90, DEFAULT_FILE_SHAPE[0], dtype=DEFAULT_FILE_DTYPE)
DEFAULT_LON = np.linspace(-180, 180, DEFAULT_FILE_SHAPE[1], dtype=DEFAULT_FILE_DTYPE)
DEFAULT_FILE_FLOAT_DATA = np.arange(DEFAULT_FILE_SHAPE[0] * DEFAULT_FILE_SHAPE[1],
                                    dtype=DEFAULT_FILE_DTYPE)
DEFAULT_FILE_DATE_DATA = np.clip(DEFAULT_FILE_FLOAT_DATA, 0, 1049)
DEFAULT_FILE_UBYTE_DATA = np.arange(DEFAULT_FILE_SHAPE[0] * DEFAULT_FILE_SHAPE[1],
                                    dtype=np.ubyte)
float_variables = ["tpwGrid", "tpwGridPrior", "tpwGridSubseq", "footGridPrior", "footGridSubseq"]
date_variables = ["timeAwayGridPrior", "timeAwayGridSubseq"]
ubyte_variables = ["satGridPrior", "satGridSubseq"]


@pytest.fixture(scope="module")
def mimic_file(module_tmp_path: Path) -> Path:
    """Mimic a real data file."""
    filename = module_tmp_path / "comp20190619.130000.nc"
    file_type = "mimicTPW2_comp"
    dt_s = DEFAULT_DATE
    dt_e = DEFAULT_DATE

    dt = xr.DataTree()
    dt.attrs["start_time"] = dt_s.strftime("%Y%m%d.%H%M%S")
    dt.attrs["end_time"] = dt_e.strftime("%Y%m%d.%H%M%S")
    dt.attrs["platform_shortname"] = "aggregated microwave"
    dt.attrs["sensor"] ="mimic"

    dt["latArr"] = xr.DataArray(DEFAULT_LAT,
                                attrs={"units": "degress_north"},
                                dims=("lat",))

    dt["lonArr"] = xr.DataArray(DEFAULT_LON,
                                attrs={"units": "degress_east"},
                                dims=("lon",))

    for float_var in float_variables:
        dt[float_var] = xr.DataArray(DEFAULT_FILE_FLOAT_DATA.reshape(DEFAULT_FILE_SHAPE),
                                     attrs={"units":"mm",
                                            "_FillValue": -999.0,
                                            "name": float_var,
                                            "file_key": float_var,
                                            "file_type": file_type},
                                     dims=("y", "x"))
    for date_var in date_variables:
        dt[date_var] = xr.DataArray(DEFAULT_FILE_DATE_DATA.reshape(DEFAULT_FILE_SHAPE),
                                    attrs=dict(units="minutes"),
                                    dims=("y", "x"))
    for ubyte_var in ubyte_variables:
        dt[ubyte_var] = xr.DataArray(DEFAULT_FILE_UBYTE_DATA.reshape(DEFAULT_FILE_SHAPE),
                                     attrs={"source_key": ("Key: 0: None, 1: NOAA-N, 2: NOAA-P, 3: Metop-A, "
                                                           "4: Metop-B, 5: SNPP, 6: SSMI-17, 7: SSMI-18"),
                                            "_FillValue": 255,
                                            "name": ubyte_var,
                                            "file_key": ubyte_var,
                                            "file_type": file_type},
                                     dims=("y", "x"))

    dt.to_netcdf(filename)
    return filename


class TestMimicTPW2Reader:
    """Test Mimic Reader."""

    yaml_file = "mimicTPW2_comp.yaml"

    def setup_method(self):
        """Wrap NetCDF4 file handler with our own fake handler."""
        from satpy._config import config_search_paths
        self.reader_configs = config_search_paths(os.path.join("readers", self.yaml_file))


    def test_init(self, mimic_file):
        """Test basic initialization of this reader."""
        from satpy.readers.core.loading import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([mimic_file])
        assert len(loadables) == 1
        r.create_filehandlers(loadables)
        # make sure we have some files
        assert r.file_handlers

    def test_load_mimic_float(self, mimic_file):
        """Load TPW mimic float data."""
        from satpy.readers.core.loading import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([mimic_file])
        r.create_filehandlers(loadables)
        ds = r.load(float_variables)
        assert len(ds) == len(float_variables)
        for d in ds.values():
            assert d.attrs["platform_shortname"] == "aggregated microwave"
            assert d.attrs["sensor"] == "mimic"
            assert d.attrs["units"] == "mm"
            assert "area" in d.attrs
            assert d.attrs["area"] is not None

    def test_load_mimic_timedelta(self, mimic_file):
        """Load TPW mimic timedelta data (data latency variables)."""
        from satpy.readers.core.loading import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([mimic_file])
        r.create_filehandlers(loadables)
        ds = r.load(date_variables)
        assert len(ds) == len(date_variables)
        for d in ds.values():
            assert d.attrs["platform_shortname"] == "aggregated microwave"
            assert d.attrs["sensor"] == "mimic"
            assert d.attrs["units"] == "minutes"
            assert "area" in d.attrs
            assert d.attrs["area"] is not None
            assert d.dtype == DEFAULT_FILE_DTYPE

    def test_load_mimic_ubyte(self, mimic_file):
        """Load TPW mimic sensor grids."""
        from satpy.readers.core.loading import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([mimic_file])
        r.create_filehandlers(loadables)
        ds = r.load(ubyte_variables)
        assert len(ds) == len(ubyte_variables)
        for d in ds.values():
            assert d.attrs["platform_shortname"] == "aggregated microwave"
            assert d.attrs["sensor"] == "mimic"
            assert "source_key" in d.attrs
            assert "area" in d.attrs
            assert d.attrs["area"] is not None
            assert d.dtype == np.uint8
