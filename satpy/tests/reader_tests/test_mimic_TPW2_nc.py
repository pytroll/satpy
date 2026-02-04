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
DEFAULT_FILE_SHAPE = (9001, 18000)
DEFAULT_LAT = np.linspace(-90, 90, DEFAULT_FILE_SHAPE[0], dtype=DEFAULT_FILE_DTYPE)
DEFAULT_LON = np.linspace(-180, 180, DEFAULT_FILE_SHAPE[1], dtype=DEFAULT_FILE_DTYPE)
DEFAULT_FILE_DATA = np.arange(DEFAULT_FILE_SHAPE[0] * DEFAULT_FILE_SHAPE[1],
                              dtype=DEFAULT_FILE_DTYPE).reshape(DEFAULT_FILE_SHAPE)


@pytest.fixture(scope="session")
def mimic_file(session_tmp_path: Path) -> Path:
    """Mimic a real data file."""
    filename = session_tmp_path / "comp20190619.130000.nc"
    dt_s = datetime(2019, 6, 19, 13, 0)
    dt_e = datetime(2019, 6, 19, 13, 0)

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

    dt["tpwGrid"] = xr.DataArray(DEFAULT_FILE_DATA,
                                 attrs=dict(units="mm"),
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

    def test_load_mimic(self, mimic_file):
        """Load Mimic data."""
        from satpy.readers.core.loading import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([mimic_file])
        r.create_filehandlers(loadables)
        ds = r.load(["tpwGrid"])
        assert len(ds) == 1
        for d in ds.values():
            assert d.attrs["platform_shortname"] == "aggregated microwave"
            assert d.attrs["sensor"] == "mimic"
            assert "area" in d.attrs
            assert "units" in d.attrs
            assert d.attrs["area"] is not None
