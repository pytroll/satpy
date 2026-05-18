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
"""Module for testing the satpy.readers.geocat module."""

import os
from pathlib import Path

import numpy as np
import pytest
from xarray import DataArray, DataTree

DEFAULT_FILE_DTYPE = np.uint16
DEFAULT_FILE_SHAPE = (10, 300)
DEFAULT_FILE_DATA = np.arange(DEFAULT_FILE_SHAPE[0] * DEFAULT_FILE_SHAPE[1],
                              dtype=DEFAULT_FILE_DTYPE).reshape(DEFAULT_FILE_SHAPE)
DEFAULT_FILE_FACTORS = np.array([2.0, 1.0], dtype=np.float32)
DEFAULT_LAT_DATA = np.linspace(45, 65, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
DEFAULT_LAT_DATA = np.repeat([DEFAULT_LAT_DATA], DEFAULT_FILE_SHAPE[0], axis=0)
DEFAULT_LON_DATA = np.linspace(5, 45, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
DEFAULT_LON_DATA = np.repeat([DEFAULT_LON_DATA], DEFAULT_FILE_SHAPE[0], axis=0)


@pytest.fixture(scope="module")
def g13_file(module_tmp_path: Path) -> Path:
    """Create a GOES 13 geocat file."""
    platform_shortname = "GOES-13"
    filename = module_tmp_path / "geocatL2.GOES-13.2015143.234500.nc"
    return _create_geocat_file(filename, platform_shortname)


@pytest.fixture(scope="module")
def h8_file(module_tmp_path: Path) -> Path:
    """Create a HIMAWARI 8 geocat file."""
    platform_shortname = "HIMAWARI-8"
    filename = module_tmp_path / "geocatL2.HIMAWARI-8.2017092.210730.R304.R20.nc"
    return _create_geocat_file(filename, platform_shortname)


@pytest.fixture(scope="module")
def g17_file(module_tmp_path: Path) -> Path:
    """Create a GOES 17 geocat file."""
    filename = module_tmp_path / "geocatL2.GOES-17.CONUS.2020041.163130.hdf"
    return _create_geocat_file(filename, platform_shortname="GOES-17")


def _create_geocat_file(filename, platform_shortname):
    """Create a geocat file."""
    dt = DataTree()
    dt.attrs["Platform_Name"] = platform_shortname
    dt.attrs["Element_Resolution"] = 2.
    dt.attrs["Line_Resolution"] = 2.
    dt.attrs["Subsatellite_Longitude"] = -70.2 if "GOES" in platform_shortname else 140.65

    lons = DEFAULT_LON_DATA

    sensor = {
        "HIMAWARI-8": "himawari8",
        "GOES-17": "goesr",
        "GOES-16": "goesr",
        "GOES-13": "goes",
        "GOES-14": "goes",
        "GOES-15": "goes",
    }[platform_shortname]
    dt.attrs["Sensor_Name"] = sensor

    if platform_shortname == "HIMAWARI-8":
        lons += 130

    dt["pixel_longitude"] = DataArray(lons.astype(float),
                                      attrs={"scale_factor": 1.,
                                             "add_offset": 0.,
                                             "_FillValue": np.nan},
                                      dims=("lines", "elements"))
    dt["pixel_latitude"] = DataArray(DEFAULT_LAT_DATA.astype(float),
                                     attrs={"scale_factor": 1.,
                                            "add_offset": 0.,
                                            "_FillValue": np.nan},
                                     dims=("lines", "elements"))

    dt["variable1"] = DataArray(DEFAULT_FILE_DATA.astype(np.float32),
                                attrs={"_FillValue": -1,
                                       "scale_factor": 1.,
                                       "add_offset": 0.,
                                       "units": "1"},
                                dims=("lines", "elements"))

    # data with fill values
    data = np.ma.masked_array(
        DEFAULT_FILE_DATA.astype(np.float32),
        mask=np.zeros_like(DEFAULT_FILE_DATA))
    data.mask[::5, ::5] = True
    dt["variable2"] = DataArray(data,
                                attrs={"_FillValue": -1,
                                       "scale_factor": 1.,
                                       "add_offset": 0.,
                                       "units": "1"},
                                dims=("lines", "elements"))

    # category
    data = DEFAULT_FILE_DATA.astype(np.byte)
    dt["variable3"] = DataArray(data,
                                attrs={"_FillValue": -128,
                                       "flag_meanings": "clear water supercooled mixed ice unknown",
                                       "flag_values": [0, 1, 2, 3, 4, 5],
                                       "units": "1"},
                                dims=("lines", "elements"))
    dt.to_netcdf(filename)
    return filename


class TestGEOCATReader:
    """Test GEOCAT Reader."""

    yaml_file = "geocat.yaml"

    def setup_method(self):
        """Wrap NetCDF4 file handler with our own fake handler."""
        from satpy._config import config_search_paths
        self.reader_configs = config_search_paths(os.path.join("readers", self.yaml_file))

    def test_init(self, g13_file):
        """Test basic init with no extra parameters."""
        from satpy.readers.core.loading import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([g13_file])
        assert len(loadables) == 1
        r.create_filehandlers(loadables)
        # make sure we have some files
        assert r.file_handlers

    def test_init_with_kwargs(self, g13_file):
        """Test basic init with extra parameters."""
        from satpy.readers.core.loading import load_reader
        r = load_reader(self.reader_configs, xarray_kwargs={"decode_times": True})
        loadables = r.select_files_from_pathnames([g13_file])
        assert len(loadables) == 1
        r.create_filehandlers(loadables, fh_kwargs={"xarray_kwargs": {"decode_times": True}})
        # make sure we have some files
        assert r.file_handlers

    def test_load_all_old_goes(self, g13_file):
        """Test loading all test datasets from old GOES files."""
        from satpy.readers.core.loading import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([g13_file])
        r.create_filehandlers(loadables)
        datasets = r.load(["variable1",
                           "variable2",
                           "variable3"])
        assert len(datasets) == 3
        for v in datasets.values():
            assert "calibration" not in v.attrs
            assert v.attrs["units"] == "1"
        assert datasets["variable3"].attrs.get("flag_meanings") is not None

    def test_load_all_himawari8(self, h8_file):
        """Test loading all test datasets from H8 NetCDF file."""
        from pyresample.geometry import AreaDefinition

        from satpy.readers.core.loading import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([h8_file])
        r.create_filehandlers(loadables)
        datasets = r.load(["variable1",
                           "variable2",
                           "variable3"])
        assert len(datasets) == 3
        for v in datasets.values():
            assert "calibration" not in v.attrs
            assert v.attrs["units"] == "1"
        assert datasets["variable3"].attrs.get("flag_meanings") is not None
        assert isinstance(datasets["variable1"].attrs["area"], AreaDefinition)

    def test_load_all_goes17_hdf4(self, g17_file):
        """Test loading all test datasets from GOES-17 HDF4 file."""
        from pyresample.geometry import AreaDefinition

        from satpy.readers.core.loading import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([g17_file])
        r.create_filehandlers(loadables)
        datasets = r.load(["variable1",
                           "variable2",
                           "variable3"])
        assert len(datasets) == 3
        for v in datasets.values():
            assert "calibration" not in v.attrs
            assert v.attrs["units"] == "1"
        assert datasets["variable3"].attrs.get("flag_meanings") is not None
        assert isinstance(datasets["variable1"].attrs["area"], AreaDefinition)
