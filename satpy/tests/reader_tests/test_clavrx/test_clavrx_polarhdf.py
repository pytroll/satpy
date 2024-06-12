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
"""Module for testing the satpy.readers.clavrx module."""

import os
import unittest
from unittest import mock

import dask.array as da
import numpy as np
import xarray as xr
from pyresample.geometry import SwathDefinition

from satpy.tests.reader_tests.test_hdf4_utils import FakeHDF4FileHandler

DEFAULT_FILE_DTYPE = np.uint16
DEFAULT_FILE_SHAPE = (10, 300)
DEFAULT_FILE_DATA = np.arange(DEFAULT_FILE_SHAPE[0] * DEFAULT_FILE_SHAPE[1],
                              dtype=DEFAULT_FILE_DTYPE).reshape(DEFAULT_FILE_SHAPE)
DEFAULT_FILE_FACTORS = np.array([2.0, 1.0], dtype=np.float32)
DEFAULT_LAT_DATA = np.linspace(45, 65, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
DEFAULT_LAT_DATA = np.repeat([DEFAULT_LAT_DATA], DEFAULT_FILE_SHAPE[0], axis=0)
DEFAULT_LON_DATA = np.linspace(5, 45, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
DEFAULT_LON_DATA = np.repeat([DEFAULT_LON_DATA], DEFAULT_FILE_SHAPE[0], axis=0)


class FakeHDF4FileHandlerPolar(FakeHDF4FileHandler):
    """Swap-in HDF4 File Handler."""

    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content."""
        file_content = {
            "/attr/platform": "SNPP",
            "/attr/sensor": "VIIRS",
        }
        file_content["longitude"] = xr.DataArray(
            da.from_array(DEFAULT_LON_DATA, chunks=4096),
            attrs={
                "_FillValue": np.nan,
                "scale_factor": 1.,
                "add_offset": 0.,
                "standard_name": "longitude",
            })
        file_content["longitude/shape"] = DEFAULT_FILE_SHAPE

        file_content["latitude"] = xr.DataArray(
            da.from_array(DEFAULT_LAT_DATA, chunks=4096),
            attrs={
                "_FillValue": np.nan,
                "scale_factor": 1.,
                "add_offset": 0.,
                "standard_name": "latitude",
            })
        file_content["latitude/shape"] = DEFAULT_FILE_SHAPE

        file_content["variable1"] = xr.DataArray(
            da.from_array(DEFAULT_FILE_DATA, chunks=4096).astype(np.float32),
            attrs={
                "_FillValue": -1,
                "scale_factor": 1.,
                "add_offset": 0.,
                "units": "1",
            })
        file_content["variable1/shape"] = DEFAULT_FILE_SHAPE

        # data with fill values
        file_content["variable2"] = xr.DataArray(
            da.from_array(DEFAULT_FILE_DATA, chunks=4096).astype(np.float32),
            attrs={
                "_FillValue": -1,
                "scale_factor": 1.,
                "add_offset": 0.,
                "units": "1",
            })
        file_content["variable2/shape"] = DEFAULT_FILE_SHAPE
        file_content["variable2"] = file_content["variable2"].where(
                                        file_content["variable2"] % 2 != 0)

        # category
        file_content["variable3"] = xr.DataArray(
            da.from_array(DEFAULT_FILE_DATA, chunks=4096).astype(np.byte),
            attrs={
                "SCALED": 0,
                "_FillValue": -128,
                "flag_meanings": "clear water supercooled mixed ice unknown",
                "flag_values": [0, 1, 2, 3, 4, 5],
                "units": "none",
            })
        file_content["variable3/shape"] = DEFAULT_FILE_SHAPE

        file_content["refl_1_38um_nom"] = xr.DataArray(
            da.from_array(DEFAULT_FILE_DATA, chunks=4096).astype(np.float32),
            attrs={
                "SCALED": 1,
                "add_offset": 59.0,
                "scale_factor": 0.0018616290763020515,
                "units": "%",
                "_FillValue": -32768,
                "valid_range": [-32767, 32767],
                "actual_range": [-2., 120.],
                "actual_missing": -999.0
            })
        file_content["refl_1_38um_nom/shape"] = DEFAULT_FILE_SHAPE

        return file_content


class TestCLAVRXReaderPolar(unittest.TestCase):
    """Test CLAVR-X Reader with Polar files."""

    yaml_file = "clavrx.yaml"

    def setUp(self):
        """Wrap HDF4 file handler with our own fake handler."""
        from satpy._config import config_search_paths
        from satpy.readers.clavrx import CLAVRXHDF4FileHandler
        self.reader_configs = config_search_paths(os.path.join("readers", self.yaml_file))
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.p = mock.patch.object(CLAVRXHDF4FileHandler, "__bases__", (FakeHDF4FileHandlerPolar,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

    def tearDown(self):
        """Stop wrapping the NetCDF4 file handler."""
        self.p.stop()

    def test_init(self):
        """Test basic init with no extra parameters."""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            "clavrx_npp_d20170520_t2053581_e2055223_b28822.level2.hdf",
        ])
        assert len(loadables) == 1
        r.create_filehandlers(loadables)
        # make sure we have some files
        assert r.file_handlers

    def test_available_datasets(self):
        """Test available_datasets with fake variables from YAML."""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            "clavrx_npp_d20170520_t2053581_e2055223_b28822.level2.hdf",
        ])
        assert len(loadables) == 1
        r.create_filehandlers(loadables)
        # make sure we have some files
        assert r.file_handlers

        # mimic the YAML file being configured for more datasets
        fake_dataset_info = [
            (None, {"name": "variable1", "resolution": None, "file_type": ["clavrx_hdf4"]}),
            (True, {"name": "variable2", "resolution": 742, "file_type": ["clavrx_hdf4"]}),
            (True, {"name": "variable2", "resolution": 1, "file_type": ["clavrx_hdf4"]}),
            (None, {"name": "variable2", "resolution": 1, "file_type": ["clavrx_hdf4"]}),
            (None, {"name": "_fake1", "file_type": ["clavrx_hdf4"]}),
            (None, {"name": "variable1", "file_type": ["level_fake"]}),
            (True, {"name": "variable3", "file_type": ["clavrx_hdf4"]}),
        ]
        new_ds_infos = list(r.file_handlers["clavrx_hdf4"][0].available_datasets(
            fake_dataset_info))
        assert len(new_ds_infos) == 9

        # we have this and can provide the resolution
        assert new_ds_infos[0][0]
        assert new_ds_infos[0][1]["resolution"] == 742  # hardcoded

        # we have this, but previous file handler said it knew about it
        # and it is producing the same resolution as what we have
        assert new_ds_infos[1][0]
        assert new_ds_infos[1][1]["resolution"] == 742

        # we have this, but don't want to change the resolution
        # because a previous handler said it has it
        assert new_ds_infos[2][0]
        assert new_ds_infos[2][1]["resolution"] == 1

        # even though the previous one was known we can still
        # produce it at our new resolution
        assert new_ds_infos[3][0]
        assert new_ds_infos[3][1]["resolution"] == 742

        # we have this and can update the resolution since
        # no one else has claimed it
        assert new_ds_infos[4][0]
        assert new_ds_infos[4][1]["resolution"] == 742

        # we don"t have this variable, don't change it
        assert not new_ds_infos[5][0]
        assert new_ds_infos[5][1].get("resolution") is None

        # we have this, but it isn't supposed to come from our file type
        assert new_ds_infos[6][0] is None
        assert new_ds_infos[6][1].get("resolution") is None

        # we could have loaded this but some other file handler said it has this
        assert new_ds_infos[7][0]
        assert new_ds_infos[7][1].get("resolution") is None

        # we can add resolution to the previous dataset, so we do
        assert new_ds_infos[8][0]
        assert new_ds_infos[8][1]["resolution"] == 742

    def test_available_datasets_with_alias(self):
        """Test availability of aliased dataset."""
        import xarray as xr

        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        with mock.patch("satpy.readers.clavrx.SDS", xr.DataArray):
            loadables = r.select_files_from_pathnames([
                "clavrx_npp_d20170520_t2053581_e2055223_b28822.level2.hdf",
            ])
            r.create_filehandlers(loadables)
            available_ds = list(r.file_handlers["clavrx_hdf4"][0].available_datasets())

            assert available_ds[5][1]["name"] == "refl_1_38um_nom"
            assert available_ds[6][1]["name"] == "M09"
            assert available_ds[6][1]["file_key"] == "refl_1_38um_nom"

    def test_load_all(self):
        """Test loading all test datasets."""
        import xarray as xr

        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        with mock.patch("satpy.readers.clavrx.SDS", xr.DataArray):
            loadables = r.select_files_from_pathnames([
                "clavrx_npp_d20170520_t2053581_e2055223_b28822.level2.hdf",
            ])
            r.create_filehandlers(loadables)

        var_list = ["M09", "variable2", "variable3"]
        datasets = r.load(var_list)
        assert len(datasets) == len(var_list)
        for v in datasets.values():
            assert v.attrs["units"] in ["1", "%"]
            assert v.attrs["platform_name"] == "npp"
            assert v.attrs["sensor"] == "viirs"
            assert isinstance(v.attrs["area"], SwathDefinition)
            assert v.attrs["area"].lons.attrs["rows_per_scan"] == 16
            assert v.attrs["area"].lats.attrs["rows_per_scan"] == 16
        assert isinstance(datasets["variable3"].attrs.get("flag_meanings"), list)
