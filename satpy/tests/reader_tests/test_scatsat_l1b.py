#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Satpy developers
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
"""Module for testing the satpy.readers.scatsat1_l2b module."""

import os
import unittest
from unittest import mock

import numpy as np

from satpy.tests.reader_tests.test_hdf5_utils import FakeHDF5FileHandler
from satpy.tests.utils import convert_file_content_to_data_array

DEFAULT_FILE_DTYPE = np.uint16
DEFAULT_FILE_SHAPE = (12, 105)
DEFAULT_FILE_DATA = np.arange(DEFAULT_FILE_SHAPE[0] * DEFAULT_FILE_SHAPE[1],
                              dtype=DEFAULT_FILE_DTYPE).reshape(DEFAULT_FILE_SHAPE)
DEFAULT_FILE_FACTORS = np.array([2.0, 1.0], dtype=np.float32)
DEFAULT_LAT_DATA = np.linspace(45, 65, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
DEFAULT_LAT_DATA = np.repeat([DEFAULT_LAT_DATA], DEFAULT_FILE_SHAPE[0], axis=0)
DEFAULT_LON_DATA = np.linspace(5, 45, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
DEFAULT_LON_DATA = np.repeat([DEFAULT_LON_DATA], DEFAULT_FILE_SHAPE[0], axis=0)


class FakeHDF5FileHandler2(FakeHDF5FileHandler):
    """Swap-in HDF5 File Handler."""

    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content."""
        file_content = {
            "science_data/attr/Range Beginning Date": "2025-139T04:52:21.469",
            "science_data/attr/Range Ending Date": "2025-139T05:42:01.023",
            "science_data/attr/Satellite Name": "EOS-06          ",
            "science_data/attr/Longitude Scale": "0.01",
            "science_data/attr/Latitude Scale": "0.01",
            "science_data/attr/Wind Speed Selection Scale": "0.1",
            "science_data/attr/Wind Direction Selection Scale": "1.0",
        }
        for wd in ["Wind_speed_selection", "Wind_direction_selection"]:
            k = f"science_data/{wd}"
            file_content[k] = DEFAULT_FILE_DATA[:, :]
            file_content[k + "/shape"] = (DEFAULT_FILE_SHAPE[0], DEFAULT_FILE_SHAPE[1])

        lon_k = "science_data/Longitude"
        lat_k = "science_data/Latitude"
        file_content[lon_k] = DEFAULT_LON_DATA
        file_content[lon_k + "/shape"] = DEFAULT_FILE_SHAPE
        file_content[lat_k] = DEFAULT_LAT_DATA
        file_content[lat_k + "/shape"] = DEFAULT_FILE_SHAPE

        convert_file_content_to_data_array(file_content)
        return file_content


class TestSCATSAT1L2BReader(unittest.TestCase):
    """Test SCATSAT L2B Reader."""

    yaml_file = "scatsat1_l2b.yaml"

    def setUp(self):
        """Wrap HDF5 file handler with our own fake handler."""
        from satpy._config import config_search_paths
        from satpy.readers.scatsat1_l2b import SCATSAT1L2BFileHandler
        self.reader_configs = config_search_paths(os.path.join("readers", self.yaml_file))
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.p = mock.patch.object(SCATSAT1L2BFileHandler, "__bases__", (FakeHDF5FileHandler2,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

    def tearDown(self):
        """Stop wrapping the HDF5 file handler."""
        self.p.stop()

    def test_init(self):
        """Test basic init with no extra parameters."""
        from satpy.readers.core.loading import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            "E06SCTL2B2025139_13087_13088_SN_12km_2025-139T07-55-10_v1.0.4.h5",
        ])
        assert len(loadables) == 1
        r.create_filehandlers(loadables)
        # make sure we have some files
        assert r.file_handlers

    def test_load_basic(self):
        """Test loading of basic channels."""
        from satpy.readers.core.loading import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            "E06SCTL2B2025139_13087_13088_SN_12km_2025-139T07-55-10_v1.0.4.h5",
        ])
        assert len(loadables) == 1
        r.create_filehandlers(loadables)
        ds = r.load(["wind_speed", "wind_direction"])
        assert len(ds) == 2
        for d in ds.values():
            assert d.shape == (DEFAULT_FILE_SHAPE[0], int(DEFAULT_FILE_SHAPE[1]))
            assert "area" in d.attrs
            assert d.attrs["area"] is not None
            assert d.attrs["area"].lons.shape == (DEFAULT_FILE_SHAPE[0], DEFAULT_FILE_SHAPE[1])
            assert d.attrs["area"].lats.shape == (DEFAULT_FILE_SHAPE[0], DEFAULT_FILE_SHAPE[1])
            assert d.attrs["sensor"] == "Scatterometer"
            assert d.attrs["platform_name"] == "EOS-06          "
            assert d.attrs["resolution"] == 12500

    def test_load_basic_25km_resolution(self):
        """Test loading of basic channels from 25km resolution data."""
        from satpy.readers.core.loading import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            "E06SCTL2B2025139_13087_13088_SN_25km_2025-139T07-55-10_v1.0.4.h5",
        ])
        assert len(loadables) == 1
        r.create_filehandlers(loadables)
        ds = r.load(["wind_speed", "wind_direction"])
        assert len(ds) == 2
        for d in ds.values():
            assert d.attrs["resolution"] == 25000

    def test_load_wind_speed(self):
        """Test loading of wind_speed."""
        from satpy.readers.core.loading import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            "E06SCTL2B2025139_13087_13088_SN_12km_2025-139T07-55-10_v1.0.4.h5",
        ])
        assert len(loadables) == 1
        r.create_filehandlers(loadables)
        ds = r.load([
            "wind_speed",
        ])
        assert len(ds) == 1
        for d in ds.values():
            assert d.shape == DEFAULT_FILE_SHAPE
            assert "area" in d.attrs
            assert d.attrs["area"] is not None
            assert d.attrs["area"].lons.shape == DEFAULT_FILE_SHAPE
            assert d.attrs["area"].lats.shape == DEFAULT_FILE_SHAPE

    def test_properties(self):
        """Test platform_name."""
        import datetime as dt

        from satpy.readers.core.loading import load_reader
        filenames = [
            "E06SCTL2B2025139_13087_13088_SN_12km_2025-139T07-55-10_v1.0.4.h5", ]

        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        reader.create_filehandlers(files)
        # Make sure we have some files
        res = reader.load(["wind_speed"])
        assert res["wind_speed"].platform_name == "EOS-06          "
        assert res["wind_speed"].start_time == dt.datetime(2025, 5, 19, 4, 52, 21, 469000)
        assert res["wind_speed"].end_time == dt.datetime(2025, 5, 19, 5, 42, 1, 23000)

    def test_available_dataset_ids(self):
        """Test available_dataset_ids method."""
        from satpy.readers.core.loading import load_reader
        from satpy.dataset.dataid import DataID, default_id_keys_config
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            "E06SCTL2B2025139_13087_13088_SN_12km_2025-139T07-55-10_v1.0.4.h5",
        ])
        assert len(loadables) == 1
        r.create_filehandlers(loadables)
        available_ids = r.available_dataset_ids
        print(type(available_ids), available_ids)
        expected_keys = {
                        DataID(default_id_keys_config, name='longitude', resolution=12500, modifiers=()),
                        DataID(default_id_keys_config, name='latitude', resolution=12500, modifiers=()),
                        DataID(default_id_keys_config, name='wind_speed', resolution=12500, modifiers=()),
                        DataID(default_id_keys_config, name='wind_direction', resolution=12500, modifiers=())
        }
        print(set(available_ids))
        assert set(available_ids) == expected_keys
