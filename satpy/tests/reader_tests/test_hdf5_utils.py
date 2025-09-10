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
"""Module for testing the satpy.readers.core.hdf5 module."""

import os
import unittest

import numpy as np

try:
    from satpy.readers.core.hdf5 import HDF5FileHandler
except ImportError:
    # fake the import so we can at least run the tests in this file
    HDF5FileHandler = object  # type: ignore


class FakeHDF5FileHandler(HDF5FileHandler):
    """Swap  HDF5 File Handler for reader tests to use."""

    def __init__(self, filename, filename_info, filetype_info, **kwargs):
        """Get fake file content from 'get_test_content'."""
        if HDF5FileHandler is object:
            raise ImportError("Base 'HDF5FileHandler' could not be "
                              "imported.")
        filename = str(filename)
        super(HDF5FileHandler, self).__init__(filename, filename_info, filetype_info)
        self.file_content = self.get_test_content(filename, filename_info, filetype_info)
        self.file_content.update(kwargs)

    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content.

        Args:
            filename (str): input filename
            filename_info (dict): Dict of metadata pulled from filename
            filetype_info (dict): Dict of metadata from the reader's yaml config for this file type

        Returns: dict of file content with keys like:

            - 'dataset'
            - '/attr/global_attr'
            - 'dataset/attr/global_attr'
            - 'dataset/shape'

        """
        raise NotImplementedError("Fake File Handler subclass must implement 'get_test_content'")


class TestHDF5FileHandler(unittest.TestCase):
    """Test HDF5 File Handler Utility class."""

    def setUp(self):
        """Create a test HDF5 file."""
        import h5py
        h = h5py.File("test.h5", "w")
        # Create Group
        g1 = h.create_group("test_group")

        # Add datasets
        ds1_f = g1.create_dataset("ds1_f",
                                  shape=(10, 100),
                                  dtype=np.float32,
                                  data=np.arange(10. * 100).reshape((10, 100)))
        ds1_i = g1.create_dataset("ds1_i",
                                  shape=(10, 100),
                                  dtype=np.int32,
                                  data=np.arange(10 * 100).reshape((10, 100)))
        ds2_f = h.create_dataset("ds2_f",
                                 shape=(10, 100),
                                 dtype=np.float32,
                                 data=np.arange(10. * 100).reshape((10, 100)))
        ds2_i = h.create_dataset("ds2_i",
                                 shape=(10, 100),
                                 dtype=np.int32,
                                 data=np.arange(10 * 100).reshape((10, 100)))

        # Add attributes
        # shows up as a scalar array of bytes (shape=(), size=1)
        h.attrs["test_attr_str"] = "test_string"
        h.attrs["test_attr_byte"] = b"test_byte"
        h.attrs["test_attr_int"] = 0
        h.attrs["test_attr_float"] = 1.2
        # shows up as a numpy bytes object
        h.attrs["test_attr_str_arr"] = np.array(b"test_string2")
        g1.attrs["test_attr_str"] = "test_string"
        g1.attrs["test_attr_byte"] = b"test_byte"
        g1.attrs["test_attr_int"] = 0
        g1.attrs["test_attr_float"] = 1.2
        for d in [ds1_f, ds1_i, ds2_f, ds2_i]:
            d.attrs["test_attr_str"] = "test_string"
            d.attrs["test_attr_byte"] = b"test_byte"
            d.attrs["test_attr_int"] = 0
            d.attrs["test_attr_float"] = 1.2
            d.attrs["test_ref"] = d.ref
        self.var_attrs = list(d.attrs.keys())

        h.close()

    def tearDown(self):
        """Remove the previously created test file."""
        os.remove("test.h5")

    def test_all_basic(self):
        """Test everything about the HDF5 class."""
        import xarray as xr

        from satpy.readers.core.hdf5 import HDF5FileHandler
        file_handler = HDF5FileHandler("test.h5", {}, {})

        for ds_name in ("test_group/ds1_f", "test_group/ds1_i", "ds2_f", "ds2_i"):
            ds = file_handler[ds_name]
            attrs = ds.attrs
            assert ds.dtype == (np.float32 if ds_name.endswith("f") else np.int32)
            assert file_handler[ds_name + "/shape"] == (10, 100)
            assert attrs["test_attr_str"] == "test_string"
            assert attrs["test_attr_byte"] == "test_byte"
            assert attrs["test_attr_int"] == 0
            assert attrs["test_attr_float"] == 1.2
            assert file_handler[ds_name + "/attr/test_attr_str"] == "test_string"
            assert file_handler[ds_name + "/attr/test_attr_byte"] == "test_byte"
            assert file_handler[ds_name + "/attr/test_attr_int"] == 0
            assert file_handler[ds_name + "/attr/test_attr_float"] == 1.2

        assert file_handler["/attr/test_attr_str"] == "test_string"
        assert file_handler["/attr/test_attr_byte"] == "test_byte"
        assert file_handler["/attr/test_attr_str_arr"] == "test_string2"
        assert file_handler["/attr/test_attr_int"] == 0
        assert file_handler["/attr/test_attr_float"] == 1.2

        assert isinstance(file_handler.get("ds2_f"), xr.DataArray)
        assert file_handler.get("fake_ds") is None
        assert file_handler.get("fake_ds", "test") == "test"

        assert "ds2_f" in file_handler
        assert "fake_ds" not in file_handler

        assert isinstance(file_handler["ds2_f/attr/test_ref"], np.ndarray)

    def test_array_name_uniqueness(self):
        """Test the dask array generated from an hdf5 dataset stay constant and unique."""
        from satpy.readers.core.hdf5 import HDF5FileHandler
        file_handler = HDF5FileHandler("test.h5", {}, {})

        dsname = "test_group/ds1_f"

        assert file_handler[dsname].data.name == file_handler[dsname].data.name
        assert file_handler[dsname].data.name.startswith("/" + dsname)
