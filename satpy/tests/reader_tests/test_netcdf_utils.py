#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2020 Satpy developers
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
"""Module for testing the satpy.readers.netcdf_utils module."""

import os
import unittest
import numpy as np

try:
    from satpy.readers.netcdf_utils import NetCDF4FileHandler
except ImportError:
    # fake the import so we can at least run the tests in this file
    NetCDF4FileHandler = object


class FakeNetCDF4FileHandler(NetCDF4FileHandler):
    """Swap-in NetCDF4 File Handler for reader tests to use."""

    def __init__(self, filename, filename_info, filetype_info, **kwargs):
        """Get fake file content from 'get_test_content'."""
        if NetCDF4FileHandler is object:
            raise ImportError("Base 'NetCDF4FileHandler' could not be "
                              "imported.")
        super(NetCDF4FileHandler, self).__init__(filename, filename_info, filetype_info)
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
            - 'dataset/dimensions'
            - '/dimension/my_dim'

        """
        raise NotImplementedError("Fake File Handler subclass must implement 'get_test_content'")


class TestNetCDF4FileHandler(unittest.TestCase):
    """Test NetCDF4 File Handler Utility class."""

    def setUp(self):
        """Create a test NetCDF4 file."""
        from netCDF4 import Dataset
        with Dataset('test.nc', 'w') as nc:
            # Create dimensions
            nc.createDimension('rows', 10)
            nc.createDimension('cols', 100)

            # Create Group
            g1 = nc.createGroup('test_group')

            # Add datasets
            ds1_f = g1.createVariable('ds1_f', np.float32,
                                      dimensions=('rows', 'cols'))
            ds1_f[:] = np.arange(10. * 100).reshape((10, 100))
            ds1_i = g1.createVariable('ds1_i', np.int32,
                                      dimensions=('rows', 'cols'))
            ds1_i[:] = np.arange(10 * 100).reshape((10, 100))
            ds2_f = nc.createVariable('ds2_f', np.float32,
                                      dimensions=('rows', 'cols'))
            ds2_f[:] = np.arange(10. * 100).reshape((10, 100))
            ds2_i = nc.createVariable('ds2_i', np.int32,
                                      dimensions=('rows', 'cols'))
            ds2_i[:] = np.arange(10 * 100).reshape((10, 100))
            ds2_s = nc.createVariable("ds2_s", np.int8,
                                      dimensions=("rows",))
            ds2_s[:] = np.arange(10)
            ds2_sc = nc.createVariable("ds2_sc", np.int8, dimensions=())
            ds2_sc[:] = 42

            # Add attributes
            nc.test_attr_str = 'test_string'
            nc.test_attr_int = 0
            nc.test_attr_float = 1.2
            nc.test_attr_str_arr = np.array(b"test_string2")
            g1.test_attr_str = 'test_string'
            g1.test_attr_int = 0
            g1.test_attr_float = 1.2
            for d in [ds1_f, ds1_i, ds2_f, ds2_i]:
                d.test_attr_str = 'test_string'
                d.test_attr_int = 0
                d.test_attr_float = 1.2

    def tearDown(self):
        """Remove the previously created test file."""
        os.remove('test.nc')

    def test_all_basic(self):
        """Test everything about the NetCDF4 class."""
        from satpy.readers.netcdf_utils import NetCDF4FileHandler
        import xarray as xr
        file_handler = NetCDF4FileHandler('test.nc', {}, {})

        self.assertEqual(file_handler['/dimension/rows'], 10)
        self.assertEqual(file_handler['/dimension/cols'], 100)

        for ds in ('test_group/ds1_f', 'test_group/ds1_i', 'ds2_f', 'ds2_i'):
            self.assertEqual(file_handler[ds].dtype, np.float32 if ds.endswith('f') else np.int32)
            self.assertTupleEqual(file_handler[ds + '/shape'], (10, 100))
            self.assertEqual(file_handler[ds + '/dimensions'], ("rows", "cols"))
            self.assertEqual(file_handler[ds + '/attr/test_attr_str'], 'test_string')
            self.assertEqual(file_handler[ds + '/attr/test_attr_int'], 0)
            self.assertEqual(file_handler[ds + '/attr/test_attr_float'], 1.2)

        test_group = file_handler['test_group']
        self.assertTupleEqual(test_group['ds1_i'].shape, (10, 100))
        self.assertTupleEqual(test_group['ds1_i'].dims, ('rows', 'cols'))

        self.assertEqual(file_handler['/attr/test_attr_str'], 'test_string')
        self.assertEqual(file_handler['/attr/test_attr_str_arr'], 'test_string2')
        self.assertEqual(file_handler['/attr/test_attr_int'], 0)
        self.assertEqual(file_handler['/attr/test_attr_float'], 1.2)

        self.assertIsInstance(file_handler.get('ds2_f')[:], xr.DataArray)
        self.assertIsNone(file_handler.get('fake_ds'))
        self.assertEqual(file_handler.get('fake_ds', 'test'), 'test')

        self.assertTrue('ds2_f' in file_handler)
        self.assertFalse('fake_ds' in file_handler)
        self.assertIsNone(file_handler.file_handle)
        self.assertEqual(file_handler["ds2_sc"], 42)

    def test_caching(self):
        """Test that caching works as intended."""
        from satpy.readers.netcdf_utils import NetCDF4FileHandler
        h = NetCDF4FileHandler("test.nc", {}, {}, cache_var_size=1000,
                               cache_handle=True)
        self.assertIsNotNone(h.file_handle)
        self.assertTrue(h.file_handle.isopen())

        self.assertEqual(sorted(h.cached_file_content.keys()),
                         ["ds2_s", "ds2_sc"])
        # with caching, these tests access different lines than without
        np.testing.assert_array_equal(h["ds2_s"], np.arange(10))
        np.testing.assert_array_equal(h["test_group/ds1_i"],
                                      np.arange(10 * 100).reshape((10, 100)))
        # check that root variables can still be read from cached file object,
        # even if not cached themselves
        np.testing.assert_array_equal(
                h["ds2_f"],
                np.arange(10. * 100).reshape((10, 100)))
        h.__del__()
        self.assertFalse(h.file_handle.isopen())

    def test_filenotfound(self):
        """Test that error is raised when file not found."""
        from satpy.readers.netcdf_utils import NetCDF4FileHandler

        with self.assertRaises(IOError):
            NetCDF4FileHandler("/thisfiledoesnotexist.nc", {}, {})
