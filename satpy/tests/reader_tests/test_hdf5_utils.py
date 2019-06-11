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
"""Module for testing the satpy.readers.hdf5_utils module.
"""

import os
import sys
import numpy as np

try:
    from satpy.readers.hdf5_utils import HDF5FileHandler
except ImportError:
    # fake the import so we can at least run the tests in this file
    HDF5FileHandler = object

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


class FakeHDF5FileHandler(HDF5FileHandler):
    """Swap-in HDF5 File Handler for reader tests to use"""

    def __init__(self, filename, filename_info, filetype_info, **kwargs):
        """Get fake file content from 'get_test_content'"""
        if HDF5FileHandler is object:
            raise ImportError("Base 'HDF5FileHandler' could not be "
                              "imported.")
        filename = str(filename)
        super(HDF5FileHandler, self).__init__(filename, filename_info, filetype_info)
        self.file_content = self.get_test_content(filename, filename_info, filetype_info)
        self.file_content.update(kwargs)

    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content

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
    """Test HDF5 File Handler Utility class"""
    def setUp(self):
        """Create a test HDF5 file"""
        import h5py
        h = h5py.File('test.h5', 'w')
        # Create Group
        g1 = h.create_group('test_group')

        # Add datasets
        ds1_f = g1.create_dataset('ds1_f',
                                  shape=(10, 100),
                                  dtype=np.float32,
                                  data=np.arange(10. * 100).reshape((10, 100)))
        ds1_i = g1.create_dataset('ds1_i',
                                  shape=(10, 100),
                                  dtype=np.int32,
                                  data=np.arange(10 * 100).reshape((10, 100)))
        ds2_f = h.create_dataset('ds2_f',
                                 shape=(10, 100),
                                 dtype=np.float32,
                                 data=np.arange(10. * 100).reshape((10, 100)))
        ds2_i = h.create_dataset('ds2_i',
                                 shape=(10, 100),
                                 dtype=np.int32,
                                 data=np.arange(10 * 100).reshape((10, 100)))

        # Add attributes
        # shows up as a scalar array of bytes (shape=(), size=1)
        h.attrs['test_attr_str'] = 'test_string'
        h.attrs['test_attr_int'] = 0
        h.attrs['test_attr_float'] = 1.2
        # shows up as a numpy bytes object
        h.attrs['test_attr_str_arr'] = np.array(b"test_string2")
        g1.attrs['test_attr_str'] = 'test_string'
        g1.attrs['test_attr_int'] = 0
        g1.attrs['test_attr_float'] = 1.2
        for d in [ds1_f, ds1_i, ds2_f, ds2_i]:
            d.attrs['test_attr_str'] = 'test_string'
            d.attrs['test_attr_int'] = 0
            d.attrs['test_attr_float'] = 1.2

        h.close()

    def tearDown(self):
        """Remove the previously created test file"""
        os.remove('test.h5')

    def test_all_basic(self):
        """Test everything about the HDF5 class"""
        from satpy.readers.hdf5_utils import HDF5FileHandler
        import xarray as xr
        file_handler = HDF5FileHandler('test.h5', {}, {})

        for ds in ('test_group/ds1_f', 'test_group/ds1_i', 'ds2_f', 'ds2_i'):
            self.assertEqual(file_handler[ds].dtype, np.float32 if ds.endswith('f') else np.int32)
            self.assertTupleEqual(file_handler[ds + '/shape'], (10, 100))
            self.assertEqual(file_handler[ds + '/attr/test_attr_str'], 'test_string')
            self.assertEqual(file_handler[ds + '/attr/test_attr_int'], 0)
            self.assertEqual(file_handler[ds + '/attr/test_attr_float'], 1.2)

        self.assertEqual(file_handler['/attr/test_attr_str'], 'test_string')
        self.assertEqual(file_handler['/attr/test_attr_str_arr'], 'test_string2')
        self.assertEqual(file_handler['/attr/test_attr_int'], 0)
        self.assertEqual(file_handler['/attr/test_attr_float'], 1.2)

        self.assertIsInstance(file_handler.get('ds2_f'), xr.DataArray)
        self.assertIsNone(file_handler.get('fake_ds'))
        self.assertEqual(file_handler.get('fake_ds', 'test'), 'test')

        self.assertTrue('ds2_f' in file_handler)
        self.assertFalse('fake_ds' in file_handler)


def suite():
    """The test suite for test_hdf5_utils."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestHDF5FileHandler))

    return mysuite
