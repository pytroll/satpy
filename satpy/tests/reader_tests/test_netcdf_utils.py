#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module for testing the satpy.readers.netcdf_utils module.
"""

import os
import sys
import numpy as np

try:
    from satpy.readers.netcdf_utils import NetCDF4FileHandler
except ImportError:
    # fake the import so we can at least run the tests in this file
    NetCDF4FileHandler = object

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


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
            self.assertEqual(file_handler[ds + '/attr/test_attr_str'], 'test_string')
            self.assertEqual(file_handler[ds + '/attr/test_attr_int'], 0)
            self.assertEqual(file_handler[ds + '/attr/test_attr_float'], 1.2)

        self.assertEqual(file_handler['/attr/test_attr_str'], 'test_string')
        self.assertEqual(file_handler['/attr/test_attr_str_arr'], 'test_string2')
        self.assertEqual(file_handler['/attr/test_attr_int'], 0)
        self.assertEqual(file_handler['/attr/test_attr_float'], 1.2)

        self.assertIsInstance(file_handler.get('ds2_f')[:], xr.DataArray)
        self.assertIsNone(file_handler.get('fake_ds'))
        self.assertEqual(file_handler.get('fake_ds', 'test'), 'test')

        self.assertTrue('ds2_f' in file_handler)
        self.assertFalse('fake_ds' in file_handler)


def suite():
    """The test suite for test_netcdf_utils."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestNetCDF4FileHandler))

    return mysuite
