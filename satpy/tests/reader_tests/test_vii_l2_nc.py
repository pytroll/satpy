#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Satpy developers
#
# satpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# satpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with satpy.  If not, see <http://www.gnu.org/licenses/>.

"""The vii_2_nc reader tests package."""

import os
import numpy as np
import xarray as xr
import datetime
from netCDF4 import Dataset

from satpy import CHUNK_SIZE
from satpy.readers.vii_l2_nc import ViiL2NCFileHandler

import unittest

try:
    from unittest import mock
except ImportError:
    import mock


class Test_ViiL2NCFileHandler(unittest.TestCase):
    """Test the ViiL2NCFileHandler reader."""

    def setUp(self):
        """Set up the test."""
        # Easiest way to test the reader is to create a test netCDF file on the fly
        with Dataset('test.nc', 'w') as nc:
            # Create data group
            g1 = nc.createGroup('data')

            # Add dimensions to data group
            g1.createDimension('num_pixels', 10)
            g1.createDimension('num_lines', 100)

            # Create measurement_data group
            g1_2 = g1.createGroup('measurement_data')

            # Add variables to data/measurement_data group
            delta_lat = g1_2.createVariable('delta_lat', np.float32, dimensions=('num_pixels', 'num_lines'))
            delta_lat[:] = 1.0

        self.reader = ViiL2NCFileHandler(
            filename='test.nc',
            filename_info={
                'creation_time': datetime.datetime(year=2017, month=9, day=22,
                                                   hour=22, minute=40, second=10),
                'sensing_start_time': datetime.datetime(year=2017, month=9, day=20,
                                                        hour=12, minute=30, second=30),
                'sensing_end_time': datetime.datetime(year=2017, month=9, day=20,
                                                      hour=18, minute=30, second=50)
                },
            filetype_info={}
            )

    def tearDown(self):
        """Remove the previously created test file."""
        os.remove('test.nc')

    @mock.patch('satpy.readers.vii_l2_nc.xr')
    @mock.patch('satpy.readers.vii_l2_nc.da')
    def test_functions(self, da_, xr_):
        """Test the functions."""
        # Checks that the _perform_orthorectification function is correctly executed
        variable = xr.DataArray(
            dims=('x', 'y'),
            name='test_name',
            attrs={
                'key_1': 'value_1',
                'key_2': 'value_2'
                },
            data=np.ones((10, 100))
        )
        self.reader._perform_orthorectification(variable, 'data/measurement_data/delta_lat')

        expected_values = np.ones((10, 100)) + np.ones((10, 100))

        # Checks that dask.array is called with the correct arguments
        name, args, kwargs = da_.mock_calls[0]
        self.assertTrue(np.allclose(args[0], expected_values))
        self.assertEqual(args[1], CHUNK_SIZE)

        # Checks that xarray.DataArray is called with the correct arguments
        name, args, kwargs = xr_.mock_calls[0]
        self.assertEqual(kwargs['attrs'], {'key_1': 'value_1', 'key_2': 'value_2'})
        self.assertEqual(kwargs['name'], 'test_name')
        self.assertEqual(kwargs['dims'], ('x', 'y'))
        # The 'data' argument must be the return result of dask.array
        self.assertEqual(kwargs['data']._extract_mock_name(), 'da.from_array()')


def suite():
    """Build the test suite for test_scene."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()

    mysuite.addTest(loader.loadTestsFromTestCase(Test_ViiL2NCFileHandler))

    return mysuite


if __name__ == '__main__':
    unittest.main()
