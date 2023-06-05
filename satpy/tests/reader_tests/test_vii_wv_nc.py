#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Satpy developers
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

"""The vii_l2_nc reader tests package for VII/METimage water vapour products."""

import datetime
import os
import unittest
import uuid

import dask.array as da
import numpy as np
import xarray as xr
from netCDF4 import Dataset

from satpy.readers.vii_l2_nc import ViiL2NCFileHandler

TEST_FILE = 'test_file_vii_wv_nc.nc'


class TestViiL2NCFileHandler(unittest.TestCase):
    """Test the ViiL2NCFileHandler reader."""

    def setUp(self):
        """Set up the test."""
        # Easiest way to test the reader is to create a test netCDF file on the fly
        # uses a UUID to avoid permission conflicts during execution of tests in parallel
        self.test_file_name = TEST_FILE + str(uuid.uuid1()) + ".nc"

        with Dataset(self.test_file_name, 'w') as nc:
            # Create data group
            g1 = nc.createGroup('data')

            # Add dimensions to data group
            g1.createDimension('num_points_act', 100)
            g1.createDimension('num_points_alt', 10)

            # Create measurement_data group
            g1_2 = g1.createGroup('measurement_data')

            # Add variables to data/measurement_data group
            delta_lat = g1_2.createVariable('delta_lat', np.float32, dimensions=('num_points_alt', 'num_points_act'))
            delta_lat[:] = 0.1

        self.reader = ViiL2NCFileHandler(
            filename=self.test_file_name,
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
        # Catch Windows PermissionError for removing the created test file.
        try:
            os.remove(self.test_file_name)
        except OSError:
            pass

    def test_functions(self):
        """Test the functions."""
        # Checks that the _perform_orthorectification function is correctly executed
        variable = xr.DataArray(
            dims=('num_points_alt', 'num_points_act'),
            name='test_name',
            attrs={
                'key_1': 'value_1',
                'key_2': 'value_2'
            },
            data=da.from_array(np.ones((10, 100)))
        )
        orthorect_variable = self.reader._perform_orthorectification(variable, 'data/measurement_data/delta_lat')

        expected_values = 1.1 * np.ones((10, 100))
        self.assertTrue(np.allclose(orthorect_variable.values, expected_values))
        self.assertEqual(orthorect_variable.attrs['key_1'], 'value_1')
