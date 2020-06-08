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

"""The vii_l1b_nc reader tests package."""

import os

import numpy as np
import xarray as xr
import dask.array as da
import datetime
from netCDF4 import Dataset
import uuid

from satpy.readers.vii_l1b_nc import ViiL1bNCFileHandler
from satpy.readers.vii_utils import MEAN_EARTH_RADIUS

import unittest

TEST_FILE = 'test_file_vii_l1b_nc.nc'


class TestViiL1bNCFileHandler(unittest.TestCase):
    """Test the ViiL1bNCFileHandler reader."""

    def setUp(self):
        """Set up the test."""
        # Easiest way to test the reader is to create a test netCDF file on the fly
        # uses a UUID to avoid permission conflicts during execution of tests in parallel
        self.test_file_name = TEST_FILE + str(uuid.uuid1())

        with Dataset(self.test_file_name, 'w') as nc:
            # Create data group
            g1 = nc.createGroup('data')

            # Add dimensions to data group
            g1.createDimension('num_chan_solar', 11)
            g1.createDimension('num_chan_thermal', 9)
            g1.createDimension('num_pixels', 72)
            g1.createDimension('num_lines', 600)

            # Create calibration_data group
            g1_1 = g1.createGroup('calibration_data')

            # Add variables to data/calibration_data group
            bt_a = g1_1.createVariable('bt_conversion_a', np.float32, dimensions=('num_chan_thermal',))
            bt_a[:] = np.arange(9)
            bt_b = g1_1.createVariable('bt_conversion_b', np.float32, dimensions=('num_chan_thermal',))
            bt_b[:] = np.arange(9)
            cw = g1_1.createVariable('channel_cw_thermal', np.float32, dimensions=('num_chan_thermal',))
            cw[:] = np.arange(9)
            isi = g1_1.createVariable('integrated_solar_irradiance', np.float32, dimensions=('num_chan_solar',))
            isi[:] = np.arange(11)

            # Create measurement_data group
            g1_2 = g1.createGroup('measurement_data')

            # Add dimensions to data/measurement_data group
            g1_2.createDimension('num_tie_points_act', 10)
            g1_2.createDimension('num_tie_points_alt', 100)

            # Add variables to data/measurement_data group
            sza = g1_2.createVariable('solar_zenith', np.float32,
                                      dimensions=('num_tie_points_act', 'num_tie_points_alt'))
            sza[:] = 25.0
            delta_lat = g1_2.createVariable('delta_lat', np.float32, dimensions=('num_pixels', 'num_lines'))
            delta_lat[:] = 1.0

        self.reader = ViiL1bNCFileHandler(
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

    def test_calibration_functions(self):
        """Test the calibration functions."""
        radiance = np.array([[1.0, 2.0, 5.0], [7.0, 10.0, 20.0]])

        cw = 13.0
        a = 3.0
        b = 100.0
        bt = self.reader._calibrate_bt(radiance, cw, a, b)
        expected_bt = np.array([[675.04993213, 753.10301462, 894.93149648],
                                [963.20401882, 1048.95086402, 1270.95546218]])
        self.assertTrue(np.allclose(bt, expected_bt))

        angle_factor = 0.4
        isi = 2.0
        refl = self.reader._calibrate_refl(radiance, angle_factor, isi)
        expected_refl = np.array([[0.628318531, 1.256637061, 3.141592654],
                                  [4.398229715, 6.283185307, 12.56637061]])
        self.assertTrue(np.allclose(refl, expected_refl))

    def test_functions(self):
        """Test the functions."""
        # Checks that the _perform_orthorectification function is correctly executed
        variable = xr.DataArray(
            dims=('num_pixels', 'num_lines'),
            name='test_name',
            attrs={
                'key_1': 'value_1',
                'key_2': 'value_2'
            },
            data=da.from_array(np.ones((72, 600)))
        )

        orthorect_variable = self.reader._perform_orthorectification(variable, 'data/measurement_data/delta_lat')
        expected_values = np.degrees(np.ones((72, 600)) / MEAN_EARTH_RADIUS) + np.ones((72, 600))
        self.assertTrue(np.allclose(orthorect_variable.values, expected_values))

        # Checks that the _perform_calibration function is correctly executed in all cases
        # radiance calibration: return value is simply a copy of the variable
        return_variable = self.reader._perform_calibration(variable, {'calibration': 'radiance'})
        self.assertTrue(np.all(return_variable == variable))

        # invalid calibration: raises a ValueError
        with self.assertRaises(ValueError):
            self.reader._perform_calibration(variable,
                                             {'calibration': 'invalid', 'name': 'test'})

        # brightness_temperature calibration: checks that the return value is correct
        calibrated_variable = self.reader._perform_calibration(variable,
                                                               {'calibration': 'brightness_temperature',
                                                                'chan_thermal_index': 3})
        expected_values = np.ones((72, 600)) * 1101.103383
        self.assertTrue(np.allclose(calibrated_variable.values, expected_values))

        # reflectance calibration: checks that the return value is correct
        calibrated_variable = self.reader._perform_calibration(variable,
                                                               {'calibration': 'reflectance', 'chan_solar_index': 2})
        expected_values = np.ones((72, 600)) * 1.733181982
        self.assertTrue(np.allclose(calibrated_variable.values, expected_values))
