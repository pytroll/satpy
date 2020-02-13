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

"""The vii_base_nc reader tests package."""

import os
import numpy as np
import xarray as xr
import datetime
from netCDF4 import Dataset

from satpy import CHUNK_SIZE
from satpy.readers.vii_base_nc import ViiNCBaseFileHandler

import unittest

try:
    from unittest import mock
except ImportError:
    import mock


class Test_ViiNCBaseFileHandler(unittest.TestCase):
    """Test the ViiNCBaseFileHandler reader."""

    def setUp(self):
        """Set up the test."""
        # Easiest way to test the reader is to create a test netCDF file on the fly
        with Dataset('test.nc', 'w') as nc:
            # Add global attributes
            nc.sensing_start_time_utc = "20170920173040.888"
            nc.sensing_end_time_utc = "20170920174117.555"
            nc.spacecraft = "test_spacecraft"
            nc.instrument = "test_instrument"

            # Create data group
            g1 = nc.createGroup('data')

            # Add dimensions to data group
            g1.createDimension('num_pixels', 10)
            g1.createDimension('num_lines', 100)

            # Create data/measurement_data group
            g1_1 = g1.createGroup('measurement_data')

            # Add dimensions to data/measurement_data group
            g1_1.createDimension('num_tie_points_act', 10)
            g1_1.createDimension('num_tie_points_alt', 100)

            # Add variables to data/measurement_data group
            tpw = g1_1.createVariable('tpw', np.float32, dimensions=('num_pixels', 'num_lines'))
            tpw[:] = 1.
            tpw.test_attr = 'attr'
            g1_1.createVariable('latitude', np.float32, dimensions=('num_tie_points_act', 'num_tie_points_alt'))
            g1_1.createVariable('longitude', np.float32, dimensions=('num_tie_points_act', 'num_tie_points_alt'))

            # Create quality group
            g2 = nc.createGroup('quality')

            # Add dimensions to quality group
            g2.createDimension('gap_items', 2)

            # Add variables to quality group
            var = g2.createVariable('duration_of_product', np.double, dimensions=())
            var[:] = 1.0
            var = g2.createVariable('duration_of_data_present', np.double, dimensions=())
            var[:] = 2.0
            var = g2.createVariable('duration_of_data_missing', np.double, dimensions=())
            var[:] = 3.0
            var = g2.createVariable('duration_of_data_degraded', np.double, dimensions=())
            var[:] = 4.0
            var = g2.createVariable('gap_start_time_utc', np.double, dimensions=('gap_items',))
            var[:] = [5.0, 6.0]
            var = g2.createVariable('gap_end_time_utc', np.double, dimensions=('gap_items',))
            var[:] = [7.0, 8.0]

        self.reader = ViiNCBaseFileHandler(
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

        # Creates a second reader where orthorectification and interpolation are inhibited
        # by means of the filetype_info flags
        self.reader_2 = ViiNCBaseFileHandler(
            filename='test.nc',
            filename_info={
                'creation_time': datetime.datetime(year=2017, month=9, day=22,
                                                   hour=22, minute=40, second=10),
                'sensing_start_time': datetime.datetime(year=2017, month=9, day=20,
                                                        hour=12, minute=30, second=30),
                'sensing_end_time': datetime.datetime(year=2017, month=9, day=20,
                                                      hour=18, minute=30, second=50)
                },
            filetype_info={
                'interpolate': False,
                'orthorect': False
                },
            orthorect=True
            )

    def tearDown(self):
        """Remove the previously created test file."""
        os.remove('test.nc')

    def test_file_reading(self):
        """Test the file product reading."""
        # Checks that the basic functionalities are correctly executed
        expected_start_time = datetime.datetime(year=2017, month=9, day=20,
                                                hour=17, minute=30, second=40, microsecond=888000)
        self.assertEqual(self.reader.start_time, expected_start_time)

        expected_end_time = datetime.datetime(year=2017, month=9, day=20,
                                              hour=17, minute=41, second=17, microsecond=555000)
        self.assertEqual(self.reader.end_time, expected_end_time)

        self.assertEqual(self.reader.spacecraft_name, "test_spacecraft")
        self.assertEqual(self.reader.sensor, "test_instrument")
        self.assertEqual(self.reader.ssp_lon, None)

        # Checks that the global attributes are correctly read
        expected_global_attributes = {
            'filename': "test.nc",
            'start_time': expected_start_time,
            'end_time': expected_end_time,
            'spacecraft_name': "test_spacecraft",
            'ssp_lon': None,
            'sensor': "test_instrument",
            'filename_start_time': datetime.datetime(year=2017, month=9, day=20,
                                                     hour=12, minute=30, second=30),
            'filename_end_time': datetime.datetime(year=2017, month=9, day=20,
                                                   hour=18, minute=30, second=50),
            'platform_name': "test_spacecraft",
            'quality_group': {
                   'duration_of_product': 1.,
                   'duration_of_data_present': 2.,
                   'duration_of_data_missing': 3.,
                   'duration_of_data_degraded': 4.,
                   'gap_start_time_utc': (5., 6.),
                   'gap_end_time_utc': (7., 8.)
                }
            }

        global_attributes = self.reader._get_global_attributes()
        # Since the global_attributes dictionary contains numpy arrays,
        # it is not possible to peform a simple equality test
        # Must iterate on all keys to confirm that the dictionaries are equal
        self.assertEqual(global_attributes.keys(), expected_global_attributes.keys())
        for key in expected_global_attributes:
            if key not in ['quality_group']:
                # Quality check must be valid for both iterable and not iterable elements
                try:
                    equal = all(global_attributes[key] == expected_global_attributes[key])
                except (TypeError, ValueError):
                    equal = global_attributes[key] == expected_global_attributes[key]
                self.assertTrue(equal)
            else:
                self.assertEqual(global_attributes[key].keys(), expected_global_attributes[key].keys())
                for inner_key in global_attributes[key]:
                    # Equality check must be valid for both iterable and not iterable elements
                    try:
                        equal = all(global_attributes[key][inner_key] == expected_global_attributes[key][inner_key])
                    except (TypeError, ValueError):
                        equal = global_attributes[key][inner_key] == expected_global_attributes[key][inner_key]
                    self.assertTrue(equal)

    @mock.patch('satpy.readers.vii_base_nc.xr')
    @mock.patch('satpy.readers.vii_base_nc.da')
    def test_functions(self, da_, xr_):
        """Test the functions."""
        with self.assertRaises(NotImplementedError):
            self.reader._perform_orthorectification(mock.Mock(), mock.Mock())

        with self.assertRaises(NotImplementedError):
            self.reader._perform_calibration(mock.Mock(), mock.Mock())

        # Checks that the _perform_interpolation function is correctly executed
        variable = xr.DataArray(
            dims=('x', 'y'),
            name='test_name',
            attrs={
                    'key_1': 'value_1',
                    'key_2': 'value_2'
                },
            data=np.ones((10, 100))
        )
        self.reader._perform_interpolation(variable)

        # Checks that dask.array has been called with the correct arguments
        name, args, kwargs = da_.mock_calls[0]
        self.assertTrue(np.all(args[0] == np.ones((72, 600))))
        self.assertEqual(args[1], CHUNK_SIZE)

        # Checks that xarray.DataArray has been called with the correct arguments
        name, args, kwargs = xr_.mock_calls[0]
        self.assertEqual(kwargs['attrs'], {'key_1': 'value_1', 'key_2': 'value_2'})
        self.assertEqual(kwargs['name'], 'test_name')
        self.assertEqual(kwargs['dims'], ('num_pixels', 'num_lines'))
        # The 'data' argument must be the return value of dask.array
        self.assertEqual(kwargs['data']._extract_mock_name(), 'da.from_array()')

        da_.reset_mock()
        xr_.reset_mock()

        # Checks that the _perform_wrapping function is correctly executed
        variable = xr.DataArray(
            dims=('x', 'y'),
            name='test_name',
            attrs={
                'key_1': 'value_1',
                'key_2': 'value_2'
            },
            data=np.ones((10, 100))
        )
        variable.values[0, :] = 361.
        variable.values[1, :] = -359.
        self.reader._perform_wrapping(variable)

        # Checks that dask.array has been called with the correct arguments
        name, args, kwargs = da_.mock_calls[0]
        self.assertTrue(np.all(args[0] == np.ones((10, 100))))
        self.assertEqual(args[1], CHUNK_SIZE)

        # Checks that xarray.DataArray has been called with the correct arguments
        name, args, kwargs = xr_.mock_calls[0]
        self.assertEqual(kwargs['attrs'], {'key_1': 'value_1', 'key_2': 'value_2'})
        self.assertEqual(kwargs['name'], 'test_name')
        self.assertEqual(kwargs['dims'], ('x', 'y'))
        # The 'data' argument must be the return value of dask.array
        self.assertEqual(kwargs['data']._extract_mock_name(), 'da.from_array()')

    @mock.patch('satpy.readers.vii_base_nc.ViiNCBaseFileHandler._perform_calibration')
    @mock.patch('satpy.readers.vii_base_nc.ViiNCBaseFileHandler._perform_interpolation')
    @mock.patch('satpy.readers.vii_base_nc.ViiNCBaseFileHandler._perform_wrapping')
    @mock.patch('satpy.readers.vii_base_nc.ViiNCBaseFileHandler._perform_orthorectification')
    def test_dataset(self, po_, pw_, pi_, pc_):
        """Test the execution of the get_dataset function."""
        # Checks the correct execution of the get_dataset function with a valid file_key
        variable = self.reader.get_dataset(None, {'file_key': 'data/measurement_data/tpw',
                                                  'calibration': None})
        pc_.assert_not_called()
        pi_.assert_not_called()
        pw_.assert_not_called()
        po_.assert_not_called()

        self.assertTrue(np.all(variable.values == np.ones((10, 100))))
        self.assertEqual(variable.dims, ('num_pixels', 'num_lines'))
        self.assertEqual(variable.attrs['test_attr'], 'attr')

        # Checks the correct execution of the get_dataset function with a valid file_key
        # and required calibration, interpolation and wrapping
        self.reader.get_dataset(None, {'file_key': 'data/measurement_data/tpw',
                                       'calibration': 'reflectance',
                                       'interpolate': True,
                                       'standard_name': 'longitude'})
        pc_.assert_called()
        pi_.assert_called()
        pw_.assert_called()
        po_.assert_not_called()

        # Checks the correct execution of the get_dataset function with a valid file_key
        # and required orthorectification
        self.reader.orthorect = True
        self.reader.get_dataset(None, {'file_key': 'data/measurement_data/tpw',
                                       'calibration': None,
                                       'orthorect_data': 'test_orthorect_data'})
        po_.assert_called()

        # Checks the correct execution of the get_dataset function with an invalid file_key
        invalid_dataset = self.reader.get_dataset(None, {'file_key': 'test_invalid', 'calibration': None})
        # Checks that the function returns None
        self.assertEqual(invalid_dataset, None)

        # Repeats some check with the reader where orthorectification and interpolation are inhibited
        # by means of the filetype_info flags

        pc_.reset_mock()
        pi_.reset_mock()
        pw_.reset_mock()
        po_.reset_mock()

        # Checks the correct execution of the get_dataset function with a valid file_key
        # and required calibration, interpolation and wrapping
        self.reader_2.get_dataset(None, {'file_key': 'data/measurement_data/tpw',
                                         'calibration': 'reflectance',
                                         'interpolate': True,
                                         'standard_name': 'longitude'})
        pc_.assert_called()
        pi_.assert_not_called()
        pw_.assert_called()
        po_.assert_not_called()

        # Checks the correct execution of the get_dataset function with a valid file_key
        # and required orthorectification
        self.reader_2.get_dataset(None, {'file_key': 'data/measurement_data/tpw',
                                         'calibration': None,
                                         'orthorect_data': 'test_orthorect_data'})
        po_.assert_not_called()


def suite():
    """Build the test suite for test_scene."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()

    mysuite.addTest(loader.loadTestsFromTestCase(Test_ViiNCBaseFileHandler))

    return mysuite


if __name__ == '__main__':
    unittest.main()
