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

"""The fci_cld_l2_nc reader tests package."""

import datetime
import os
import unittest
from contextlib import suppress
from unittest import mock

import numpy as np
from netCDF4 import Dataset

from satpy.readers.fci_l2_nc import FciL2NCFileHandler, FciL2NCSegmentFileHandler, PRODUCT_DATA_DURATION_MINUTES

TEST_FILE = 'test_file_fci_l2_nc.nc'
SEG_TEST_FILE = 'test_seg_file_fci_l2_nc.nc'
TEST_ERROR_FILE = 'test_error_file_fci_l2_nc.nc'
TEST_BYTE_FILE = 'test_byte_file_fci_l2_nc.nc'


class TestFciL2NCFileHandler(unittest.TestCase):
    """Test the FciL2NCFileHandler reader."""

    def setUp(self):
        """Set up the test by creating a test file and opening it with the reader."""
        # Easiest way to test the reader is to create a test netCDF file on the fly
        with Dataset(TEST_FILE, 'w') as nc:
            # Create dimensions
            nc.createDimension('number_of_columns', 10)
            nc.createDimension('number_of_rows', 100)
            nc.createDimension('maximum_number_of_layers', 2)

            # add global attributes
            nc.data_source = 'test_data_source'
            nc.platform = 'test_platform'
            nc.time_coverage_start = '20170920173040'
            nc.time_coverage_end = '20170920174117'

            # Add datasets
            x = nc.createVariable('x', np.float32, dimensions=('number_of_columns',))
            x.standard_name = 'projection_x_coordinate'
            x[:] = np.arange(10)

            y = nc.createVariable('y', np.float32, dimensions=('number_of_rows',))
            x.standard_name = 'projection_y_coordinate'
            y[:] = np.arange(100)

            one_layer_dataset = nc.createVariable('test_one_layer', np.float32,
                                                  dimensions=('number_of_rows', 'number_of_columns'))
            one_layer_dataset[:] = np.ones((100, 10))
            one_layer_dataset.test_attr = 'attr'
            one_layer_dataset.units = 'test_units'

            two_layers_dataset = nc.createVariable('test_two_layers', np.float32,
                                                   dimensions=('maximum_number_of_layers',
                                                               'number_of_rows',
                                                               'number_of_columns'))
            two_layers_dataset[0, :, :] = np.ones((100, 10))
            two_layers_dataset[1, :, :] = 2 * np.ones((100, 10))

            mtg_geos_projection = nc.createVariable('mtg_geos_projection', int, dimensions=())
            mtg_geos_projection.longitude_of_projection_origin = 10.0
            mtg_geos_projection.semi_major_axis = 6378137.
            mtg_geos_projection.semi_minor_axis = 6356752.
            mtg_geos_projection.perspective_point_height = 35786400.

        self.reader = FciL2NCFileHandler(
            filename=TEST_FILE,
            filename_info={
                'creation_time':  datetime.datetime(year=2017, month=9, day=20,
                                                    hour=12, minute=30, second=30),
            },
            filetype_info={}
        )

    def tearDown(self):
        """Remove the previously created test file."""
        # First delete the reader, forcing the file to be closed if still open
        del self.reader
        # Then can safely remove it from the system
        with suppress(OSError):
            os.remove(TEST_FILE)

    def test_all_basic(self):
        """Test all basic functionalities."""
        self.assertEqual(PRODUCT_DATA_DURATION_MINUTES, 20)

        self.assertEqual(self.reader._start_time,
                         datetime.datetime(year=2017, month=9, day=20,
                                           hour=17, minute=30, second=40))

        self.assertEqual(self.reader._end_time,
                         datetime.datetime(year=2017, month=9, day=20,
                                           hour=17, minute=41, second=17))

        self.assertEqual(self.reader._spacecraft_name, 'test_platform')
        self.assertEqual(self.reader._sensor_name, 'test_data_source')
        self.assertEqual(self.reader.ssp_lon, 10.0)

        global_attributes = self.reader._get_global_attributes()
        expected_global_attributes = {
            'filename': TEST_FILE,
            'start_time': datetime.datetime(year=2017, month=9, day=20,
                                            hour=17, minute=30, second=40),
            'end_time': datetime.datetime(year=2017, month=9, day=20,
                                          hour=17, minute=41, second=17),
            'spacecraft_name': 'test_platform',
            'ssp_lon': 10.0,
            'sensor': 'test_data_source',
            'creation_time': datetime.datetime(year=2017, month=9, day=20,
                                               hour=12, minute=30, second=30),
            'platform_name': 'test_platform'
        }
        self.assertEqual(global_attributes, expected_global_attributes)

    @mock.patch('satpy.readers.fci_l2_nc.get_area_definition')
    @mock.patch('satpy.readers.fci_l2_nc.make_ext')
    def test_area_definition(self, me_, gad_):
        """Test the area definition computation."""
        self.reader._compute_area_def()

        # Asserts that the make_ext function was called with the correct arguments
        me_.assert_called_once()
        name, args, kwargs = me_.mock_calls[0]
        self.assertTrue(np.allclose(args[0], 0.0))
        self.assertTrue(np.allclose(args[1], 515.6620))
        self.assertTrue(np.allclose(args[2], 0.0))
        self.assertTrue(np.allclose(args[3], 5672.28217))
        self.assertTrue(np.allclose(args[4], 35786400.))

        p_dict = {
            'nlines': 100,
            'ncols': 10,
            'ssp_lon': 10.0,
            'a': 6378137.,
            'b': 6356752.,
            'h': 35786400.,
            'a_name': 'FCI Area',
            'a_desc': 'Area for FCI instrument',
            'p_id': 'geos'
        }

        # Asserts that the get_area_definition function was called with the correct arguments
        gad_.assert_called_once()
        name, args, kwargs = gad_.mock_calls[0]
        self.assertEqual(args[0], p_dict)
        # The second argument must be the return result of the make_ext function
        self.assertEqual(args[1]._extract_mock_name(), 'make_ext()')

    def test_dataset(self):
        """Test the execution of the get_dataset function."""
        # Checks the correct execution of the get_dataset function with a valid file_key
        dataset = self.reader.get_dataset(None,
                                          {'file_key': 'test_one_layer',
                                           'fill_value': -999, 'mask_value': 0.,
                                           'file_type': 'test_file_type'})

        self.assertTrue(np.allclose(dataset.values, np.ones((100, 10))))
        self.assertEqual(dataset.attrs['test_attr'], 'attr')
        self.assertEqual(dataset.attrs['units'], 'test_units')
        self.assertEqual(dataset.attrs['fill_value'], -999)

        # Checks the correct execution of the get_dataset function with a valid file_key & layer
        dataset = self.reader.get_dataset(None,
                                          {'file_key': 'test_two_layers', 'layer': 1,
                                           'fill_value': -999, 'mask_value': 0,
                                           'file_type': 'test_file_type'})
        self.assertTrue(np.allclose(dataset.values, 2 * np.ones((100, 10))))
        self.assertEqual(dataset.attrs['units'], None)
        self.assertEqual(dataset.attrs['spacecraft_name'], 'test_platform')

        # Checks the correct execution of the get_dataset function with an invalid file_key
        invalid_dataset = self.reader.get_dataset(None,
                                                  {'file_key': 'test_invalid',
                                                   'fill_value': -999, 'mask_value': 0,
                                                   'file_type': 'test_file_type'})
        # Checks that the function returns None
        self.assertEqual(invalid_dataset, None)


class TestFciL2NCSegmentFileHandler(unittest.TestCase):
    """Test the FciL2NCSegmentFileHandler reader."""

    def setUp(self):
        """Set up the test by creating a test file and opening it with the reader."""
        # Easiest way to test the reader is to create a test netCDF file on the fly
        with Dataset(SEG_TEST_FILE, 'w') as nc:
            # Create dimensions
            nc.createDimension('number_of_FoR_cols', 10)
            nc.createDimension('number_of_FoR_rows', 100)
            nc.createDimension('number_of_channels', 8)
            nc.createDimension('number_of_categories', 6)

            # add global attributes
            nc.data_source = 'test_fci_data_source'
            nc.platform = 'test_fci_platform'
            nc.time_coverage_start = '20170920173040'
            nc.time_coverage_end = '20170920174117'

            # Add datasets
            x = nc.createVariable('x', np.float32, dimensions=('number_of_FoR_cols',))
            x.standard_name = 'projection_x_coordinate'
            x[:] = np.arange(10)

            y = nc.createVariable('y', np.float32, dimensions=('number_of_FoR_rows',))
            x.standard_name = 'projection_y_coordinate'
            y[:] = np.arange(100)

            chans = nc.createVariable('channels', np.float32, dimensions=('number_of_channels',))
            chans.standard_name = 'fci_channels'
            chans[:] = np.arange(8)

            cats = nc.createVariable('categories', np.float32, dimensions=('number_of_categories',))
            cats.standard_name = 'product_categories'
            cats[:] = np.arange(6)

            test_dataset = nc.createVariable('test_values', np.float32,
                                             dimensions=('number_of_FoR_rows', 'number_of_FoR_cols',
                                                         'number_of_channels', 'number_of_categories'))
            test_dataset[:] = np.ones((100, 10, 8, 6))
            test_dataset.test_attr = 'attr'
            test_dataset.units = 'test_units'

        self.segment_reader = FciL2NCSegmentFileHandler(
            filename=SEG_TEST_FILE,
            filename_info={
                'creation_time':  datetime.datetime(year=2017, month=9, day=20,
                                                    hour=12, minute=30, second=30),
            },
            filetype_info={}
        )

    def tearDown(self):
        """Remove the previously created test file."""
        # First delete the reader, forcing the file to be closed if still open
        del self.segment_reader
        # Then can safely remove it from the system
        with suppress(OSError):
            os.remove(SEG_TEST_FILE)

    def test_all_basic(self):
        """Test all basic functionalities."""
        self.assertEqual(PRODUCT_DATA_DURATION_MINUTES, 20)

        self.assertEqual(self.segment_reader._start_time,
                         datetime.datetime(year=2017, month=9, day=20,
                                           hour=17, minute=30, second=40))

        self.assertEqual(self.segment_reader._end_time,
                         datetime.datetime(year=2017, month=9, day=20,
                                           hour=17, minute=41, second=17))

        self.assertEqual(self.segment_reader._spacecraft_name, 'test_fci_platform')
        self.assertEqual(self.segment_reader._sensor_name, 'test_fci_data_source')
        self.assertEqual(self.segment_reader.ssp_lon, 0.0)

        global_attributes = self.segment_reader._get_global_attributes()

        expected_global_attributes = {
            'filename': SEG_TEST_FILE,
            'start_time': datetime.datetime(year=2017, month=9, day=20,
                                            hour=17, minute=30, second=40),
            'end_time': datetime.datetime(year=2017, month=9, day=20,
                                          hour=17, minute=41, second=17),
            'spacecraft_name': 'test_fci_platform',
            'ssp_lon': 0.0,
            'sensor': 'test_fci_data_source',
            'creation_time': datetime.datetime(year=2017, month=9, day=20,
                                               hour=12, minute=30, second=30),
            'platform_name': 'test_fci_platform'
        }
        self.assertEqual(global_attributes, expected_global_attributes)

    def test_dataset(self):
        """Test the execution of the get_dataset function."""
        # Checks the correct execution of the get_dataset function with a valid file_key
        dataset = self.segment_reader.get_dataset(None,
                                                  {'file_key': 'test_values',
                                                   'fill_value': -999, 'mask_value': 0, })
        self.assertTrue(np.allclose(dataset.values, np.ones((100, 10, 8, 6))))
        self.assertEqual(dataset.attrs['test_attr'], 'attr')
        self.assertEqual(dataset.attrs['units'], 'test_units')
        self.assertEqual(dataset.attrs['fill_value'], -999)

        # Checks the correct execution of the get_dataset function with an invalid file_key
        invalid_dataset = self.segment_reader.get_dataset(None,
                                                          {'file_key': 'test_invalid',
                                                           'fill_value': -999, 'mask_value': 0})
        # Checks that the function returns None
        self.assertEqual(invalid_dataset, None)


class TestFciL2NCErrorFileHandler(unittest.TestCase):
    """Test the FciL2NCFileHandler reader."""

    def setUp(self):
        """Set up the test by creating a test file and opening it with the reader."""
        # Easiest way to test the reader is to create a test netCDF file on the fly

        with Dataset(TEST_ERROR_FILE, 'w') as nc_err:
            # Create dimensions
            nc_err.createDimension('number_of_FoR_cols', 10)
            nc_err.createDimension('number_of_FoR_rows', 100)
            nc_err.createDimension('number_of_channels', 8)
            nc_err.createDimension('number_of_categories', 6)
            # add erroneous global attributes
            nc_err.data_source = 'test_fci_data_source'  # Error in key name
            nc_err.platform_err = 'test_fci_platform'  # Error in key name
            nc_err.time_coverage_start = '2017092017304000'  # Error in time format
            nc_err.time_coverage_end_err = '20170920174117'  # Error in key name

            # Add datasets
            x = nc_err.createVariable('x', np.float32, dimensions=('number_of_FoR_cols',))
            x.standard_name = 'projection_x_coordinate'
            x[:] = np.arange(10)

            y = nc_err.createVariable('y', np.float32, dimensions=('number_of_FoR_rows',))
            x.standard_name = 'projection_y_coordinate'
            y[:] = np.arange(100)

            chans = nc_err.createVariable('channels', np.float32, dimensions=('number_of_channels',))
            chans.standard_name = 'fci_channels'
            chans[:] = np.arange(8)

            cats = nc_err.createVariable('categories', np.float32, dimensions=('number_of_categories',))
            cats.standard_name = 'product_categories'
            cats[:] = np.arange(6)

            test_dataset = nc_err.createVariable('test_values', np.float32,
                                                 dimensions=('number_of_FoR_rows', 'number_of_FoR_cols',
                                                             'number_of_channels', 'number_of_categories'))
            test_dataset[:] = np.ones((100, 10, 8, 6))
            test_dataset.test_attr = 'attr'
            test_dataset.units = 'test_units'

        self.error_reader = FciL2NCSegmentFileHandler(
            filename=TEST_ERROR_FILE,
            filename_info={
                'creation_time': datetime.datetime(year=2017, month=9, day=20,
                                                   hour=12, minute=30, second=30),
            },
            filetype_info={}
        )

    def tearDown(self):
        """Remove the previously created test file."""
        # First delete the reader, forcing the file to be closed if still open
        del self.error_reader
        # Then can safely remove it from the system
        with suppress(OSError):
            os.remove(TEST_ERROR_FILE)

    def test_errors(self):
        """Test that certain properties cause errors."""
        self.assertRaises(TypeError, self.error_reader._start_time,
                          datetime.datetime(year=2017, month=9, day=20,
                                            hour=17, minute=30, second=40))

        self.assertRaises(TypeError, self.error_reader._end_time,
                          datetime.datetime(year=2017, month=9, day=20,
                                            hour=17, minute=41, second=17))

        self.assertRaises(TypeError, self.error_reader._spacecraft_name)

        self.assertRaises(TypeError, self.error_reader._sensor_name)


class TestFciL2NCReadingByteData(unittest.TestCase):
    """Test the FciL2NCFileHandler when reading and extracting byte data."""

    def setUp(self):
        """Set up the test by creating a test file and opening it with the reader."""
        # Easiest way to test the reader is to create a test netCDF file on the fly

        with Dataset(TEST_BYTE_FILE, 'w') as nc_byte:
            # Create dimensions
            nc_byte.createDimension('number_of_columns', 1)
            nc_byte.createDimension('number_of_rows', 1)

            # Add datasets
            x = nc_byte.createVariable('x', np.float32, dimensions=('number_of_columns',))
            x.standard_name = 'projection_x_coordinate'
            x[:] = np.arange(1)

            y = nc_byte.createVariable('y', np.float32, dimensions=('number_of_rows',))
            x.standard_name = 'projection_y_coordinate'
            y[:] = np.arange(1)

            mtg_geos_projection = nc_byte.createVariable('mtg_geos_projection', int, dimensions=())
            mtg_geos_projection.longitude_of_projection_origin = 10.0
            mtg_geos_projection.semi_major_axis = 6378137.
            mtg_geos_projection.semi_minor_axis = 6356752.
            mtg_geos_projection.perspective_point_height = 35786400.

            test_dataset = nc_byte.createVariable('cloud_mask_test_flag', np.float32,
                                                  dimensions=('number_of_rows', 'number_of_columns',))

            # This number was chosen as we know the expected byte values
            test_dataset[:] = 4544767

        self.byte_reader = FciL2NCFileHandler(
            filename=TEST_BYTE_FILE,
            filename_info={
                'creation_time': datetime.datetime(year=2017, month=9, day=20,
                                                   hour=12, minute=30, second=30),
            },
            filetype_info={}
        )

    def tearDown(self):
        """Remove the previously created test file."""
        # First delete the reader, forcing the file to be closed if still open
        del self.byte_reader
        # Then can safely remove it from the system
        with suppress(OSError):
            os.remove(TEST_BYTE_FILE)

    def test_byte_extraction(self):
        """Test the execution of the get_dataset function."""
        # Value of 1 is expected to be returned for this test
        dataset = self.byte_reader.get_dataset(None,
                                               {'file_key': 'cloud_mask_test_flag',
                                                'fill_value': -999, 'mask_value': 0.,
                                                'file_type': 'nc_fci_test_clm',
                                                'extract_byte': 1,
                                                })

        self.assertEqual(dataset.values, 1)

        # Value of 0 is expected fto be returned or this test
        dataset = self.byte_reader.get_dataset(None,
                                               {'file_key': 'cloud_mask_test_flag',
                                                'fill_value': -999, 'mask_value': 0.,
                                                'file_type': 'nc_fci_test_clm',
                                                'extract_byte': 23,
                                                })

        self.assertEqual(dataset.values, 0)
