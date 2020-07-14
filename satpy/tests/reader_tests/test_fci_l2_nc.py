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

import os
import numpy as np
import datetime
from netCDF4 import Dataset

from satpy.readers.fci_l2_nc import FciL2NCFileHandler, PRODUCT_DATA_DURATION_MINUTES

import unittest

try:
    from unittest import mock
except ImportError:
    import mock


TEST_FILE = 'test_file_fci_l2_nc.nc'


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

            mtg_geos_projection = nc.createVariable('mtg_geos_projection', np.int, dimensions=())
            mtg_geos_projection.longitude_of_projection_origin = 10.0
            mtg_geos_projection.semi_major_axis = 6378137.
            mtg_geos_projection.semi_minor_axis = 6356752.
            mtg_geos_projection.perspective_point_height = 35786400.

        self.reader = FciL2NCFileHandler(
            filename=TEST_FILE,
            filename_info={
                'creation_time':  datetime.datetime(year=2017, month=9, day=20,
                                                    hour=12, minute=30, second=30)
            },
            filetype_info={}
        )

    def tearDown(self):
        """Remove the previously created test file."""
        # First delets the reader, forcing the file to be closed if still open
        del self.reader
        # Then can safely remove it from the system
        try:
            os.remove(TEST_FILE)
        except OSError:
            pass

    def test_all_basic(self):
        """Test all basic functionalities."""
        self.assertEqual(PRODUCT_DATA_DURATION_MINUTES, 20)

        self.assertEqual(self.reader.start_time,
                         datetime.datetime(year=2017, month=9, day=20,
                                           hour=17, minute=30, second=40))

        self.assertEqual(self.reader.end_time,
                         datetime.datetime(year=2017, month=9, day=20,
                                           hour=17, minute=41, second=17))

        self.assertEqual(self.reader.spacecraft_name, 'test_platform')
        self.assertEqual(self.reader.sensor, 'test_data_source')
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
                                           'fill_value': -999, 'mask_value': 0})
        self.assertTrue(np.allclose(dataset.values, np.ones((100, 10))))
        self.assertEqual(dataset.attrs['test_attr'], 'attr')
        self.assertEqual(dataset.attrs['units'], 'test_units')
        self.assertEqual(dataset.attrs['fill_value'], -999)

        # Checks the correct execution of the get_dataset function with a valid file_key & layer
        dataset = self.reader.get_dataset(None,
                                          {'file_key': 'test_two_layers', 'layer': 1,
                                           'fill_value': -999, 'mask_value': 0})
        self.assertTrue(np.allclose(dataset.values, 2 * np.ones((100, 10))))
        self.assertEqual(dataset.attrs['units'], None)
        self.assertEqual(dataset.attrs['spacecraft_name'], 'test_platform')

        # Checks the correct execution of the get_dataset function with an invalid file_key
        invalid_dataset = self.reader.get_dataset(None,
                                                  {'file_key': 'test_invalid',
                                                   'fill_value': -999, 'mask_value': 0})
        # Checks that the function returns None
        self.assertEqual(invalid_dataset, None)
