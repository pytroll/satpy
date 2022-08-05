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

"""SEVIRI L2 GRIB-reader test package."""

import datetime
import sys
import unittest
from unittest import mock

import numpy as np

from satpy.tests.utils import make_dataid

# Dictionary to be used as fake GRIB message
FAKE_MESSAGE = {
    'longitudeOfSubSatellitePointInDegrees': 9.5,
    'dataDate': 20191020,
    'dataTime': 1745,
    'Nx': 1000,
    'Ny': 1200,
    'earthMajorAxis': 6400.,
    'earthMinorAxis': 6300.,
    'NrInRadiusOfEarth': 6.,
    'XpInGridLengths': 500,
    'parameterNumber': 30,
    'missingValue': 9999,
}

# List to be used as fake GID source
FAKE_GID = [0, 1, 2, 3, None]


class Test_SeviriL2GribFileHandler(unittest.TestCase):
    """Test the SeviriL2GribFileHandler reader."""

    @mock.patch('satpy.readers.seviri_l2_grib.ec')
    def setUp(self, ec_):
        """Set up the test by creating a mocked eccodes library."""
        fake_gid_generator = (i for i in FAKE_GID)
        ec_.codes_grib_new_from_file.side_effect = lambda fh: next(fake_gid_generator)
        ec_.codes_get.side_effect = lambda gid, key: FAKE_MESSAGE[key]
        ec_.codes_get_values.return_value = np.ones(1000*1200)
        self.ec_ = ec_

    @unittest.skipIf(sys.platform.startswith('win'), "'eccodes' not supported on Windows")
    @mock.patch('satpy.readers.seviri_l2_grib.xr')
    @mock.patch('satpy.readers.seviri_l2_grib.da')
    def test_data_reading(self, da_, xr_):
        """Test the reading of data from the product."""
        from satpy import CHUNK_SIZE
        from satpy.readers.seviri_l2_grib import REPEAT_CYCLE_DURATION, SeviriL2GribFileHandler
        with mock.patch("builtins.open", mock.mock_open()) as mock_file:
            with mock.patch('satpy.readers.seviri_l2_grib.ec', self.ec_):
                self.reader = SeviriL2GribFileHandler(
                    filename='test.grib',
                    filename_info={
                        'spacecraft': 'MET11',
                        'start_time': datetime.datetime(year=2020, month=10, day=20,
                                                        hour=19, minute=45, second=0)
                    },
                    filetype_info={}
                )

                dataset_id = make_dataid(name='dummmy', resolution=3000)

                # Checks that the codes_grib_multi_support_on function has been called
                self.ec_.codes_grib_multi_support_on.assert_called()

                # Restarts the id generator and clears the call history
                fake_gid_generator = (i for i in FAKE_GID)
                self.ec_.codes_grib_new_from_file.side_effect = lambda fh: next(fake_gid_generator)
                self.ec_.codes_grib_new_from_file.reset_mock()
                self.ec_.codes_release.reset_mock()

                # Checks the correct execution of the get_dataset function with a valid parameter_number
                valid_dataset = self.reader.get_dataset(dataset_id, {'parameter_number': 30})
                # Checks the correct file open call
                mock_file.assert_called_with('test.grib', 'rb')
                # Checks that the dataset has been created as a DataArray object
                self.assertEqual(valid_dataset._extract_mock_name(), 'xr.DataArray()')
                # Checks that codes_release has been called after each codes_grib_new_from_file call
                # (except after the last one which has returned a None)
                self.assertEqual(self.ec_.codes_grib_new_from_file.call_count,
                                 self.ec_.codes_release.call_count + 1)

                # Restarts the id generator and clears the call history
                fake_gid_generator = (i for i in FAKE_GID)
                self.ec_.codes_grib_new_from_file.side_effect = lambda fh: next(fake_gid_generator)
                self.ec_.codes_grib_new_from_file.reset_mock()
                self.ec_.codes_release.reset_mock()

                # Checks the correct execution of the get_dataset function with an invalid parameter_number
                invalid_dataset = self.reader.get_dataset(dataset_id, {'parameter_number': 50})
                # Checks that the function returns None
                self.assertEqual(invalid_dataset, None)
                # Checks that codes_release has been called after each codes_grib_new_from_file call
                # (except after the last one which has returned a None)
                self.assertEqual(self.ec_.codes_grib_new_from_file.call_count,
                                 self.ec_.codes_release.call_count + 1)

                # Checks the basic data reading
                self.assertEqual(REPEAT_CYCLE_DURATION, 15)

                # Checks the correct execution of the _get_global_attributes and _get_metadata_from_msg functions
                attributes = self.reader._get_attributes()
                expected_attributes = {
                    'orbital_parameters': {
                        'projection_longitude': 9.5
                    },
                    'sensor': 'seviri',
                    'platform_name': 'Meteosat-11'
                }
                self.assertEqual(attributes, expected_attributes)

                # Checks the reading of an array from the message
                self.reader._get_xarray_from_msg(0)

                # Checks that dask.array has been called with the correct arguments
                name, args, kwargs = da_.mock_calls[0]
                self.assertTrue(np.all(args[0] == np.ones((1200, 1000))))
                self.assertEqual(args[1], CHUNK_SIZE)

                # Checks that xarray.DataArray has been called with the correct arguments
                name, args, kwargs = xr_.mock_calls[0]
                self.assertEqual(kwargs['dims'], ('y', 'x'))

                # Checks the correct execution of the _get_proj_area function
                pdict, area_dict = self.reader._get_proj_area(0)

                expected_pdict = {
                    'a': 6400000.,
                    'b': 6300000.,
                    'h': 32000000.,
                    'ssp_lon': 9.5,
                    'nlines': 1000,
                    'ncols': 1200,
                    'a_name': 'msg_seviri_rss_3km',
                    'a_desc': 'MSG SEVIRI Rapid Scanning Service area definition with 3 km resolution',
                    'p_id': '',
                }
                self.assertEqual(pdict, expected_pdict)
                expected_area_dict = {
                    'center_point': 500,
                    'north': 1200,
                    'east': 1,
                    'west': 1000,
                    'south': 1,
                }
                self.assertEqual(area_dict, expected_area_dict)

                # Checks the correct execution of the get_area_def function
                with mock.patch('satpy.readers.seviri_l2_grib.calculate_area_extent',
                                mock.Mock(name='calculate_area_extent')) as cae:
                    with mock.patch('satpy.readers.seviri_l2_grib.get_area_definition', mock.Mock()) as gad:
                        self.reader.get_area_def(mock.Mock(resolution=400.))
                        # Asserts that calculate_area_extent has been called with the correct arguments
                        expected_args = ({'center_point': 500, 'east': 1, 'west': 1000, 'south': 1, 'north': 1200,
                                         'column_step': 400., 'line_step': 400.},)
                        name, args, kwargs = cae.mock_calls[0]
                        self.assertEqual(args, expected_args)
                        # Asserts that get_area_definition has been called with the correct arguments
                        name, args, kwargs = gad.mock_calls[0]
                        self.assertEqual(args[0], expected_pdict)
                        # The second argument must be the return result of calculate_area_extent
                        self.assertEqual(args[1]._extract_mock_name(), 'calculate_area_extent()')
