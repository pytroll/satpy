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

"""The vii_base_nc reader tests package."""

import datetime
import os
import unittest
import uuid
from unittest import mock

import numpy as np
import pytest
import xarray as xr
from netCDF4 import Dataset

from satpy.readers.core.vii_nc import SCAN_ALT_TIE_POINTS, TIE_POINTS_FACTOR, ViiNCBaseFileHandler

TEST_FILE = "test_file_vii_base_nc.nc"


class TestViiNCBaseFileHandler(unittest.TestCase):
    """Test the ViiNCBaseFileHandler reader."""

    @mock.patch("satpy.readers.core.vii_nc.ViiNCBaseFileHandler._perform_geo_interpolation")
    def setUp(self, pgi_):
        """Set up the test."""
        # Easiest way to test the reader is to create a test netCDF file on the fly
        # uses a UUID to avoid permission conflicts during execution of tests in parallel
        self.test_file_name = TEST_FILE + str(uuid.uuid1()) + ".nc"

        with Dataset(self.test_file_name, "w") as nc:
            # Add global attributes
            nc.sensing_start_time_utc = "20170920173040.888"
            nc.sensing_end_time_utc = "20170920174117.555"
            nc.spacecraft = "test_spacecraft"
            nc.instrument = "test_instrument"

            # Create data group
            g1 = nc.createGroup("data")

            # Add dimensions to data group
            g1.createDimension("num_pixels", 10)
            g1.createDimension("num_lines", 100)

            # Create data/measurement_data group
            g1_1 = g1.createGroup("measurement_data")

            # Add dimensions to data/measurement_data group
            g1_1.createDimension("num_tie_points_act", 10)
            g1_1.createDimension("num_tie_points_alt", 100)

            # Add variables to data/measurement_data group
            tpw = g1_1.createVariable("tpw", np.float32, dimensions=("num_pixels", "num_lines"))
            tpw[:] = 1.
            tpw.test_attr = "attr"
            lon = g1_1.createVariable("longitude",
                                      np.float32,
                                      dimensions=("num_tie_points_act", "num_tie_points_alt"))
            lon[:] = 100.
            lat = g1_1.createVariable("latitude",
                                      np.float32,
                                      dimensions=("num_tie_points_act", "num_tie_points_alt"))
            lat[:] = 10.

            # Create quality group
            g2 = nc.createGroup("quality")

            # Add dimensions to quality group
            g2.createDimension("gap_items", 2)

            # Add variables to quality group
            var = g2.createVariable("duration_of_product", np.double, dimensions=())
            var[:] = 1.0
            var = g2.createVariable("duration_of_data_present", np.double, dimensions=())
            var[:] = 2.0
            var = g2.createVariable("duration_of_data_missing", np.double, dimensions=())
            var[:] = 3.0
            var = g2.createVariable("duration_of_data_degraded", np.double, dimensions=())
            var[:] = 4.0
            var = g2.createVariable("gap_start_time_utc", np.double, dimensions=("gap_items",))
            var[:] = [5.0, 6.0]
            var = g2.createVariable("gap_end_time_utc", np.double, dimensions=("gap_items",))
            var[:] = [7.0, 8.0]

        # Create longitude and latitude "interpolated" arrays
        interp_longitude = xr.DataArray(np.ones((10, 100)))
        interp_latitude = xr.DataArray(np.ones((10, 100)) * 2.)
        pgi_.return_value = (interp_longitude, interp_latitude)

        # Filename info valid for all readers
        filename_info = {
            "creation_time": datetime.datetime(year=2017, month=9, day=22,
                                               hour=22, minute=40, second=10),
            "sensing_start_time": datetime.datetime(year=2017, month=9, day=20,
                                                    hour=12, minute=30, second=30),
            "sensing_end_time": datetime.datetime(year=2017, month=9, day=20,
                                                  hour=18, minute=30, second=50)
        }

        # Create a reader
        self.reader = ViiNCBaseFileHandler(
            filename=self.test_file_name,
            filename_info=filename_info,
            filetype_info={
                "cached_longitude": "data/measurement_data/longitude",
                "cached_latitude": "data/measurement_data/latitude"
            }
        )

        # Create a second reader where orthorectification and interpolation are inhibited
        # by means of the filetype_info flags
        self.reader_2 = ViiNCBaseFileHandler(
            filename=self.test_file_name,
            filename_info=filename_info,
            filetype_info={
                "cached_longitude": "data/measurement_data/longitude",
                "cached_latitude": "data/measurement_data/latitude",
                "interpolate": False,
                "orthorect": False
            },
            orthorect=True
        )

        # Create a third reader without defining cached latitude and longitude
        # by means of the filetype_info flags
        self.reader_3 = ViiNCBaseFileHandler(
            filename=self.test_file_name,
            filename_info=filename_info,
            filetype_info={},
            orthorect=True
        )

    def tearDown(self):
        """Remove the previously created test file."""
        # Catch Windows PermissionError for removing the created test file.
        try:
            os.remove(self.test_file_name)
        except OSError:
            pass

    def test_file_reading(self):
        """Test the file product reading."""
        # Checks that the basic functionalities are correctly executed
        expected_start_time = datetime.datetime(year=2017, month=9, day=20,
                                                hour=17, minute=30, second=40, microsecond=888000)
        assert self.reader.start_time == expected_start_time

        expected_end_time = datetime.datetime(year=2017, month=9, day=20,
                                              hour=17, minute=41, second=17, microsecond=555000)
        assert self.reader.end_time == expected_end_time

        assert self.reader.spacecraft_name == "test_spacecraft"
        assert self.reader.sensor == "test_instrument"
        assert self.reader.ssp_lon is None

        # Checks that the global attributes are correctly read
        expected_global_attributes = {
            "filename": self.test_file_name,
            "start_time": expected_start_time,
            "end_time": expected_end_time,
            "spacecraft_name": "test_spacecraft",
            "ssp_lon": None,
            "sensor": "test_instrument",
            "filename_start_time": datetime.datetime(year=2017, month=9, day=20,
                                                     hour=12, minute=30, second=30),
            "filename_end_time": datetime.datetime(year=2017, month=9, day=20,
                                                   hour=18, minute=30, second=50),
            "platform_name": "test_spacecraft",
            "quality_group": {
                "duration_of_product": 1.,
                "duration_of_data_present": 2.,
                "duration_of_data_missing": 3.,
                "duration_of_data_degraded": 4.,
                "gap_start_time_utc": (5., 6.),
                "gap_end_time_utc": (7., 8.)
            }
        }

        global_attributes = self.reader._get_global_attributes()
        # Since the global_attributes dictionary contains numpy arrays,
        # it is not possible to peform a simple equality test
        # Must iterate on all keys to confirm that the dictionaries are equal
        assert global_attributes.keys() == expected_global_attributes.keys()
        for key in expected_global_attributes:
            if key not in ["quality_group"]:
                # Quality check must be valid for both iterable and not iterable elements
                try:
                    equal = all(global_attributes[key] == expected_global_attributes[key])
                except (TypeError, ValueError):
                    equal = global_attributes[key] == expected_global_attributes[key]
                assert equal
            else:
                assert global_attributes[key].keys() == expected_global_attributes[key].keys()
                for inner_key in global_attributes[key]:
                    # Equality check must be valid for both iterable and not iterable elements
                    try:
                        equal = all(global_attributes[key][inner_key] == expected_global_attributes[key][inner_key])
                    except (TypeError, ValueError):
                        equal = global_attributes[key][inner_key] == expected_global_attributes[key][inner_key]
                    assert equal

    @mock.patch("satpy.readers.core.vii_nc.tie_points_interpolation")
    @mock.patch("satpy.readers.core.vii_nc.tie_points_geo_interpolation")
    def test_functions(self, tpgi_, tpi_):
        """Test the functions."""
        with pytest.raises(NotImplementedError):
            self.reader._perform_orthorectification(mock.Mock(), mock.Mock())

        with pytest.raises(NotImplementedError):
            self.reader._perform_calibration(mock.Mock(), mock.Mock())

        # Checks that the _perform_interpolation function is correctly executed
        variable = xr.DataArray(
            dims=("y", "x"),
            name="test_name",
            attrs={
                "key_1": "value_1",
                "key_2": "value_2"
            },
            data=np.zeros((10, 100)),
        )
        tpi_.return_value = [xr.DataArray(
            dims=("num_tie_points_act", "num_tie_points_alt"),
            data=np.ones((10, 100))
        )]

        return_value = self.reader._perform_interpolation(variable)

        tpi_.assert_called_with([variable], SCAN_ALT_TIE_POINTS, TIE_POINTS_FACTOR)
        assert np.allclose(return_value, np.ones((10, 100)))
        assert return_value.attrs == {"key_1": "value_1", "key_2": "value_2"}
        assert return_value.name == "test_name"
        assert return_value.dims == ("num_pixels", "num_lines")

        # Checks that the _perform_geo_interpolation function is correctly executed
        variable_lon = xr.DataArray(
            dims=("y", "x"),
            name="test_lon",
            attrs={
                "key_1": "value_lon_1",
                "key_2": "value_lon_2"
            },
            data=np.zeros((10, 100))
        )
        variable_lat = xr.DataArray(
            dims=("y", "x"),
            name="test_lat",
            attrs={
                "key_1": "value_lat_1",
                "key_2": "value_lat_2"
            },
            data=np.ones((10, 100)) * 2.
        )

        tpgi_.return_value = (
            xr.DataArray(
                dims=("num_tie_points_act", "num_tie_points_alt"),
                data=np.ones((10, 100))
            ),
            xr.DataArray(
                dims=("num_tie_points_act", "num_tie_points_alt"),
                data=6 * np.ones((10, 100))
            )
        )

        return_lon, return_lat = self.reader._perform_geo_interpolation(variable_lon, variable_lat)

        tpgi_.assert_called_with(variable_lon, variable_lat, SCAN_ALT_TIE_POINTS, TIE_POINTS_FACTOR)

        assert np.allclose(return_lon, np.ones((10, 100)))
        assert return_lon.attrs == {"key_1": "value_lon_1", "key_2": "value_lon_2"}
        assert return_lon.name == "test_lon"
        assert return_lon.dims == ("num_pixels", "num_lines")

        assert np.allclose(return_lat, 6 * np.ones((10, 100)))
        assert return_lat.attrs == {"key_1": "value_lat_1", "key_2": "value_lat_2"}
        assert return_lat.name == "test_lat"
        assert return_lat.dims == ("num_pixels", "num_lines")

    def test_standardize_dims(self):
        """Test the standardize dims function."""
        test_variable = xr.DataArray(
            dims=("num_pixels", "num_lines"),
            name="test_data",
            attrs={
                "key_1": "value_lat_1",
                "key_2": "value_lat_2"
            },
            data=np.ones((10, 100)) * 1.
        )
        out_variable = self.reader._standardize_dims(test_variable)
        assert np.allclose(out_variable.values, np.ones((100, 10)))
        assert out_variable.dims == ("y", "x")
        assert out_variable.attrs["key_1"] == "value_lat_1"

    @mock.patch("satpy.readers.core.vii_nc.ViiNCBaseFileHandler._perform_calibration")
    @mock.patch("satpy.readers.core.vii_nc.ViiNCBaseFileHandler._perform_interpolation")
    @mock.patch("satpy.readers.core.vii_nc.ViiNCBaseFileHandler._perform_orthorectification")
    def test_dataset(self, po_, pi_, pc_):
        """Test the execution of the get_dataset function."""
        # Checks the correct execution of the get_dataset function with a valid file_key
        variable = self.reader.get_dataset(None, {"file_key": "data/measurement_data/tpw",
                                                  "calibration": None})
        pc_.assert_not_called()
        pi_.assert_not_called()
        po_.assert_not_called()

        assert np.allclose(variable.values, np.ones((100, 10)))
        assert variable.dims == ("y", "x")
        assert variable.attrs["test_attr"] == "attr"
        assert variable.attrs["units"] is None

        # Checks the correct execution of the get_dataset function with a valid file_key
        # and required calibration and interpolation
        self.reader.get_dataset(None, {"file_key": "data/measurement_data/tpw",
                                       "calibration": "reflectance",
                                       "interpolate": True,
                                       "standard_name": "longitude"})
        pc_.assert_called()
        pi_.assert_called()
        po_.assert_not_called()

        # Checks the correct execution of the get_dataset function with a valid file_key
        # and required orthorectification
        self.reader.orthorect = True
        self.reader.get_dataset(None, {"file_key": "data/measurement_data/tpw",
                                       "calibration": None,
                                       "orthorect_data": "test_orthorect_data"})
        po_.assert_called()

        # Checks the correct execution of the get_dataset function with an invalid file_key
        invalid_dataset = self.reader.get_dataset(None, {"file_key": "test_invalid", "calibration": None})
        # Checks that the function returns None
        assert invalid_dataset is None

        pc_.reset_mock()
        pi_.reset_mock()
        po_.reset_mock()

        # Checks the correct execution of the get_dataset function with a 'cached_longitude' file_key
        longitude = self.reader.get_dataset(None, {"file_key": "cached_longitude",
                                                   "calibration": "reflectance",
                                                   "interpolate": True})
        pc_.assert_not_called()
        pi_.assert_not_called()
        assert longitude[0, 0] == 1.0

        # Checks the correct execution of the get_dataset function with a 'cached_latitude' file_key
        latitude = self.reader.get_dataset(None, {"file_key": "cached_latitude",
                                                  "calibration": None})
        assert latitude[0, 0] == 2.0

        # Repeats some check with the reader where orthorectification and interpolation are inhibited
        # by means of the filetype_info flags

        pc_.reset_mock()
        pi_.reset_mock()
        po_.reset_mock()

        # Checks the correct execution of the get_dataset function with a valid file_key
        # and required calibration and interpolation
        self.reader_2.get_dataset(None, {"file_key": "data/measurement_data/tpw",
                                         "calibration": "reflectance",
                                         "interpolate": True,
                                         "standard_name": "longitude"})
        pc_.assert_called()
        pi_.assert_not_called()
        po_.assert_not_called()

        # Checks the correct execution of the get_dataset function with a valid file_key
        # and required orthorectification
        self.reader_2.get_dataset(None, {"file_key": "data/measurement_data/tpw",
                                         "calibration": None,
                                         "orthorect_data": "test_orthorect_data"})
        po_.assert_not_called()

        # Checks the correct execution of the get_dataset function with a 'cached_longitude' file_key
        longitude = self.reader_2.get_dataset(None, {"file_key": "cached_longitude",
                                                     "calibration": None})
        assert longitude[0, 0] == 100.0

        # Checks the correct execution of the get_dataset function with a 'cached_longitude' file_key
        # in a reader without defined longitude
        longitude = self.reader_3.get_dataset(None, {"file_key": "cached_longitude",
                                                     "calibration": "reflectance",
                                                     "interpolate": True})
        # Checks that the function returns None
        assert longitude is None
