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

"""The Thr3mi_l1c_nc reader tests package."""

import datetime
import os
import unittest
import uuid

import numpy as np
import xarray as xr
from netCDF4 import Dataset

from satpy.readers.thr3mi_l1c_nc import Thr3miL1cNCFileHandler

TEST_FILE = "test_file_thr3mi_l1c_nc.nc"


class TestThr3miNCL1cFileHandler(unittest.TestCase):
    """Test the Thr3miNCL1cFileHandler reader."""

    def setUp(self):
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

            nc.createDimension("overlaps", 1)

            # Create data group
            g1 = nc.createGroup("data")

            # Create data/measurement_data group
            g1_1 = g1.createGroup("overlap_000")

            # Add dimensions to data/measurement_data group
            g1_1.createDimension("geo_reference_grid_cells", 10)
            g1_1.createDimension("viewing_directions_VNIR", 3)

            g1_1_1 = g1_1.createGroup("measurement_data")
            g1_1_2 = g1_1.createGroup("geolocation_data")

            g1_1_1_1 = g1_1_1.createGroup("r_0865")

            # Add variables to data/measurement_data group
            reflectance_Q = g1_1_1_1.createVariable("reflectance_Q", np.float32,
                                                    dimensions=("geo_reference_grid_cells", "viewing_directions_VNIR"))
            reflectance_Q[:, 0] = 75.
            reflectance_Q[:, 1] = 76.
            reflectance_Q[:, 2] = 77.

            reflectance_Q.test_attr = "attr"

            lon = g1_1_2.createVariable("longitude", np.float32, dimensions="geo_reference_grid_cells")
            lon[:] = 150.
            lat = g1_1_2.createVariable("latitude", np.float32, dimensions="geo_reference_grid_cells")
            lat[:] = 12.

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
        self.reader = Thr3miL1cNCFileHandler(
            filename=self.test_file_name,
            filename_info=filename_info,
            filetype_info={}
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

        expected_longitude = np.ones(10)*150.
        expected_latitude = np.ones(10)*12.
        expected_Q = np.ones((10, 3))
        expected_Q[:, 0] = 75.
        expected_Q[:, 1] = 76.
        expected_Q[:, 2] = 77.

        longitude = self.reader.get_dataset(None, {"file_key": "data/overlap_XXX/geolocation_data/longitude",
                                                   "file_key_overlap": "/dimension/overlaps"})
        latitude = self.reader.get_dataset(None, {"file_key": "data/overlap_XXX/geolocation_data/latitude",
                                                  "file_key_overlap": "/dimension/overlaps"})
        reflectance_Q = self.reader.get_dataset(None, {"file_key":
                                                       "data/overlap_XXX/measurement_data/r_0865/reflectance_",
                                                       "file_key_overlap": "/dimension/overlaps", "view":
                                                       "view2", "polarization": "Q"})

        assert (longitude == expected_longitude).all()
        assert (latitude == expected_latitude).all()
        assert (reflectance_Q == expected_Q).all()

        global_attributes = self.reader._get_global_attributes()
        # Since the global_attributes dictionary contains numpy arrays,
        # it is not possible to perform a simple equality test
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

    def test_standardize_dims(self):
        """Test the standardize_dims function."""
        test_variable = xr.DataArray(
            dims="geo_reference_grid_cells",
            name="test_data",
            attrs={
                "key_1": "value_lat_1",
                "key_2": "value_lat_2"
            },
            data=np.ones(10) * 1.
        )
        out_variable = self.reader._standardize_dims(test_variable)
        print("out_variable ", out_variable)
        assert np.allclose(out_variable.values, np.ones(10))
        assert out_variable.dims == ("y",)
        assert out_variable.attrs["key_1"] == "value_lat_1"
