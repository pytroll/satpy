#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Module for testing the satpy.readers.osisaf_l3 module."""

import os
from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from satpy import DataQuery
from satpy.readers.osisaf_l3_nc import OSISAFL3NCFileHandler

stere_ds = xr.DataArray(
    -999,
    attrs={"grid_mapping_name": "polar_stereographic",
    "false_easting": 0.0,
    "false_northing": 0.0,
    "semi_major_axis": 6378273.0,
    "semi_minor_axis": 6356889.44891,
    "straight_vertical_longitude_from_pole": 0.0,
    "latitude_of_projection_origin": -90.0,
    "standard_parallel": -70.0,
    "proj4_string": "+proj=stere +a=6378273 +b=6356889.44891 +lat_0=-90 +lat_ts=-70 +lon_0=0",
})

ease_ds = xr.DataArray(
    -999,
    attrs={"grid_mapping_name": "lambert_azimuthal_equal_area",
           "false_easting": 0.0,
           "false_northing": 0.0,
           "semi_major_axis": 6371228.0,
           "longitude_of_projection_origin": 0.0,
           "latitude_of_projection_origin": -90.0,
           "proj4_string": "+proj=laea +a=6371228.0 +lat_0=-90 +lon_0=0",
           })


class TestOSISAFL3Reader:
    """Test OSI-SAF level 3 netCDF reader."""

    def setup_method(self, proj_type):
        """Create a fake dataset."""
        self.base_data = np.array(([-999, 1215, 1125, 11056, 9500], [200, 1, -999, 4215, 5756]))
        self.base_data = np.expand_dims(self.base_data, axis=0)
        self.unc_data = np.array(([0, 1, 2, 3, 4], [4, 3, 2, 1, 0]))
        self.yc_data = np.array(([-10, -5, 0, 5, 10], [-10, -5, 0, 5, 10]))
        self.xc_data = np.array(([-5, -5, -5, -5, -5], [5, 5, 5, 5, 5]))
        self.time_data = np.array([1.])

        self.lat_data = np.array(([-68, -69, -70, -71, -72], [-68, -69, -70, -71, -72]))
        self.lon_data = np.array(([-60, -60, -60, -60, -60], [-65, -65, -65, -65, -65]))
        self.xc = xr.DataArray(
            self.xc_data,
            dims=("yc", "xc"),
            attrs={"standard_name": "projection_x_coordinate", "units": "km"}
        )
        self.yc = xr.DataArray(
            self.yc_data,
            dims=("yc", "xc"),
            attrs={"standard_name": "projection_y_coordinate", "units": "km"}
        )
        self.time = xr.DataArray(
            self.time_data,
            dims=("time"),
            attrs={"standard_name": "projection_y_coordinate", "units": "km"}
        )
        self.lat = xr.DataArray(
            self.lat_data,
            dims=("yc", "xc"),
            attrs={"standard_name": "latitude", "units": "degrees_north"}
        )
        self.lon = xr.DataArray(
            self.lon_data,
            dims=("yc", "xc"),
            attrs={"standard_name": "longitude", "units": "degrees_east"}
        )
        self.conc = xr.DataArray(
            self.base_data,
            dims=("time", "yc", "xc"),
            attrs={"scale_factor": 0.01, "add_offset": 0., "_FillValue": -999, "units": "%",
                   "valid_min": 0, "valid_max": 10000, "standard_name": "sea_ice_area_fraction"}
        )
        self.uncert = xr.DataArray(
            self.unc_data,
            dims=("yc", "xc"),
            attrs={"scale_factor": 0.01, "add_offset": 0., "_FillValue": -999,
                   "valid_min": 0, "valid_max": 10000, "standard_name": "total_uncertainty"}
        )

        data_vars = {
                "ice_conc": self.conc,
                "total_uncertainty": self.uncert,
                "xc": self.xc,
                "yc": self.yc,
                "time": self.time,
                "lat": self.lat,
                "lon": self.lon,
                "Lambert_Azimuthal_Grid": ease_ds,
                "Polar_Stereographic_Grid": stere_ds}
        self.fake_dataset = xr.Dataset(
            data_vars=data_vars,
            attrs={
                "start_date": "2022-12-15 00:00:00",
                "stop_date": "2022-12-16 00:00:00",
                "platform_name": "Multi-sensor analysis",
                "instrument_type": "Multi-sensor analysis"},
        )

    def test_instantiate_single_netcdf_file(self, tmp_path):
        """Test initialization of file handlers - given a single netCDF file."""
        filename_info = {}
        filetype_info = {}
        tmp_filepath = tmp_path / "fake_dataset.nc"
        self.fake_dataset.to_netcdf(os.fspath(tmp_filepath))

        OSISAFL3NCFileHandler(os.fspath(tmp_filepath), filename_info, filetype_info)


    def test_get_dataset(self, tmp_path):
        """Test retrieval of datasets."""
        filename_info = {}
        filetype_info = {}
        tmp_filepath = tmp_path / "fake_dataset.nc"
        self.fake_dataset.to_netcdf(os.fspath(tmp_filepath))

        test = OSISAFL3NCFileHandler(os.fspath(tmp_filepath), filename_info, filetype_info)

        res = test.get_dataset(DataQuery(name="ice_conc"), {"standard_name": "sea_ice_area_fraction"})
        # Check we remove singleton dimension
        assert res.shape[0] == 2
        assert res.shape[1] == 5

        # Test values are correct
        test_ds = self.fake_dataset["ice_conc"][0].values
        test_ds = np.where(test_ds == -999, np.nan, test_ds)
        test_ds = np.where(test_ds > 10000, np.nan, test_ds)
        np.testing.assert_allclose(res.values, test_ds / 100)

        res = test.get_dataset(DataQuery(name="total_uncertainty"), {"standard_name": "sea_ice_area_fraction"})
        assert res.shape[0] == 2
        assert res.shape[1] == 5

        with pytest.raises(KeyError):
            test.get_dataset(DataQuery(name="erroneous dataset"), {"standard_name": "erroneous dataset"})

    def test_get_start_and_end_times(self, tmp_path):
        """Test retrieval of the sensor name from the netCDF file."""
        good_start_time = datetime(2022, 12, 15, 0, 0, 0)
        good_stop_time = datetime(2022, 12, 16, 0, 0, 0)

        filename_info = {}
        filetype_info = {}

        tmp_filepath = tmp_path / "fake_dataset.nc"
        self.fake_dataset.to_netcdf(os.fspath(tmp_filepath))

        test = OSISAFL3NCFileHandler(os.fspath(tmp_filepath), filename_info, filetype_info)

        assert test.start_time == good_start_time
        assert test.end_time == good_stop_time


    def test_get_area_def_ease(self, tmp_path):
        """Test getting the area definition for the EASE grid."""
        filename_info = {"grid": "ease"}
        filetype_info = {}
        tmp_filepath = tmp_path / "fake_dataset.nc"
        self.fake_dataset.to_netcdf(os.fspath(tmp_filepath))

        test = OSISAFL3NCFileHandler(os.fspath(tmp_filepath), filename_info, filetype_info)

        area_def = test.get_area_def(None)
        assert area_def.description == "osisaf_lambert_azimuthal_equal_area"
        assert area_def.proj_dict["R"] == 6371228
        assert area_def.proj_dict["lat_0"] == -90
        assert area_def.proj_dict["lon_0"] == 0
        assert area_def.proj_dict["proj"] == "laea"

        assert area_def.width == 5
        assert area_def.height == 2
        np.testing.assert_allclose(area_def.area_extent,
                                   (-2203574.302335, 1027543.572492, -1726299.781982, 996679.643829))


    def test_get_area_def_stere(self, tmp_path):
        """Test getting the area definition for the polar stereographic grid."""
        filename_info = {"grid": "stere"}
        filetype_info = {}
        tmp_filepath = tmp_path / "fake_dataset.nc"
        self.fake_dataset.to_netcdf(os.fspath(tmp_filepath))

        test = OSISAFL3NCFileHandler(os.fspath(tmp_filepath), filename_info, filetype_info)

        area_def = test.get_area_def(None)
        assert area_def.description == "osisaf_polar_stereographic"
        assert area_def.proj_dict["a"] == 6378273.0
        assert area_def.proj_dict["lat_0"] == -90
        assert area_def.proj_dict["lat_ts"] == -70
        assert area_def.proj_dict["lon_0"] == 0
        assert area_def.proj_dict["proj"] == "stere"

        assert area_def.width == 5
        assert area_def.height == 2
        np.testing.assert_allclose(area_def.area_extent,
                                   (-2185821.7955, 1019265.4426, -1702157.4538, 982741.0642))

    def test_get_area_def_bad(self, tmp_path):
        """Test getting the area definition for the polar stereographic grid."""
        filename_info = {"grid": "turnips"}
        filetype_info = {}
        tmp_filepath = tmp_path / "fake_dataset.nc"
        self.fake_dataset.to_netcdf(os.fspath(tmp_filepath))

        test = OSISAFL3NCFileHandler(os.fspath(tmp_filepath), filename_info, filetype_info)
        with pytest.raises(ValueError, match="Unknown grid type: turnips"):
            test.get_area_def(None)
