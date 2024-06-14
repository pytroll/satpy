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

import datetime as dt
import os

import numpy as np
import pytest
import xarray as xr
from pyproj import CRS

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

stere_ds_noproj = xr.DataArray(
    -999,
    attrs={"grid_mapping_name": "polar_stereographic",
           "false_easting": 0.0,
           "false_northing": 0.0,
           "semi_major_axis": 6378273.0,
           "semi_minor_axis": 6356889.44891,
           "straight_vertical_longitude_from_pole": 0.0,
           "latitude_of_projection_origin": -90.0,
           "standard_parallel": -70.0,
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

attrs_ice = {
    "start_date": "2022-12-15 00:00:00",
    "stop_date": "2022-12-16 00:00:00",
    "platform_name": "Multi-sensor analysis",
    "instrument_type": "Multi-sensor analysis"}

attrs_flux = {
    "time_coverage_start": "2023-10-10T00:00:00Z",
    "time_coverage_end": "2023-10-10T23:59:59Z",
    "platform": "NOAA-19, NOAA-20, Metop-B, Metop-C, SNPP",
    "sensor": "AVHRR, VIIRS, AVHRR, AVHRR, VIIRS"}

attrs_geo = {
    "start_time": "20221228T183000Z",
    "stop_time": "20221228T193000Z",
    "platform": "MSG4"}


class OSISAFL3ReaderTests:
    """Test OSI-SAF level 3 netCDF reader ice files."""

    def setup_method(self, tester="ice"):
        """Create a fake dataset."""
        base_data = np.array(([-999, 1215, 1125, 11056, 9500], [200, 1, -999, 4215, 5756]))
        base_data_ssi = np.array(([-999.99, 121.5, 11.25, 110.56, 950.0], [200, 1, -999.99, 42.15, 5.756]))
        base_data_sst = np.array(([-32768, 273.2, 194.2, 220.78, 301.], [-32768, -32768, 273.22, 254.34, 204.21]))
        base_data_ssi_geo = np.array(([-32768, 121.5, 11.25, 110.56, 950.0], [200, 1, -32768, 42.15, 5.756]))
        base_data = np.expand_dims(base_data, axis=0)
        base_data_ssi = np.expand_dims(base_data_ssi, axis=0)
        base_data_sst = np.expand_dims(base_data_sst, axis=0)
        unc_data = np.array(([0, 1, 2, 3, 4], [4, 3, 2, 1, 0]))
        yc_data = np.array(([-10, -5, 0, 5, 10], [-10, -5, 0, 5, 10]))
        xc_data = np.array(([-5, -5, -5, -5, -5], [5, 5, 5, 5, 5]))
        time_data = np.array([1.])
        self.scl = 1.
        self.add = 0.

        lat_data = np.array(([-68, -69, -70, -71, -72], [-68, -69, -70, -71, -72]))
        lon_data = np.array(([-60, -60, -60, -60, -60], [-65, -65, -65, -65, -65]))

        xc = xr.DataArray(xc_data, dims=("yc", "xc"),
                          attrs={"standard_name": "projection_x_coordinate", "units": "km"})
        yc = xr.DataArray(yc_data, dims=("yc", "xc"),
                          attrs={"standard_name": "projection_y_coordinate", "units": "km"})
        time = xr.DataArray(time_data, dims="time",
                            attrs={"standard_name": "projection_y_coordinate", "units": "km"})
        lat = xr.DataArray(lat_data, dims=("yc", "xc"),
                           attrs={"standard_name": "latitude", "units": "degrees_north"})
        lon = xr.DataArray(lon_data, dims=("yc", "xc"),
                           attrs={"standard_name": "longitude", "units": "degrees_east"})
        conc = xr.DataArray(base_data, dims=("time", "yc", "xc"),
                            attrs={"scale_factor": 0.01, "add_offset": 0., "_FillValue": -999, "units": "%",
                                   "valid_min": 0, "valid_max": 10000, "standard_name": "sea_ice_area_fraction"})
        uncert = xr.DataArray(unc_data, dims=("yc", "xc"),
                              attrs={"scale_factor": 0.01, "add_offset": 0., "_FillValue": -999, "valid_min": 0,
                                     "valid_max": 10000, "standard_name": "total_uncertainty"})
        ssi_geo = xr.DataArray(base_data_ssi_geo, dims=("lat", "lon"),
                               attrs={"scale_factor": 0.1, "add_offset": 0., "_FillValue": 32768, "valid_min": 0.,
                                      "valid_max": 1000., "standard_name": "surface_downwelling_shortwave_flux_in_air"})
        ssi = xr.DataArray(base_data_ssi, dims=("time", "yc", "xc"),
                           attrs={"_FillValue": -999.99, "units": "W m-2", "valid_min": 0., "valid_max": 1000.,
                                  "standard_name": "surface_downwelling_shortwave_flux_in_air"})
        sst = xr.DataArray(base_data_sst, dims=("time", "yc", "xc"),
                           attrs={"scale_factor": 0.01, "add_offset": 273.15, "_FillValue": -32768, "units": "K",
                                  "valid_min": -8000., "valid_max": 5000.,
                                  "standard_name": "sea_ice_surface_temperature"})
        data_vars = {"xc": xc, "yc": yc, "time": time, "lat": lat, "lon": lon}

        if tester == "ice":
            data_vars["Lambert_Azimuthal_Grid"] = ease_ds
            data_vars["Polar_Stereographic_Grid"] = stere_ds
            data_vars["ice_conc"] = conc
            data_vars["total_uncertainty"] = uncert
            self.fake_dataset = xr.Dataset(data_vars=data_vars, attrs=attrs_ice)
        elif tester == "sst":
            data_vars["Polar_Stereographic_Grid"] = stere_ds
            data_vars["surface_temperature"] = sst
            self.fake_dataset = xr.Dataset(data_vars=data_vars, attrs=attrs_ice)
        elif tester == "flux_stere":
            data_vars["Polar_Stereographic_Grid"] = stere_ds_noproj
            data_vars["ssi"] = ssi
            self.fake_dataset = xr.Dataset(data_vars=data_vars, attrs=attrs_flux)
        elif tester == "flux_geo":
            data_vars["ssi"] = ssi_geo
            self.fake_dataset = xr.Dataset(data_vars=data_vars, attrs=attrs_geo)

    def test_instantiate_single_netcdf_file(self, tmp_path):
        """Test initialization of file handlers - given a single netCDF file."""
        tmp_filepath = tmp_path / "fake_dataset.nc"
        self.fake_dataset.to_netcdf(os.fspath(tmp_filepath))

        OSISAFL3NCFileHandler(os.fspath(tmp_filepath), self.filename_info, self.filetype_info)

    def test_get_dataset(self, tmp_path):
        """Test retrieval of datasets."""
        tmp_filepath = tmp_path / "fake_dataset.nc"
        self.fake_dataset.to_netcdf(os.fspath(tmp_filepath))

        test = OSISAFL3NCFileHandler(os.fspath(tmp_filepath), self.filename_info, self.filetype_info)

        res = test.get_dataset(DataQuery(name=self.varname), {"standard_name": self.stdname})
        # Check we remove singleton dimension
        assert res.shape[0] == 2
        assert res.shape[1] == 5

        # Test values are correct
        test_ds = self.fake_dataset[self.varname].values.squeeze()
        test_ds = np.where(test_ds == self.fillv, np.nan, test_ds)
        test_ds = np.where(test_ds > self.maxv, np.nan, test_ds)
        test_ds = test_ds / self.scl + self.add
        np.testing.assert_allclose(res.values, test_ds)

        with pytest.raises(KeyError):
            test.get_dataset(DataQuery(name="erroneous dataset"), {"standard_name": "erroneous dataset"})

    def test_get_start_and_end_times(self, tmp_path):
        """Test retrieval of the sensor name from the netCDF file."""
        tmp_filepath = tmp_path / "fake_dataset.nc"
        self.fake_dataset.to_netcdf(os.fspath(tmp_filepath))

        test = OSISAFL3NCFileHandler(os.fspath(tmp_filepath), self.filename_info, self.filetype_info)

        assert test.start_time == self.good_start_time
        assert test.end_time == self.good_stop_time

    def test_get_area_def_bad(self, tmp_path):
        """Test getting the area definition for the polar stereographic grid."""
        filename_info = {"grid": "turnips"}
        tmp_filepath = tmp_path / "fake_dataset.nc"
        self.fake_dataset.to_netcdf(os.fspath(tmp_filepath))

        test = OSISAFL3NCFileHandler(os.fspath(tmp_filepath), filename_info, self.filetype_info)
        with pytest.raises(ValueError, match="Unknown grid type: turnips"):
            test.get_area_def(None)


class TestOSISAFL3ReaderICE(OSISAFL3ReaderTests):
    """Test OSI-SAF level 3 netCDF reader ice files."""

    def setup_method(self):
        """Set up the tests."""
        super().setup_method(tester="ice")
        self.filename_info = {"grid": "ease"}
        self.filetype_info = {"file_type": "osi_sea_ice_conc"}
        self.good_start_time = dt.datetime(2022, 12, 15, 0, 0, 0)
        self.good_stop_time = dt.datetime(2022, 12, 16, 0, 0, 0)
        self.varname = "ice_conc"
        self.stdname = "sea_ice_area_fraction"
        self.fillv = -999
        self.maxv = 10000
        self.scl = 100

    def test_get_area_def_stere(self, tmp_path):
        """Test getting the area definition for the polar stereographic grid."""
        self.filename_info = {"grid": "stere"}
        tmp_filepath = tmp_path / "fake_dataset.nc"
        self.fake_dataset.to_netcdf(os.fspath(tmp_filepath))

        test = OSISAFL3NCFileHandler(os.fspath(tmp_filepath), self.filename_info, self.filetype_info)

        area_def = test.get_area_def(None)
        assert area_def.description == "osisaf_polar_stereographic"

        expected_crs = CRS(dict(a=6378273.0, lat_0=-90, lat_ts=-70, lon_0=0, proj="stere", rf=298.27940986765))
        assert area_def.crs == expected_crs

        assert area_def.width == 5
        assert area_def.height == 2
        np.testing.assert_allclose(area_def.area_extent,
                                   (-2185821.7955, 1019265.4426, -1702157.4538, 982741.0642))

    def test_get_area_def_ease(self, tmp_path):
        """Test getting the area definition for the EASE grid."""
        tmp_filepath = tmp_path / "fake_dataset.nc"
        self.fake_dataset.to_netcdf(os.fspath(tmp_filepath))

        test = OSISAFL3NCFileHandler(os.fspath(tmp_filepath), {"grid": "ease"}, self.filetype_info)

        area_def = test.get_area_def(None)
        assert area_def.description == "osisaf_lambert_azimuthal_equal_area"

        expected_crs = CRS(dict(R=6371228, lat_0=-90, lon_0=0, proj="laea"))
        assert area_def.crs == expected_crs

        assert area_def.width == 5
        assert area_def.height == 2
        np.testing.assert_allclose(area_def.area_extent,
                                   (-2203574.302335, 1027543.572492, -1726299.781982, 996679.643829))


class TestOSISAFL3ReaderFluxStere(OSISAFL3ReaderTests):
    """Test OSI-SAF level 3 netCDF reader flux files on stereographic grid."""

    def setup_method(self):
        """Set up the tests."""
        super().setup_method(tester="flux_stere")
        self.filename_info = {"grid": "polstere"}
        self.filetype_info = {"file_type": "osi_radflux_stere"}
        self.good_start_time = dt.datetime(2023, 10, 10, 0, 0, 0)
        self.good_stop_time = dt.datetime(2023, 10, 10, 23, 59, 59)
        self.varname = "ssi"
        self.stdname = "surface_downwelling_shortwave_flux_in_air"
        self.fillv = -999.99
        self.maxv = 1000
        self.scl = 1

    def test_get_area_def_stere(self, tmp_path):
        """Test getting the area definition for the polar stereographic grid."""
        tmp_filepath = tmp_path / "fake_dataset.nc"
        self.fake_dataset.to_netcdf(os.fspath(tmp_filepath))

        test = OSISAFL3NCFileHandler(os.fspath(tmp_filepath), self.filename_info, self.filetype_info)

        area_def = test.get_area_def(None)
        assert area_def.description == "osisaf_polar_stereographic"

        expected_crs = CRS(dict(a=6378273.0, lat_0=-90, lat_ts=-70, lon_0=0, proj="stere", rf=298.27940986765))
        assert area_def.crs == expected_crs

        assert area_def.width == 5
        assert area_def.height == 2
        np.testing.assert_allclose(area_def.area_extent,
                                   (-2185821.7955, 1019265.4426, -1702157.4538, 982741.0642))


class TestOSISAFL3ReaderFluxGeo(OSISAFL3ReaderTests):
    """Test OSI-SAF level 3 netCDF reader flux files on lat/lon grid (GEO sensors)."""

    def setup_method(self):
        """Set up the tests."""
        super().setup_method(tester="flux_geo")
        self.filename_info = {}
        self.filetype_info = {"file_type": "osi_radflux_grid"}
        self.good_start_time = dt.datetime(2022, 12, 28, 18, 30, 0)
        self.good_stop_time = dt.datetime(2022, 12, 28, 19, 30, 0)
        self.varname = "ssi"
        self.stdname = "surface_downwelling_shortwave_flux_in_air"
        self.fillv = -32768
        self.maxv = 1000
        self.scl = 10

    def test_get_area_def_grid(self, tmp_path):
        """Test getting the area definition for the lat/lon grid."""
        tmp_filepath = tmp_path / "fake_dataset.nc"
        self.filename_info = {}
        self.filetype_info = {"file_type": "osi_radflux_grid"}
        self.fake_dataset.to_netcdf(os.fspath(tmp_filepath))

        test = OSISAFL3NCFileHandler(os.fspath(tmp_filepath), self.filename_info, self.filetype_info)

        area_def = test.get_area_def(None)
        assert area_def.description == "osisaf_geographic_area"

        expected_crs = CRS(dict(datum="WGS84", proj="longlat"))
        assert area_def.crs == expected_crs

        assert area_def.width == 5
        assert area_def.height == 2
        np.testing.assert_allclose(area_def.area_extent,
                                   (-65, -68, -60, -72))


class TestOSISAFL3ReaderSST(OSISAFL3ReaderTests):
    """Test OSI-SAF level 3 netCDF reader surface temperature files."""

    def setup_method(self):
        """Set up the tests."""
        super().setup_method(tester="sst")
        self.filename_info = {}
        self.filetype_info = {"file_type": "osi_sst"}
        self.good_start_time = dt.datetime(2022, 12, 15, 0, 0, 0)
        self.good_stop_time = dt.datetime(2022, 12, 16, 0, 0, 0)
        self.varname = "surface_temperature"
        self.stdname = "sea_ice_surface_temperature"
        self.fillv = -32768
        self.maxv = 1000
        self.scl = 100
        self.add = 273.15

    def test_get_area_def_stere(self, tmp_path):
        """Test getting the area definition for the polar stereographic grid."""
        tmp_filepath = tmp_path / "fake_dataset.nc"
        self.fake_dataset.to_netcdf(os.fspath(tmp_filepath))

        test = OSISAFL3NCFileHandler(os.fspath(tmp_filepath), self.filename_info, self.filetype_info)

        area_def = test.get_area_def(None)
        assert area_def.description == "osisaf_polar_stereographic"

        expected_crs = CRS(dict(a=6378273.0, lat_0=-90, lat_ts=-70, lon_0=0, proj="stere", rf=298.27940986765))

        assert area_def.crs == expected_crs

        assert area_def.width == 5
        assert area_def.height == 2
        np.testing.assert_allclose(area_def.area_extent,
                                   (-2185821.7955, 1019265.4426, -1702157.4538, 982741.0642))
