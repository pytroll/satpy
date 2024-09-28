#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2019 Satpy developers
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

"""Tests for the CF reader."""

import datetime as dt
import warnings

import numpy as np
import pytest
import xarray as xr
from pyresample import AreaDefinition, SwathDefinition

from satpy import Scene
from satpy.dataset.dataid import WavelengthRange
from satpy.readers.satpy_cf_nc import SatpyCFFileHandler

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - tmp_path


def _create_test_netcdf(filename, resolution=742):
    size = 2
    if resolution == 371:
        size = 4
    data_visir = np.array(np.arange(1, size * size + 1)).reshape(size, size)
    lat = 33.0 * data_visir
    lon = -13.0 * data_visir

    lat = xr.DataArray(lat,
                       dims=("y", "x"),
                       attrs={"name": "lat",
                              "standard_name": "latitude",
                              "modifiers": np.array([])})
    lon = xr.DataArray(lon,
                       dims=("y", "x"),
                       attrs={"name": "lon",
                              "standard_name": "longitude",
                              "modifiers": np.array([])})

    solar_zenith_angle_i = xr.DataArray(data_visir,
                                        dims=("y", "x"),
                                        attrs={"name": "solar_zenith_angle",
                                               "coordinates": "lat lon",
                                               "resolution": resolution})

    scene = Scene()
    scene.attrs["sensor"] = ["viirs"]
    scene_dict = {
        "lat": lat,
        "lon": lon,
        "solar_zenith_angle": solar_zenith_angle_i
    }

    tstart = dt.datetime(2019, 4, 1, 12, 0)
    tend = dt.datetime(2019, 4, 1, 12, 15)
    common_attrs = {
        "start_time": tstart,
        "end_time": tend,
        "platform_name": "NOAA 20",
        "orbit_number": 99999
    }

    for key in scene_dict:
        scene[key] = scene_dict[key]
        if key != "swath_data":
            scene[key].attrs.update(common_attrs)
    scene.save_datasets(writer="cf",
                        filename=filename,
                        engine="h5netcdf",
                        flatten_attrs=True,
                        pretty=True)
    return filename


@pytest.fixture(scope="session")
def area():
    """Get area definition."""
    area_extent = (339045.5577, 4365586.6063, 1068143.527, 4803645.4685)
    proj_dict = {"a": 6378169.0, "b": 6356583.8, "h": 35785831.0,
                 "lon_0": 0.0, "proj": "geos", "units": "m"}
    area = AreaDefinition("test",
                          "test",
                          "test",
                          proj_dict,
                          2,
                          2,
                          area_extent)
    return area


@pytest.fixture(scope="session")
def common_attrs(area):
    """Get common dataset attributes."""
    return {
        "start_time": dt.datetime(2019, 4, 1, 12, 0, 0, 123456),
        "end_time": dt.datetime(2019, 4, 1, 12, 15),
        "platform_name": "tirosn",
        "orbit_number": 99999,
        "area": area,
        "my_timestamp": dt.datetime(2000, 1, 1)
    }


@pytest.fixture(scope="session")
def xy_coords(area):
    """Get projection coordinates."""
    x, y = area.get_proj_coords()
    y = y[:, 0]
    x = x[0, :]
    return x, y


@pytest.fixture(scope="session")
def vis006(xy_coords, common_attrs):
    """Get VIS006 dataset."""
    x, y = xy_coords
    attrs = {
        "name": "image0",
        "id_tag": "ch_r06",
        "coordinates": "lat lon",
        "resolution": 1000,
        "calibration": "reflectance",
        "wavelength": WavelengthRange(min=0.58, central=0.63, max=0.68, unit="µm"),
        "orbital_parameters": {
          "projection_longitude": 1,
          "projection_latitude": 1,
          "projection_altitude": 1,
          "satellite_nominal_longitude": 1,
          "satellite_nominal_latitude": 1,
          "satellite_actual_longitude": 1,
          "satellite_actual_latitude": 1,
          "satellite_actual_altitude": 1,
          "nadir_longitude": 1,
          "nadir_latitude": 1,
          "only_in_1": False
        },
        "time_parameters": {
          "nominal_start_time": common_attrs["start_time"],
          "nominal_end_time": common_attrs["end_time"]
        }
    }
    coords = {"y": y, "x": x, "acq_time": ("y", [1, 2])}
    vis006 = xr.DataArray(np.array([[1, 2], [3, 4]]),
                          dims=("y", "x"),
                          coords=coords,
                          attrs=attrs)
    return vis006


@pytest.fixture(scope="session")
def ir_108(xy_coords):
    """Get IR_108 dataset."""
    x, y = xy_coords
    coords = {"y": y, "x": x, "acq_time": ("y", [1, 2])}
    attrs = {"name": "image1", "id_tag": "ch_tb11", "coordinates": "lat lon"}
    ir_108 = xr.DataArray(np.array([[1, 2], [3, 4]]),
                          dims=("y", "x"),
                          coords=coords,
                          attrs=attrs)
    return ir_108


@pytest.fixture(scope="session")
def qual_flags(xy_coords):
    """Get quality flags."""
    qual_data = [[1, 2, 3, 4, 5, 6, 7],
                 [1, 2, 3, 4, 5, 6, 7]]
    x, y = xy_coords
    z = [1, 2, 3, 4, 5, 6, 7]
    coords = {"y": y, "z": z, "acq_time": ("y", [1, 2])}
    qual_f = xr.DataArray(qual_data,
                          dims=("y", "z"),
                          coords=coords,
                          attrs={"name": "qual_flags",
                                 "id_tag": "qual_flags"})
    return qual_f


@pytest.fixture(scope="session")
def lonlats(xy_coords):
    """Get longitudes and latitudes."""
    x, y = xy_coords
    lat = 33.0 * np.array([[1, 2], [3, 4]])
    lon = -13.0 * np.array([[1, 2], [3, 4]])
    attrs = {"name": "lat",
             "standard_name": "latitude",
             "modifiers": np.array([])}
    dims = ("y", "x")
    coords = {"y": y, "x": x}
    lat = xr.DataArray(lat, dims=dims, coords=coords, attrs=attrs)
    lon = xr.DataArray(lon, dims=dims, coords=coords, attrs=attrs)
    return lon, lat


@pytest.fixture(scope="session")
def prefix_data(xy_coords, area):
    """Get dataset whose name should be prefixed."""
    x, y = xy_coords
    attrs = {"name": "1",
             "id_tag": "ch_r06",
             "coordinates": "lat lon",
             "resolution": 1000,
             "calibration": "reflectance",
             "wavelength": WavelengthRange(min=0.58, central=0.63, max=0.68, unit="µm"),
             "area": area}
    prefix_data = xr.DataArray(np.array([[1, 2], [3, 4]]),
                               dims=("y", "x"),
                               coords={"y": y, "x": x},
                               attrs=attrs)
    return prefix_data


@pytest.fixture(scope="session")
def swath_data(prefix_data, lonlats):
    """Get swath data."""
    lon, lat = lonlats
    area = SwathDefinition(lons=lon, lats=lat)
    swath_data = prefix_data.copy()
    swath_data.attrs.update({"name": "swath_data", "area": area})
    return swath_data


@pytest.fixture(scope="session")
def datasets(vis006, ir_108, qual_flags, lonlats, prefix_data, swath_data):
    """Get datasets belonging to the scene."""
    lon, lat = lonlats
    return {"image0": vis006,
            "image1": ir_108,
            "swath_data": swath_data,
            "1": prefix_data,
            "lat": lat,
            "lon": lon,
            "qual_flags": qual_flags}


@pytest.fixture(scope="session")
def cf_scene(datasets, common_attrs):
    """Create a cf scene."""
    scene = Scene()
    scene.attrs["sensor"] = ["avhrr-1", "avhrr-2", "avhrr-3"]
    for key in datasets:
        scene[key] = datasets[key]
        if key != "swath_data":
            scene[key].attrs.update(common_attrs)
    return scene


@pytest.fixture
def nc_filename(tmp_path):
    """Create an nc filename for viirs m band."""
    now = dt.datetime.utcnow()
    filename = f"testingcfwriter{now:%Y%j%H%M%S}-viirs-mband-20201007075915-20201007080744.nc"
    return str(tmp_path / filename)


@pytest.fixture
def nc_filename_i(tmp_path):
    """Create an nc filename for viirs i band."""
    now = dt.datetime.utcnow()
    filename = f"testingcfwriter{now:%Y%j%H%M%S}-viirs-iband-20201007075915-20201007080744.nc"
    return str(tmp_path / filename)


class TestCFReader:
    """Test case for CF reader."""

    def test_write_and_read_with_area_definition(self, cf_scene, nc_filename):
        """Save a dataset with an area definition to file with cf_writer and read the data again."""
        cf_scene.save_datasets(writer="cf",
                               filename=nc_filename,
                               engine="h5netcdf",
                               flatten_attrs=True,
                               pretty=True)
        scn_ = Scene(reader="satpy_cf_nc",
                     filenames=[nc_filename])
        scn_.load(["image0", "image1", "lat"])
        np.testing.assert_array_equal(scn_["image0"].data, cf_scene["image0"].data)
        np.testing.assert_array_equal(scn_["lat"].data, cf_scene["lat"].data)  # lat loaded as dataset
        np.testing.assert_array_equal(scn_["image0"].coords["lon"], cf_scene["lon"].data)  # lon loded as coord
        assert isinstance(scn_["image0"].attrs["wavelength"], WavelengthRange)
        expected_area = cf_scene["image0"].attrs["area"]
        actual_area = scn_["image0"].attrs["area"]
        assert pytest.approx(expected_area.area_extent, 0.000001) == actual_area.area_extent
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    message=r"You will likely lose important projection information",
                                    category=UserWarning)
            assert expected_area.proj_dict == actual_area.proj_dict
        assert expected_area.shape == actual_area.shape
        assert expected_area.area_id == actual_area.area_id
        assert expected_area.description == actual_area.description

    def test_write_and_read_with_swath_definition(self, cf_scene, nc_filename):
        """Save a dataset with a swath definition to file with cf_writer and read the data again."""
        with warnings.catch_warnings():
            # Filter out warning about missing lon/lat DataArray coordinates
            warnings.filterwarnings("ignore", category=UserWarning, message=r"Coordinate .* referenced")
            cf_scene.save_datasets(writer="cf",
                                filename=nc_filename,
                                engine="h5netcdf",
                                flatten_attrs=True,
                                pretty=True,
                                datasets=["swath_data"])
        scn_ = Scene(reader="satpy_cf_nc",
                     filenames=[nc_filename])
        scn_.load(["swath_data"])
        expected_area = cf_scene["swath_data"].attrs["area"]
        actual_area = scn_["swath_data"].attrs["area"]
        assert expected_area.shape == actual_area.shape
        np.testing.assert_array_equal(expected_area.lons.data, actual_area.lons.data)
        np.testing.assert_array_equal(expected_area.lats.data, actual_area.lats.data)

    def test_fix_modifier_attr(self):
        """Check that fix modifier can handle empty list as modifier attribute."""
        reader = SatpyCFFileHandler("filename",
                                    {},
                                    {"filetype": "info"})
        ds_info = {"modifiers": []}
        reader.fix_modifier_attr(ds_info)
        assert ds_info["modifiers"] == ()

    def test_read_prefixed_channels(self, cf_scene, nc_filename):
        """Check channels starting with digit is prefixed and read back correctly."""
        cf_scene.save_datasets(writer="cf",
                               filename=nc_filename,
                               engine="netcdf4",
                               flatten_attrs=True,
                               pretty=True)
        scn_ = Scene(reader="satpy_cf_nc",
                     filenames=[nc_filename])
        scn_.load(["1"])
        np.testing.assert_array_equal(scn_["1"].data, cf_scene["1"].data)
        np.testing.assert_array_equal(scn_["1"].coords["lon"], cf_scene["lon"].data)  # lon loaded as coord

        scn_ = Scene(reader="satpy_cf_nc",
                     filenames=[nc_filename], reader_kwargs={})
        scn_.load(["1"])
        np.testing.assert_array_equal(scn_["1"].data, cf_scene["1"].data)
        np.testing.assert_array_equal(scn_["1"].coords["lon"], cf_scene["lon"].data)  # lon loaded as coord

        # Check that variables starting with a digit is written to filename variable prefixed
        with xr.open_dataset(nc_filename) as ds_disk:
            np.testing.assert_array_equal(ds_disk["CHANNEL_1"].data, cf_scene["1"].data)

    def test_read_prefixed_channels_include_orig_name(self, cf_scene, nc_filename):
        """Check channels starting with digit and includeed orig name is prefixed and read back correctly."""
        cf_scene.save_datasets(writer="cf",
                               filename=nc_filename,
                               engine="netcdf4",
                               flatten_attrs=True,
                               pretty=True,
                               include_orig_name=True)
        scn_ = Scene(reader="satpy_cf_nc",
                     filenames=[nc_filename])
        scn_.load(["1"])
        np.testing.assert_array_equal(scn_["1"].data, cf_scene["1"].data)
        np.testing.assert_array_equal(scn_["1"].coords["lon"], cf_scene["lon"].data)  # lon loaded as coord

        assert scn_["1"].attrs["original_name"] == "1"

        # Check that variables starting with a digit is written to filename variable prefixed
        with xr.open_dataset(nc_filename) as ds_disk:
            np.testing.assert_array_equal(ds_disk["CHANNEL_1"].data, cf_scene["1"].data)

    def test_read_prefixed_channels_by_user(self, cf_scene, nc_filename):
        """Check channels starting with digit is prefixed by user and read back correctly."""
        cf_scene.save_datasets(writer="cf",
                               filename=nc_filename,
                               engine="netcdf4",
                               flatten_attrs=True,
                               pretty=True,
                               numeric_name_prefix="USER")
        scn_ = Scene(reader="satpy_cf_nc",
                     filenames=[nc_filename], reader_kwargs={"numeric_name_prefix": "USER"})
        scn_.load(["1"])
        np.testing.assert_array_equal(scn_["1"].data, cf_scene["1"].data)
        np.testing.assert_array_equal(scn_["1"].coords["lon"], cf_scene["lon"].data)  # lon loded as coord

        # Check that variables starting with a digit is written to filename variable prefixed
        with xr.open_dataset(nc_filename) as ds_disk:
            np.testing.assert_array_equal(ds_disk["USER1"].data, cf_scene["1"].data)

    def test_read_prefixed_channels_by_user2(self, cf_scene, nc_filename):
        """Check channels starting with digit is prefixed by user when saving and read back correctly without prefix."""
        cf_scene.save_datasets(writer="cf",
                               filename=nc_filename,
                               engine="netcdf4",
                               flatten_attrs=True,
                               pretty=True,
                               include_orig_name=False,
                               numeric_name_prefix="USER")
        scn_ = Scene(reader="satpy_cf_nc",
                     filenames=[nc_filename])
        scn_.load(["USER1"])
        np.testing.assert_array_equal(scn_["USER1"].data, cf_scene["1"].data)
        np.testing.assert_array_equal(scn_["USER1"].coords["lon"], cf_scene["lon"].data)  # lon loded as coord

    def test_read_prefixed_channels_by_user_include_prefix(self, cf_scene, nc_filename):
        """Check channels starting with digit is prefixed by user and include original name when saving."""
        cf_scene.save_datasets(writer="cf",
                               filename=nc_filename,
                               engine="netcdf4",
                               flatten_attrs=True,
                               pretty=True,
                               include_orig_name=True,
                               numeric_name_prefix="USER")
        scn_ = Scene(reader="satpy_cf_nc",
                     filenames=[nc_filename])
        scn_.load(["1"])
        np.testing.assert_array_equal(scn_["1"].data, cf_scene["1"].data)
        np.testing.assert_array_equal(scn_["1"].coords["lon"], cf_scene["lon"].data)  # lon loded as coord

    def test_read_prefixed_channels_by_user_no_prefix(self, cf_scene, nc_filename):
        """Check channels starting with digit is not prefixed by user."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*starts with a digit.*")
            cf_scene.save_datasets(writer="cf",
                                   filename=nc_filename,
                                   engine="netcdf4",
                                   flatten_attrs=True,
                                   pretty=True,
                                   numeric_name_prefix="")
        scn_ = Scene(reader="satpy_cf_nc",
                     filenames=[nc_filename])
        scn_.load(["1"])
        np.testing.assert_array_equal(scn_["1"].data, cf_scene["1"].data)
        np.testing.assert_array_equal(scn_["1"].coords["lon"], cf_scene["lon"].data)  # lon loded as coord

    def test_decoding_of_dict_type_attributes(self, cf_scene, nc_filename):
        """Test decoding of dict type attributes."""
        cf_scene.save_datasets(writer="cf",
                               filename=nc_filename)
        scn_ = Scene(reader="satpy_cf_nc",
                     filenames=[nc_filename])
        scn_.load(["image0"])
        for attr_name in ["orbital_parameters", "time_parameters"]:
            orig_attrs = cf_scene["image0"].attrs[attr_name]
            new_attrs = scn_["image0"].attrs[attr_name]
            assert new_attrs == orig_attrs

    def test_decoding_of_timestamps(self, cf_scene, nc_filename):
        """Test decoding of timestamps."""
        cf_scene.save_datasets(writer="cf", filename=nc_filename)
        scn = Scene(reader="satpy_cf_nc", filenames=[nc_filename])
        scn.load(["image0"])
        expected = cf_scene["image0"].attrs["my_timestamp"]
        assert scn["image0"].attrs["my_timestamp"] == expected

    def test_write_and_read_from_two_files(self, nc_filename, nc_filename_i):
        """Save two datasets with different resolution and read the solar_zenith_angle again."""
        _create_test_netcdf(nc_filename, resolution=742)
        _create_test_netcdf(nc_filename_i, resolution=371)
        scn_ = Scene(reader="satpy_cf_nc",
                     filenames=[nc_filename, nc_filename_i])
        scn_.load(["solar_zenith_angle"], resolution=742)
        assert scn_["solar_zenith_angle"].attrs["resolution"] == 742
        scn_.unload()
        scn_.load(["solar_zenith_angle"], resolution=371)
        assert scn_["solar_zenith_angle"].attrs["resolution"] == 371

    def test_dataid_attrs_equal_matching_dataset(self, cf_scene, nc_filename):
        """Check that get_dataset returns valid dataset when keys matches."""
        from satpy.dataset.dataid import DataID, default_id_keys_config
        _create_test_netcdf(nc_filename, resolution=742)
        reader = SatpyCFFileHandler(nc_filename, {}, {"filetype": "info"})
        ds_id = DataID(default_id_keys_config, name="solar_zenith_angle", resolution=742, modifiers=())
        res = reader.get_dataset(ds_id, {})
        assert res.attrs["resolution"] == 742

    def test_dataid_attrs_equal_not_matching_dataset(self, cf_scene, nc_filename):
        """Check that get_dataset returns None when key(s) are not matching."""
        from satpy.dataset.dataid import DataID, default_id_keys_config
        _create_test_netcdf(nc_filename, resolution=742)
        reader = SatpyCFFileHandler(nc_filename, {}, {"filetype": "info"})
        not_existing_resolution = 9999999
        ds_id = DataID(default_id_keys_config, name="solar_zenith_angle", resolution=not_existing_resolution,
                       modifiers=())
        assert reader.get_dataset(ds_id, {}) is None

    def test_dataid_attrs_equal_contains_not_matching_key(self, cf_scene, nc_filename):
        """Check that get_dataset returns valid dataset when dataid have key(s) not existing in data."""
        from satpy.dataset.dataid import DataID, default_id_keys_config
        _create_test_netcdf(nc_filename, resolution=742)
        reader = SatpyCFFileHandler(nc_filename, {}, {"filetype": "info"})
        ds_id = DataID(default_id_keys_config, name="solar_zenith_angle", resolution=742,
                       modifiers=(), calibration="counts")
        res = reader.get_dataset(ds_id, {})
        assert res.attrs["resolution"] == 742
