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

from datetime import datetime

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
                       dims=('y', 'x'),
                       attrs={'name': 'lat',
                              'standard_name': 'latitude',
                              'modifiers': np.array([])})
    lon = xr.DataArray(lon,
                       dims=('y', 'x'),
                       attrs={'name': 'lon',
                              'standard_name': 'longitude',
                              'modifiers': np.array([])})

    solar_zenith_angle_i = xr.DataArray(data_visir,
                                        dims=('y', 'x'),
                                        attrs={'name': 'solar_zenith_angle',
                                               'coordinates': 'lat lon',
                                               'resolution': resolution})

    scene = Scene()
    scene.attrs['sensor'] = ['viirs']
    scene_dict = {
        'lat': lat,
        'lon': lon,
        'solar_zenith_angle': solar_zenith_angle_i
    }

    tstart = datetime(2019, 4, 1, 12, 0)
    tend = datetime(2019, 4, 1, 12, 15)
    common_attrs = {
        'start_time': tstart,
        'end_time': tend,
        'platform_name': 'NOAA 20',
        'orbit_number': 99999
    }

    for key in scene_dict:
        scene[key] = scene_dict[key]
        if key != 'swath_data':
            scene[key].attrs.update(common_attrs)
    scene.save_datasets(writer='cf',
                        filename=filename,
                        engine='h5netcdf',
                        flatten_attrs=True,
                        pretty=True)
    return filename


@pytest.fixture(scope="session")
def _cf_scene():
    tstart = datetime(2019, 4, 1, 12, 0)
    tend = datetime(2019, 4, 1, 12, 15)
    data_visir = np.array([[1, 2], [3, 4]])
    z_visir = [1, 2, 3, 4, 5, 6, 7]
    qual_data = [[1, 2, 3, 4, 5, 6, 7],
                 [1, 2, 3, 4, 5, 6, 7]]
    time_vis006 = [1, 2]
    lat = 33.0 * np.array([[1, 2], [3, 4]])
    lon = -13.0 * np.array([[1, 2], [3, 4]])

    proj_dict = {
        'a': 6378169.0, 'b': 6356583.8, 'h': 35785831.0,
        'lon_0': 0.0, 'proj': 'geos', 'units': 'm'
    }
    x_size, y_size = data_visir.shape
    area_extent = (339045.5577, 4365586.6063, 1068143.527, 4803645.4685)
    area = AreaDefinition(
        'test',
        'test',
        'test',
        proj_dict,
        x_size,
        y_size,
        area_extent,
    )

    x, y = area.get_proj_coords()
    y_visir = y[:, 0]
    x_visir = x[0, :]

    common_attrs = {
        'start_time': tstart,
        'end_time': tend,
        'platform_name': 'tirosn',
        'orbit_number': 99999,
        'area': area
    }

    vis006 = xr.DataArray(data_visir,
                          dims=('y', 'x'),
                          coords={'y': y_visir, 'x': x_visir, 'acq_time': ('y', time_vis006)},
                          attrs={
                              'name': 'image0', 'id_tag': 'ch_r06',
                              'coordinates': 'lat lon', 'resolution': 1000, 'calibration': 'reflectance',
                              'wavelength': WavelengthRange(min=0.58, central=0.63, max=0.68, unit='µm'),
                              'orbital_parameters': {
                                  'projection_longitude': 1,
                                  'projection_latitude': 1,
                                  'projection_altitude': 1,
                                  'satellite_nominal_longitude': 1,
                                  'satellite_nominal_latitude': 1,
                                  'satellite_actual_longitude': 1,
                                  'satellite_actual_latitude': 1,
                                  'satellite_actual_altitude': 1,
                                  'nadir_longitude': 1,
                                  'nadir_latitude': 1,
                                  'only_in_1': False
                              }
                          })

    ir_108 = xr.DataArray(data_visir,
                          dims=('y', 'x'),
                          coords={'y': y_visir, 'x': x_visir, 'acq_time': ('y', time_vis006)},
                          attrs={'name': 'image1', 'id_tag': 'ch_tb11', 'coordinates': 'lat lon'})
    qual_f = xr.DataArray(qual_data,
                          dims=('y', 'z'),
                          coords={'y': y_visir, 'z': z_visir, 'acq_time': ('y', time_vis006)},
                          attrs={
                              'name': 'qual_flags',
                              'id_tag': 'qual_flags'
                          })
    lat = xr.DataArray(lat,
                       dims=('y', 'x'),
                       coords={'y': y_visir, 'x': x_visir},
                       attrs={
                           'name': 'lat',
                           'standard_name': 'latitude',
                           'modifiers': np.array([])
                       })
    lon = xr.DataArray(lon,
                       dims=('y', 'x'),
                       coords={'y': y_visir, 'x': x_visir},
                       attrs={
                           'name': 'lon',
                           'standard_name': 'longitude',
                           'modifiers': np.array([])
                       })

    # for prefix testing
    prefix_data = xr.DataArray(data_visir,
                               dims=('y', 'x'),
                               coords={'y': y_visir, 'x': x_visir},
                               attrs={
                                   'name': '1', 'id_tag': 'ch_r06',
                                   'coordinates': 'lat lon', 'resolution': 1000, 'calibration': 'reflectance',
                                   'wavelength': WavelengthRange(min=0.58, central=0.63, max=0.68, unit='µm'),
                                   'area': area
                               })

    # for swath testing
    area = SwathDefinition(lons=lon, lats=lat)
    swath_data = prefix_data.copy()
    swath_data.attrs.update({'name': 'swath_data', 'area': area})

    scene = Scene()
    scene.attrs['sensor'] = ['avhrr-1', 'avhrr-2', 'avhrr-3']
    scene_dict = {
        'image0': vis006,
        'image1': ir_108,
        'swath_data': swath_data,
        '1': prefix_data,
        'lat': lat,
        'lon': lon,
        'qual_flags': qual_f
    }

    for key in scene_dict:
        scene[key] = scene_dict[key]
        if key != 'swath_data':
            scene[key].attrs.update(common_attrs)
    return scene


@pytest.fixture
def _nc_filename(tmp_path):
    now = datetime.utcnow()
    filename = f'testingcfwriter{now:%Y%j%H%M%S}-viirs-mband-20201007075915-20201007080744.nc'
    return str(tmp_path / filename)


@pytest.fixture
def _nc_filename_i(tmp_path):
    now = datetime.utcnow()
    filename = f'testingcfwriter{now:%Y%j%H%M%S}-viirs-iband-20201007075915-20201007080744.nc'
    return str(tmp_path / filename)


class TestCFReader:
    """Test case for CF reader."""

    def test_write_and_read_with_area_definition(self, _cf_scene, _nc_filename):
        """Save a dataset with an area definition to file with cf_writer and read the data again."""
        _cf_scene.save_datasets(writer='cf',
                                filename=_nc_filename,
                                engine='h5netcdf',
                                flatten_attrs=True,
                                pretty=True)
        scn_ = Scene(reader='satpy_cf_nc',
                     filenames=[_nc_filename])
        scn_.load(['image0', 'image1', 'lat'])
        np.testing.assert_array_equal(scn_['image0'].data, _cf_scene['image0'].data)
        np.testing.assert_array_equal(scn_['lat'].data, _cf_scene['lat'].data)  # lat loaded as dataset
        np.testing.assert_array_equal(scn_['image0'].coords['lon'], _cf_scene['lon'].data)  # lon loded as coord
        assert isinstance(scn_['image0'].attrs['wavelength'], WavelengthRange)
        expected_area = _cf_scene['image0'].attrs['area']
        actual_area = scn_['image0'].attrs['area']
        assert pytest.approx(expected_area.area_extent, 0.000001) == actual_area.area_extent
        assert expected_area.proj_dict == actual_area.proj_dict
        assert expected_area.shape == actual_area.shape
        assert expected_area.area_id == actual_area.area_id
        assert expected_area.description == actual_area.description
        assert expected_area.proj_dict == actual_area.proj_dict

    def test_write_and_read_with_swath_definition(self, _cf_scene, _nc_filename):
        """Save a dataset with a swath definition to file with cf_writer and read the data again."""
        _cf_scene.save_datasets(writer='cf',
                                filename=_nc_filename,
                                engine='h5netcdf',
                                flatten_attrs=True,
                                pretty=True,
                                datasets=["swath_data"])
        scn_ = Scene(reader='satpy_cf_nc',
                     filenames=[_nc_filename])
        scn_.load(['swath_data'])
        expected_area = _cf_scene['swath_data'].attrs['area']
        actual_area = scn_['swath_data'].attrs['area']
        assert expected_area.shape == actual_area.shape
        np.testing.assert_array_equal(expected_area.lons.data, actual_area.lons.data)
        np.testing.assert_array_equal(expected_area.lats.data, actual_area.lats.data)

    def test_fix_modifier_attr(self):
        """Check that fix modifier can handle empty list as modifier attribute."""
        reader = SatpyCFFileHandler('filename',
                                    {},
                                    {'filetype': 'info'})
        ds_info = {'modifiers': []}
        reader.fix_modifier_attr(ds_info)
        assert ds_info['modifiers'] == ()

    def test_read_prefixed_channels(self, _cf_scene, _nc_filename):
        """Check channels starting with digit is prefixed and read back correctly."""
        _cf_scene.save_datasets(writer='cf',
                                filename=_nc_filename,
                                engine='netcdf4',
                                flatten_attrs=True,
                                pretty=True)
        scn_ = Scene(reader='satpy_cf_nc',
                     filenames=[_nc_filename])
        scn_.load(['1'])
        np.testing.assert_array_equal(scn_['1'].data, _cf_scene['1'].data)
        np.testing.assert_array_equal(scn_['1'].coords['lon'], _cf_scene['lon'].data)  # lon loaded as coord

        scn_ = Scene(reader='satpy_cf_nc',
                     filenames=[_nc_filename], reader_kwargs={})
        scn_.load(['1'])
        np.testing.assert_array_equal(scn_['1'].data, _cf_scene['1'].data)
        np.testing.assert_array_equal(scn_['1'].coords['lon'], _cf_scene['lon'].data)  # lon loaded as coord

        # Check that variables starting with a digit is written to filename variable prefixed
        with xr.open_dataset(_nc_filename) as ds_disk:
            np.testing.assert_array_equal(ds_disk['CHANNEL_1'].data, _cf_scene['1'].data)

    def test_read_prefixed_channels_include_orig_name(self, _cf_scene, _nc_filename):
        """Check channels starting with digit and includeed orig name is prefixed and read back correctly."""
        _cf_scene.save_datasets(writer='cf',
                                filename=_nc_filename,
                                engine='netcdf4',
                                flatten_attrs=True,
                                pretty=True,
                                include_orig_name=True)
        scn_ = Scene(reader='satpy_cf_nc',
                     filenames=[_nc_filename])
        scn_.load(['1'])
        np.testing.assert_array_equal(scn_['1'].data, _cf_scene['1'].data)
        np.testing.assert_array_equal(scn_['1'].coords['lon'], _cf_scene['lon'].data)  # lon loaded as coord

        assert scn_['1'].attrs['original_name'] == '1'

        # Check that variables starting with a digit is written to filename variable prefixed
        with xr.open_dataset(_nc_filename) as ds_disk:
            np.testing.assert_array_equal(ds_disk['CHANNEL_1'].data, _cf_scene['1'].data)

    def test_read_prefixed_channels_by_user(self, _cf_scene, _nc_filename):
        """Check channels starting with digit is prefixed by user and read back correctly."""
        _cf_scene.save_datasets(writer='cf',
                                filename=_nc_filename,
                                engine='netcdf4',
                                flatten_attrs=True,
                                pretty=True,
                                numeric_name_prefix='USER')
        scn_ = Scene(reader='satpy_cf_nc',
                     filenames=[_nc_filename], reader_kwargs={'numeric_name_prefix': 'USER'})
        scn_.load(['1'])
        np.testing.assert_array_equal(scn_['1'].data, _cf_scene['1'].data)
        np.testing.assert_array_equal(scn_['1'].coords['lon'], _cf_scene['lon'].data)  # lon loded as coord

        # Check that variables starting with a digit is written to filename variable prefixed
        with xr.open_dataset(_nc_filename) as ds_disk:
            np.testing.assert_array_equal(ds_disk['USER1'].data, _cf_scene['1'].data)

    def test_read_prefixed_channels_by_user2(self, _cf_scene, _nc_filename):
        """Check channels starting with digit is prefixed by user when saving and read back correctly without prefix."""
        _cf_scene.save_datasets(writer='cf',
                                filename=_nc_filename,
                                engine='netcdf4',
                                flatten_attrs=True,
                                pretty=True,
                                include_orig_name=False,
                                numeric_name_prefix='USER')
        scn_ = Scene(reader='satpy_cf_nc',
                     filenames=[_nc_filename])
        scn_.load(['USER1'])
        np.testing.assert_array_equal(scn_['USER1'].data, _cf_scene['1'].data)
        np.testing.assert_array_equal(scn_['USER1'].coords['lon'], _cf_scene['lon'].data)  # lon loded as coord

    def test_read_prefixed_channels_by_user_include_prefix(self, _cf_scene, _nc_filename):
        """Check channels starting with digit is prefixed by user and include original name when saving."""
        _cf_scene.save_datasets(writer='cf',
                                filename=_nc_filename,
                                engine='netcdf4',
                                flatten_attrs=True,
                                pretty=True,
                                include_orig_name=True,
                                numeric_name_prefix='USER')
        scn_ = Scene(reader='satpy_cf_nc',
                     filenames=[_nc_filename])
        scn_.load(['1'])
        np.testing.assert_array_equal(scn_['1'].data, _cf_scene['1'].data)
        np.testing.assert_array_equal(scn_['1'].coords['lon'], _cf_scene['lon'].data)  # lon loded as coord

    def test_read_prefixed_channels_by_user_no_prefix(self, _cf_scene, _nc_filename):
        """Check channels starting with digit is not prefixed by user."""
        _cf_scene.save_datasets(writer='cf',
                                filename=_nc_filename,
                                engine='netcdf4',
                                flatten_attrs=True,
                                pretty=True,
                                numeric_name_prefix='')
        scn_ = Scene(reader='satpy_cf_nc',
                     filenames=[_nc_filename])
        scn_.load(['1'])
        np.testing.assert_array_equal(scn_['1'].data, _cf_scene['1'].data)
        np.testing.assert_array_equal(scn_['1'].coords['lon'], _cf_scene['lon'].data)  # lon loded as coord

    def test_orbital_parameters(self, _cf_scene, _nc_filename):
        """Test that the orbital parameters in attributes are handled correctly."""
        _cf_scene.save_datasets(writer='cf',
                                filename=_nc_filename)
        scn_ = Scene(reader='satpy_cf_nc',
                     filenames=[_nc_filename])
        scn_.load(['image0'])
        orig_attrs = _cf_scene['image0'].attrs['orbital_parameters']
        new_attrs = scn_['image0'].attrs['orbital_parameters']
        assert isinstance(new_attrs, dict)
        for key in orig_attrs:
            assert orig_attrs[key] == new_attrs[key]

    def test_write_and_read_from_two_files(self, _nc_filename, _nc_filename_i):
        """Save two datasets with different resolution and read the solar_zenith_angle again."""
        _create_test_netcdf(_nc_filename, resolution=742)
        _create_test_netcdf(_nc_filename_i, resolution=371)
        scn_ = Scene(reader='satpy_cf_nc',
                     filenames=[_nc_filename, _nc_filename_i])
        scn_.load(['solar_zenith_angle'], resolution=742)
        assert scn_['solar_zenith_angle'].attrs['resolution'] == 742
        scn_.unload()
        scn_.load(['solar_zenith_angle'], resolution=371)
        assert scn_['solar_zenith_angle'].attrs['resolution'] == 371

    def test_dataid_attrs_equal_matching_dataset(self, _cf_scene, _nc_filename):
        """Check that get_dataset returns valid dataset when keys matches."""
        from satpy.dataset.dataid import DataID, default_id_keys_config
        _create_test_netcdf(_nc_filename, resolution=742)
        reader = SatpyCFFileHandler(_nc_filename, {}, {'filetype': 'info'})
        ds_id = DataID(default_id_keys_config, name='solar_zenith_angle', resolution=742, modifiers=())
        res = reader.get_dataset(ds_id, {})
        assert res.attrs['resolution'] == 742

    def test_dataid_attrs_equal_not_matching_dataset(self, _cf_scene, _nc_filename):
        """Check that get_dataset returns None when key(s) are not matching."""
        from satpy.dataset.dataid import DataID, default_id_keys_config
        _create_test_netcdf(_nc_filename, resolution=742)
        reader = SatpyCFFileHandler(_nc_filename, {}, {'filetype': 'info'})
        not_existing_resolution = 9999999
        ds_id = DataID(default_id_keys_config, name='solar_zenith_angle', resolution=not_existing_resolution,
                       modifiers=())
        assert reader.get_dataset(ds_id, {}) is None

    def test_dataid_attrs_equal_contains_not_matching_key(self, _cf_scene, _nc_filename):
        """Check that get_dataset returns valid dataset when dataid have key(s) not existing in data."""
        from satpy.dataset.dataid import DataID, default_id_keys_config
        _create_test_netcdf(_nc_filename, resolution=742)
        reader = SatpyCFFileHandler(_nc_filename, {}, {'filetype': 'info'})
        ds_id = DataID(default_id_keys_config, name='solar_zenith_angle', resolution=742,
                       modifiers=(), calibration='counts')
        res = reader.get_dataset(ds_id, {})
        assert res.attrs['resolution'] == 742
