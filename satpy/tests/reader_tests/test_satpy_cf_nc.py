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

import os
import unittest
from contextlib import suppress
from datetime import datetime

import numpy as np
import xarray as xr

from satpy import Scene
from satpy.dataset.dataid import WavelengthRange
from satpy.readers.satpy_cf_nc import SatpyCFFileHandler


class TestCFReader(unittest.TestCase):
    """Test case for CF reader."""

    def setUp(self):
        """Create a test scene."""
        tstart = datetime(2019, 4, 1, 12, 0)
        tend = datetime(2019, 4, 1, 12, 15)
        data_visir = [[1, 2], [3, 4]]
        y_visir = [1, 2]
        x_visir = [1, 2]
        z_visir = [1, 2, 3, 4, 5, 6, 7]
        qual_data = [[1, 2, 3, 4, 5, 6, 7],
                     [1, 2, 3, 4, 5, 6, 7]]
        time_vis006 = [1, 2]
        lat = 33.0 * np.array([[1, 2], [3, 4]])
        lon = -13.0 * np.array([[1, 2], [3, 4]])
        common_attrs = {'start_time': tstart,
                        'end_time': tend,
                        'platform_name': 'tirosn',
                        'orbit_number': 99999}
        vis006 = xr.DataArray(data_visir,
                              dims=('y', 'x'),
                              coords={'y': y_visir, 'x': x_visir, 'acq_time': ('y', time_vis006)},
                              attrs={'name': 'image0', 'id_tag': 'ch_r06',
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
                                        'only_in_1': False}
                                     })

        ir_108 = xr.DataArray(data_visir,
                              dims=('y', 'x'),
                              coords={'y': y_visir, 'x': x_visir, 'acq_time': ('y', time_vis006)},
                              attrs={'name': 'image1', 'id_tag': 'ch_tb11', 'coordinates': 'lat lon'})
        qual_f = xr.DataArray(qual_data,
                              dims=('y', 'z'),
                              coords={'y': y_visir, 'z': z_visir, 'acq_time': ('y', time_vis006)},
                              attrs={'name': 'qual_flags',
                                     'id_tag': 'qual_flags'})
        lat = xr.DataArray(lat,
                           dims=('y', 'x'),
                           coords={'y': y_visir, 'x': x_visir},
                           attrs={'name': 'lat',
                                  'standard_name': 'latitude',
                                  'modifiers': np.array([])})
        lon = xr.DataArray(lon,
                           dims=('y', 'x'),
                           coords={'y': y_visir, 'x': x_visir},
                           attrs={'name': 'lon',
                                  'standard_name': 'longitude',
                                  'modifiers': np.array([])})
        self.scene = Scene()
        self.scene.attrs['sensor'] = ['avhrr-1', 'avhrr-2', 'avhrr-3']
        scene_dict = {'image0': vis006,
                      'image1': ir_108,
                      'lat': lat,
                      'lon': lon,
                      'qual_flags': qual_f}
        for key in scene_dict:
            self.scene[key] = scene_dict[key]
            self.scene[key].attrs.update(common_attrs)

    def test_write_and_read(self):
        """Save a file with cf_writer and read the data again."""
        filename = 'testingcfwriter{:s}-viirs-mband-20201007075915-20201007080744.nc'.format(
            datetime.utcnow().strftime('%Y%j%H%M%S'))
        try:
            self.scene.save_datasets(writer='cf',
                                     filename=filename,
                                     header_attrs={'instrument': 'avhrr'},
                                     engine='h5netcdf',
                                     flatten_attrs=True,
                                     pretty=True)
            scn_ = Scene(reader='satpy_cf_nc',
                         filenames=[filename])
            scn_.load(['image0', 'image1', 'lat'])
            np.testing.assert_array_equal(scn_['image0'].data, self.scene['image0'].data)
            np.testing.assert_array_equal(scn_['lat'].data, self.scene['lat'].data)  # lat loaded as dataset
            np.testing.assert_array_equal(scn_['image0'].coords['lon'], self.scene['lon'].data)  # lon loded as coord
            assert isinstance(scn_['image0'].attrs['wavelength'], WavelengthRange)
        finally:
            with suppress(PermissionError):
                os.remove(filename)

    def test_fix_modifier_attr(self):
        """Check that fix modifier can handle empty list as modifier attribute."""
        self.reader = SatpyCFFileHandler('filename',
                                         {},
                                         {'filetype': 'info'})
        ds_info = {'modifiers': []}
        self.reader.fix_modifier_attr(ds_info)
        self.assertEqual(ds_info['modifiers'], ())

    def _dataset_for_prefix_testing(self):
        data_visir = [[1, 2], [3, 4]]
        y_visir = [1, 2]
        x_visir = [1, 2]
        lat = 33.0 * np.array([[1, 2], [3, 4]])
        lon = -13.0 * np.array([[1, 2], [3, 4]])
        vis006 = xr.DataArray(data_visir,
                              dims=('y', 'x'),
                              coords={'y': y_visir, 'x': x_visir},
                              attrs={'name': '1', 'id_tag': 'ch_r06',
                                     'coordinates': 'lat lon', 'resolution': 1000, 'calibration': 'reflectance',
                                     'wavelength': WavelengthRange(min=0.58, central=0.63, max=0.68, unit='µm')
                                     })
        lat = xr.DataArray(lat,
                           dims=('y', 'x'),
                           coords={'y': y_visir, 'x': x_visir},
                           attrs={'name': 'lat',
                                  'standard_name': 'latitude',
                                  'modifiers': np.array([])})
        lon = xr.DataArray(lon,
                           dims=('y', 'x'),
                           coords={'y': y_visir, 'x': x_visir},
                           attrs={'name': 'lon',
                                  'standard_name': 'longitude',
                                  'modifiers': np.array([])})
        scene = Scene()
        scene.attrs['sensor'] = ['avhrr-1', 'avhrr-2', 'avhrr-3']
        scene['1'] = vis006
        scene['lat'] = lat
        scene['lon'] = lon

        return scene

    def test_read_prefixed_channels(self):
        """Check channels starting with digit is prefixed and read back correctly."""
        scene = self._dataset_for_prefix_testing()
        # Testing with default prefixing
        filename = 'testingcfwriter{:s}-viirs-mband-20201007075915-20201007080744.nc'.format(
            datetime.utcnow().strftime('%Y%j%H%M%S'))
        try:
            scene.save_datasets(writer='cf',
                                filename=filename,
                                header_attrs={'instrument': 'avhrr'},
                                engine='netcdf4',
                                flatten_attrs=True,
                                pretty=True)
            scn_ = Scene(reader='satpy_cf_nc',
                         filenames=[filename])
            scn_.load(['1'])
            np.testing.assert_array_equal(scn_['1'].data, scene['1'].data)
            np.testing.assert_array_equal(scn_['1'].coords['lon'], scene['lon'].data)  # lon loaded as coord

            scn_ = Scene(reader='satpy_cf_nc',
                         filenames=[filename], reader_kwargs={})
            scn_.load(['1'])
            np.testing.assert_array_equal(scn_['1'].data, scene['1'].data)
            np.testing.assert_array_equal(scn_['1'].coords['lon'], scene['lon'].data)  # lon loaded as coord

            # Check that variables starting with a digit is written to filename variable prefixed
            with xr.open_dataset(filename) as ds_disk:
                np.testing.assert_array_equal(ds_disk['CHANNEL_1'].data, scene['1'].data)
        finally:
            with suppress(PermissionError):
                os.remove(filename)

    def test_read_prefixed_channels_include_orig_name(self):
        """Check channels starting with digit and includeed orig name is prefixed and read back correctly."""
        scene = self._dataset_for_prefix_testing()
        # Testing with default prefixing
        filename = 'testingcfwriter{:s}-viirs-mband-20201007075915-20201007080744.nc'.format(
            datetime.utcnow().strftime('%Y%j%H%M%S'))
        try:
            scene.save_datasets(writer='cf',
                                filename=filename,
                                header_attrs={'instrument': 'avhrr'},
                                engine='netcdf4',
                                flatten_attrs=True,
                                pretty=True,
                                include_orig_name=True)
            scn_ = Scene(reader='satpy_cf_nc',
                         filenames=[filename])
            scn_.load(['1'])
            np.testing.assert_array_equal(scn_['1'].data, scene['1'].data)
            np.testing.assert_array_equal(scn_['1'].coords['lon'], scene['lon'].data)  # lon loaded as coord

            self.assertEqual(scn_['1'].attrs['original_name'], '1')

            # Check that variables starting with a digit is written to filename variable prefixed
            with xr.open_dataset(filename) as ds_disk:
                np.testing.assert_array_equal(ds_disk['CHANNEL_1'].data, scene['1'].data)
        finally:
            with suppress(PermissionError):
                os.remove(filename)

    def test_read_prefixed_channels_by_user(self):
        """Check channels starting with digit is prefixed by user and read back correctly."""
        scene = self._dataset_for_prefix_testing()
        filename = 'testingcfwriter{:s}-viirs-mband-20201007075915-20201007080744.nc'.format(
            datetime.utcnow().strftime('%Y%j%H%M%S'))
        try:
            scene.save_datasets(writer='cf',
                                filename=filename,
                                header_attrs={'instrument': 'avhrr'},
                                engine='netcdf4',
                                flatten_attrs=True,
                                pretty=True,
                                numeric_name_prefix='USER')
            scn_ = Scene(reader='satpy_cf_nc',
                         filenames=[filename], reader_kwargs={'numeric_name_prefix': 'USER'})
            scn_.load(['1'])
            np.testing.assert_array_equal(scn_['1'].data, scene['1'].data)
            np.testing.assert_array_equal(scn_['1'].coords['lon'], scene['lon'].data)  # lon loded as coord

            # Check that variables starting with a digit is written to filename variable prefixed
            with xr.open_dataset(filename) as ds_disk:
                np.testing.assert_array_equal(ds_disk['USER1'].data, scene['1'].data)
        finally:
            with suppress(PermissionError):
                os.remove(filename)

    def test_read_prefixed_channels_by_user2(self):
        """Check channels starting with digit is prefixed by user when saving and read back correctly without prefix."""
        scene = self._dataset_for_prefix_testing()
        filename = 'testingcfwriter{:s}-viirs-mband-20201007075915-20201007080744.nc'.format(
            datetime.utcnow().strftime('%Y%j%H%M%S'))
        try:
            scene.save_datasets(writer='cf',
                                filename=filename,
                                header_attrs={'instrument': 'avhrr'},
                                engine='netcdf4',
                                flatten_attrs=True,
                                pretty=True,
                                include_orig_name=False,
                                numeric_name_prefix='USER')
            scn_ = Scene(reader='satpy_cf_nc',
                         filenames=[filename])
            scn_.load(['USER1'])
            np.testing.assert_array_equal(scn_['USER1'].data, scene['1'].data)
            np.testing.assert_array_equal(scn_['USER1'].coords['lon'], scene['lon'].data)  # lon loded as coord

        finally:
            with suppress(PermissionError):
                os.remove(filename)

    def test_read_prefixed_channels_by_user_include_prefix(self):
        """Check channels starting with digit is prefixed by user and include original name when saving."""
        scene = self._dataset_for_prefix_testing()
        filename = 'testingcfwriter2{:s}-viirs-mband-20201007075915-20201007080744.nc'.format(
            datetime.utcnow().strftime('%Y%j%H%M%S'))
        try:
            scene.save_datasets(writer='cf',
                                filename=filename,
                                header_attrs={'instrument': 'avhrr'},
                                engine='netcdf4',
                                flatten_attrs=True,
                                pretty=True,
                                include_orig_name=True,
                                numeric_name_prefix='USER')
            scn_ = Scene(reader='satpy_cf_nc',
                         filenames=[filename])
            scn_.load(['1'])
            np.testing.assert_array_equal(scn_['1'].data, scene['1'].data)
            np.testing.assert_array_equal(scn_['1'].coords['lon'], scene['lon'].data)  # lon loded as coord

        finally:
            with suppress(PermissionError):
                os.remove(filename)

    def test_read_prefixed_channels_by_user_no_prefix(self):
        """Check channels starting with digit is not prefixed by user."""
        scene = self._dataset_for_prefix_testing()
        filename = 'testingcfwriter3{:s}-viirs-mband-20201007075915-20201007080744.nc'.format(
            datetime.utcnow().strftime('%Y%j%H%M%S'))
        try:
            scene.save_datasets(writer='cf',
                                filename=filename,
                                header_attrs={'instrument': 'avhrr'},
                                engine='netcdf4',
                                flatten_attrs=True,
                                pretty=True,
                                numeric_name_prefix='')
            scn_ = Scene(reader='satpy_cf_nc',
                         filenames=[filename])
            scn_.load(['1'])
            np.testing.assert_array_equal(scn_['1'].data, scene['1'].data)
            np.testing.assert_array_equal(scn_['1'].coords['lon'], scene['lon'].data)  # lon loded as coord

        finally:
            with suppress(PermissionError):
                os.remove(filename)

    def test_orbital_parameters(self):
        """Test that the orbital parameters in attributes are handled correctly."""
        filename = 'testingcfwriter4{:s}-viirs-mband-20201007075915-20201007080744.nc'.format(
            datetime.utcnow().strftime('%Y%j%H%M%S'))
        try:
            self.scene.save_datasets(writer='cf',
                                     filename=filename,
                                     header_attrs={'instrument': 'avhrr'})
            scn_ = Scene(reader='satpy_cf_nc',
                         filenames=[filename])
            scn_.load(['image0'])
            orig_attrs = self.scene['image0'].attrs['orbital_parameters']
            new_attrs = scn_['image0'].attrs['orbital_parameters']
            assert isinstance(new_attrs, dict)
            for key in orig_attrs:
                assert orig_attrs[key] == new_attrs[key]
        finally:
            with suppress(PermissionError):
                os.remove(filename)
