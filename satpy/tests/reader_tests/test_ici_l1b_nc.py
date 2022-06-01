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
"""The ici_l1b_nc reader tests package.

This version tests the reader for ICI test data as per PFS V3A.

"""

import os
import unittest
import uuid
from datetime import datetime
from unittest.mock import ANY, patch

import numpy as np
import pytest
import xarray as xr
from netCDF4 import Dataset

from satpy.readers.ici_l1b_nc import IciL1bNCFileHandler

TEST_FILE = 'test_file_ici_l1b_nc.nc'


class TestIciL1bNCFileHandler(unittest.TestCase):
    """Test the IciL1bNCFileHandler reader."""

    def setUp(self):
        """Set up the test."""
        # Easiest way to test the reader is to create a test netCDF file on the
        # fly, uses a UUID to avoid permission conflicts during execution of
        # tests in parallel
        self.test_file_name = TEST_FILE + str(uuid.uuid1()) + ".nc"
        self.n_channels = 13
        self.n_scan = 10
        self.n_samples = 784
        self.n_subs = 158
        self.n_horns = 7
        self.n_183 = 3

        with Dataset(self.test_file_name, 'w') as nc:
            nc.sensing_start_time_utc = "2000-01-02 03:04:05.000"
            nc.sensing_end_time_utc = "2000-01-02 04:05:06.000"
            nc.instrument = "ICI"
            nc.spacecraft = "SGB"
            # Create quality group
            quality_group = nc.createGroup('quality')
            duration_of_product = quality_group.createVariable(
                'duration_of_product', "f4"
            )
            duration_of_product[:] = 1000.
            # Create data group
            data_group = nc.createGroup('data')
            self._setup_measurement_data_group(data_group)
            self._setup_navigation_data_group(data_group)

        self.reader = IciL1bNCFileHandler(
            filename=self.test_file_name,
            filename_info={
                'sensing_start_time': (
                    datetime.fromisoformat('2000-01-01T01:00:00')
                ),
                'sensing_end_time': (
                    datetime.fromisoformat('2000-01-01T02:00:00'),
                ),
                'creation_time': (
                    datetime.fromisoformat('2000-01-01T03:00:00'),
                ),
            },
            filetype_info={
                'cached_longitude': 'data/navigation_data/longitude',
                'cached_latitude': 'data/navigation_data/latitude',
            }
        )

    def _setup_navigation_data_group(self, dataset):
        """Set up the navigation data group."""
        group = dataset.createGroup('navigation_data')
        group.createDimension('n_scan', self.n_scan)
        group.createDimension('n_samples', self.n_samples)
        group.createDimension('n_subs', self.n_subs)
        group.createDimension('n_horns', self.n_horns)
        subs = group.createVariable('n_subs', "i4", dimensions=('n_subs',))
        subs[:] = np.arange(self.n_subs)
        longitude = group.createVariable(
            'longitude',
            np.float32,
            dimensions=('n_scan', 'n_subs', 'n_horns'),
        )
        longitude[:] = np.ones((self.n_scan, self.n_subs, self.n_horns))
        latitude = group.createVariable(
            'latitude',
            np.float32,
            dimensions=('n_scan', 'n_subs', 'n_horns'),
        )
        latitude[:] = np.ones((self.n_scan, self.n_subs, self.n_horns))
        delta_longitude = group.createVariable(
            'delta_longitude',
            np.float32,
            dimensions=('n_scan', 'n_samples', 'n_horns'),
        )
        delta_longitude[:] = 1000. * np.ones(
            (self.n_scan, self.n_samples, self.n_horns)
        )
        delta_latitude = group.createVariable(
            'delta_latitude',
            np.float32,
            dimensions=('n_scan', 'n_samples', 'n_horns'),
        )
        delta_latitude[:] = 1000. * np.ones(
            (self.n_scan, self.n_samples, self.n_horns)
        )

    def _setup_measurement_data_group(self, dataset):
        """Set up the measurement data group."""
        group = dataset.createGroup('measurement_data')
        group.createDimension('n_scan', self.n_scan)
        group.createDimension('n_samples', self.n_samples)
        group.createDimension('n_channels', self.n_channels)
        group.createDimension('n_183', self.n_183)
        scan = group.createVariable('n_scan', "i4", dimensions=('n_scan',))
        scan[:] = np.arange(self.n_scan)
        samples = group.createVariable(
            'n_samples', "i4", dimensions=('n_samples',)
        )
        samples[:] = np.arange(self.n_samples)
        bt_a = group.createVariable(
            'bt_conversion_a', np.float32, dimensions=('n_channels',)
        )
        bt_a[:] = np.ones(self.n_channels)
        bt_b = group.createVariable(
            'bt_conversion_b', np.float32, dimensions=('n_channels',)
        )
        bt_b[:] = np.zeros(self.n_channels)
        cw = group.createVariable(
            'centre_wavenumber', np.float32, dimensions=('n_channels',)
        )
        cw[:] = np.arange(self.n_channels)
        ici_radiance_183 = group.createVariable(
            'ici_radiance_183',
            np.float32,
            dimensions=('n_scan', 'n_samples', 'n_183'),
        )
        ici_radiance_183[:] = 0.08 * np.ones(
            (self.n_scan, self.n_samples, self.n_183)
        )

    def tearDown(self):
        """Remove the previously created test file."""
        # Catch Windows PermissionError for removing the created test file.
        try:
            os.remove(self.test_file_name)
        except OSError:
            pass

    def test_start_time(self):
        """Test start time."""
        assert self.reader.start_time == datetime(2000, 1, 2, 3, 4, 5)

    def test_end_time(self):
        """Test end time."""
        assert self.reader.end_time == datetime(2000, 1, 2, 4, 5, 6)

    def test_sensor(self):
        """Test sensor."""
        assert self.reader.sensor == "ICI"

    def test_spacecraft_name(self):
        """Test spacecraft name."""
        assert self.reader.spacecraft_name == "SGB"

    def test_ssp_lon(self):
        """Test sub satellite path longitude."""
        assert self.reader.ssp_lon is None

    def test_perform_calibration_raises(self):
        """Test perform calibration raises for unknown calibration method."""
        variable = xr.DataArray(np.ones(3))
        dataset_info = {'calibration': 'unknown', 'name': 'radiance'}
        with pytest.raises(ValueError, match='Unknown calibration'):
            self.reader._perform_calibration(variable, dataset_info)

    @patch('satpy.readers.ici_l1b_nc.IciL1bNCFileHandler._calibrate_bt')
    def test_perform_calibration_does_not_call_calibrate_if_not_needed(
        self,
        mocked_calibrate,
    ):
        """Test perform calibration does not call calibrate if not needed."""
        variable = xr.DataArray(
            np.array([
                [0.060, 0.065, 0.070, 0.075],
                [0.080, 0.085, 0.090, 0.095],
            ]),
            dims=('n_scan', 'n_samples'),
        )
        dataset_info = {'calibration': 'radiance'}
        self.reader._perform_calibration(variable, dataset_info)
        mocked_calibrate.assert_not_called()

    @patch('satpy.readers.ici_l1b_nc.IciL1bNCFileHandler._calibrate_bt')
    def test_perform_calibration_calls_calibrate(
        self,
        mocked_calibrate,
    ):
        """Test perform calibration calls calibrate."""
        variable = xr.DataArray(
            np.array([
                [0.060, 0.065, 0.070, 0.075],
                [0.080, 0.085, 0.090, 0.095],
            ]),
            dims=('n_scan', 'n_samples'),
        )
        dataset_info = {
            'calibration': 'brightness_temperature',
            'chan_index': 2,
        }
        self.reader._perform_calibration(variable, dataset_info)
        mocked_calibrate.assert_called_once_with(
            variable,
            2.0,
            1.0,
            0.0,
        )

    def test_calibrate_bt(self):
        """Test calibrate brightness temperature."""
        radiance = np.array([
            [0.060, 0.065, 0.070, 0.075],
            [0.080, 0.085, 0.090, 0.095],
        ])
        cw = 6.1145
        a = 1.
        b = 0.0
        bt = self.reader._calibrate_bt(radiance, cw, a, b)
        expected_bt = np.array([
            [198.22929022, 214.38700287, 230.54437184, 246.70146465],
            [262.85833223, 279.01501371, 295.17153966, 311.32793429],
        ])
        assert np.allclose(bt, expected_bt)

    def test_standardize_dims(self):
        """Test standardize dims."""
        variable = xr.DataArray(
            np.arange(6).reshape(2, 3),
            dims=('n_scan', 'n_samples'),
        )
        standardized = self.reader._standardize_dims(variable)
        assert standardized.dims == ('y', 'x')

    def test_filter_variable(self):
        """Test filter variable."""
        data = np.arange(24).reshape(2, 3, 4)
        variable = xr.DataArray(
            np.arange(24).reshape(2, 3, 4),
            dims=('y', 'x', 'n_horns'),
        )
        dataset_info = {"n_horns": 1}
        filtered = self.reader._filter_variable(variable, dataset_info)
        assert (filtered == data[:, :, 1]).all()

    def test_drop_coords(self):
        """Test drop coordinates."""
        coords = "dummy"
        data = xr.DataArray(
            np.ones(10),
            dims=('y'),
            coords={coords: 0},
        )
        assert coords in data.coords
        data = self.reader._drop_coords(data, coords)
        assert coords not in data.coords

    def test_get_dataset_return_none_if_data_not_exist(self):
        """Tes get dataset return none if data does not exist."""
        dataset_id = {'name': 'unknown'}
        dataset_info = {'file_key': 'non/existing/data'}
        dataset = self.reader.get_dataset(dataset_id, dataset_info)
        assert dataset is None

    @patch('satpy.readers.ici_l1b_nc.IciL1bNCFileHandler._calibrate_bt')
    def test_get_dataset_does_not_calibrate_if_not_desired(
        self,
        mocked_calibrate,
    ):
        """Test get dataset does not calibrate if not desired."""
        dataset_id = {'name': 'lon_pixels_horn_1'}
        dataset_info = {
            'name': 'lon_pixels_horn_1',
            'file_type': 'nc_ici_l1b_rad',
            'file_key': 'cached_longitude',
            'standard_name': 'longitude',
            'n_horns': 0,
            'modifiers': (),
        }
        dataset = self.reader.get_dataset(dataset_id, dataset_info)
        assert dataset.dims == ('y', 'x')
        mocked_calibrate.assert_not_called()
        assert isinstance(dataset, xr.DataArray)

    @patch('satpy.readers.ici_l1b_nc.IciL1bNCFileHandler._perform_orthorectification')  # noqa: E501
    def test_get_dataset_orthorectifies_if_orthorect_data_defined(
        self,
        mocked_orthorectification,
    ):
        """Test get dataset orthorectifies if orthorect data is defined."""
        dataset_id = {'name': 'lon_pixels_horn_1'}
        dataset_info = {
            'name': 'lon_pixels_horn_1',
            'file_type': 'nc_ici_l1b_rad',
            'file_key': 'cached_longitude',
            'orthorect_data': 'data/navigation_data/delta_longitude',
            'standard_name': 'longitude',
            'n_horns': 0,
            'modifiers': (),
        }
        self.reader.get_dataset(dataset_id, dataset_info)
        mocked_orthorectification.assert_called_once_with(
            ANY,  # the data
            dataset_info['orthorect_data'],
        )

    @patch('satpy.readers.ici_l1b_nc.IciL1bNCFileHandler._perform_calibration')
    def test_get_dataset_handles_calibration(
        self,
        mocked_calibration,
    ):
        """Test get dataset handles calibration."""
        dataset_id = {'name': 'ici_1'}
        dataset_info = {
            'name': 'ici_1',
            'file_type': 'nc_ici_l1b_rad',
            'file_key': 'data/measurement_data/ici_radiance_183',
            'coordinates': ['lat_pixels_horn_1', 'lon_pixels_horn_1'],
            'n_183': 0,
            'n_horns': 0,
            'chan_index': 0,
            'calibration': 'brightness_temperature',
        }
        self.reader.get_dataset(dataset_id, dataset_info)
        mocked_calibration.assert_called_once_with(
            ANY,  # the data
            dataset_info,
        )

    def test_perform_geo_interpolation(self):
        """Test perform geo interpolation."""
        longitude = xr.DataArray(
            np.ones((self.n_scan, self.n_subs, self.n_horns)),
            dims=('n_scan', 'n_subs', 'n_horns'),
            coords={
                'n_horns': np.arange(self.n_horns),
                'n_subs': np.arange(self.n_subs),
            },
        )
        latitude = xr.DataArray(
            2. * np.ones((self.n_scan, self.n_subs, self.n_horns)),
            dims=('n_scan', 'n_subs', 'n_horns'),
        )
        lon, lat = self.reader._perform_geo_interpolation(
            longitude,
            latitude,
            self.n_samples,
        )
        assert lon.shape == (self.n_scan, self.n_samples, self.n_horns)
        assert lat.shape == (self.n_scan, self.n_samples, self.n_horns)
        assert np.allclose(lon, 1.0)
        assert np.allclose(lat, 2.0)

    def test_perform_viewing_angle_interpolation(self):
        """Test perform viewing angle interpolation."""
        azimuth = xr.DataArray(
            np.ones((self.n_scan, self.n_subs, self.n_horns)),
            dims=('n_scan', 'n_subs', 'n_horns'),
            coords={
                'n_horns': np.arange(self.n_horns),
                'n_subs': np.arange(self.n_subs),
            },
        )
        zenith = xr.DataArray(
            100. * np.ones((self.n_scan, self.n_subs, self.n_horns)),
            dims=('n_scan', 'n_subs', 'n_horns'),
            coords={
                'n_horns': np.arange(self.n_horns),
                'n_subs': np.arange(self.n_subs),
            },
        )
        azimuth, zenith = self.reader._perform_viewing_angle_interpolation(
            azimuth,
            zenith,
            self.n_samples,
        )
        assert azimuth.shape == (self.n_scan, self.n_samples, self.n_horns)
        assert zenith.shape == (self.n_scan, self.n_samples, self.n_horns)
        assert np.allclose(azimuth, 1.0)
        assert np.allclose(zenith, 100.0)

    def test_perform_orthorectification(self):
        """Test perform orthorectification."""
        variable = xr.DataArray(
            np.ones((self.n_scan, self.n_samples, self.n_horns)),
            dims=('y', 'x', 'n_horns'),
            coords={'n_horns': np.arange(self.n_horns)}
        )
        variable = variable.sel({'n_horns': 0})
        orthorect_data_name = 'data/navigation_data/delta_longitude'
        orthorectified = self.reader._perform_orthorectification(
            variable,
            orthorect_data_name,
        )
        assert np.allclose(orthorectified, 1.009)

    def test_get_global_attributes(self):
        """Test get global attributes."""
        attributes = self.reader._get_global_attributes()
        assert attributes['quality_group'] == {
            'duration_of_product': np.array(1000., dtype=np.float32)
        }
