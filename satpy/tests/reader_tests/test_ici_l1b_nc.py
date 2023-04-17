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

from datetime import datetime
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from netCDF4 import Dataset

from satpy.readers.ici_l1b_nc import IciL1bNCFileHandler, InterpolationType

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - tmp_path

N_CHANNELS = 13
N_SCAN = 10
N_SAMPLES = 784
N_SUBS = 158
N_HORNS = 7
N_183 = 3


@pytest.fixture
def reader(fake_file):
    """Return reader of ici level1b data."""
    return IciL1bNCFileHandler(
        filename=fake_file,
        filename_info={
            'sensing_start_time': (
                datetime.fromisoformat('2000-01-01T01:00:00')
            ),
            'sensing_end_time': (
                datetime.fromisoformat('2000-01-01T02:00:00')
            ),
            'creation_time': (
                datetime.fromisoformat('2000-01-01T03:00:00')
            ),
        },
        filetype_info={
            'longitude': 'data/navigation_data/longitude',
            'latitude': 'data/navigation_data/latitude',
            'solar_azimuth': 'data/navigation_data/ici_solar_azimuth_angle',
            'solar_zenith': 'data/navigation_data/ici_solar_zenith_angle',
        }
    )


@pytest.fixture
def fake_file(tmp_path):
    """Return file path to level1b file."""
    file_path = tmp_path / 'test_file_ici_l1b_nc.nc'
    writer = IciL1bFakeFileWriter(file_path)
    writer.write()
    yield file_path


@pytest.fixture
def dataset_info():
    """Return dataset info."""
    return {
        'name': '1',
        'file_type': 'nc_ici_l1b_rad',
        'file_key': 'data/measurement_data/ici_radiance_183',
        'coordinates': ['lat_pixels_horn_1', 'lon_pixels_horn_1'],
        'n_183': 0,
        'chan_index': 0,
        'calibration': 'brightness_temperature',
    }


class IciL1bFakeFileWriter:
    """Writer class of fake ici level1b data."""

    def __init__(self, file_path):
        """Init."""
        self.file_path = file_path

    def write(self):
        """Write fake data to file."""
        with Dataset(self.file_path, 'w') as dataset:
            self._write_attributes(dataset)
            self._write_quality_group(dataset)
            data_group = dataset.createGroup('data')
            self._write_measurement_data_group(data_group)
            self._write_navigation_data_group(data_group)

    @staticmethod
    def _write_attributes(dataset):
        """Write attributes."""
        dataset.sensing_start_time_utc = "2000-01-02 03:04:05.000"
        dataset.sensing_end_time_utc = "2000-01-02 04:05:06.000"
        dataset.instrument = "ICI"
        dataset.spacecraft = "SGB"

    @staticmethod
    def _write_quality_group(dataset):
        """Write the quality group."""
        group = dataset.createGroup('quality')
        group.overall_quality_flag = 0
        duration_of_product = group.createVariable(
            'duration_of_product', "f4"
        )
        duration_of_product[:] = 1000.

    @staticmethod
    def _write_navigation_data_group(dataset):
        """Write the navigation data group."""
        group = dataset.createGroup('navigation_data')
        group.createDimension('n_scan', N_SCAN)
        group.createDimension('n_samples', N_SAMPLES)
        group.createDimension('n_subs', N_SUBS)
        group.createDimension('n_horns', N_HORNS)
        subs = group.createVariable('n_subs', "i4", dimensions=('n_subs',))
        subs[:] = np.arange(N_SUBS)
        dimensions = ('n_scan', 'n_subs', 'n_horns')
        shape = (N_SCAN, N_SUBS, N_HORNS)
        longitude = group.createVariable(
            'longitude',
            np.float32,
            dimensions=dimensions,
        )
        longitude[:] = np.ones(shape)
        latitude = group.createVariable(
            'latitude',
            np.float32,
            dimensions=dimensions,
        )
        latitude[:] = 2. * np.ones(shape)
        azimuth = group.createVariable(
            'ici_solar_azimuth_angle',
            np.float32,
            dimensions=dimensions,
        )
        azimuth[:] = 3. * np.ones(shape)
        zenith = group.createVariable(
            'ici_solar_zenith_angle',
            np.float32,
            dimensions=dimensions,
        )
        zenith[:] = 4. * np.ones(shape)
        dimensions = ('n_scan', 'n_samples', 'n_horns')
        shape = (N_SCAN, N_SAMPLES, N_HORNS)
        delta_longitude = group.createVariable(
            'delta_longitude',
            np.float32,
            dimensions=dimensions,
        )
        delta_longitude[:] = 1000. * np.ones(shape)
        delta_latitude = group.createVariable(
            'delta_latitude',
            np.float32,
            dimensions=dimensions,
        )
        delta_latitude[:] = 1000. * np.ones(shape)

    @staticmethod
    def _write_measurement_data_group(dataset):
        """Write the measurement data group."""
        group = dataset.createGroup('measurement_data')
        group.createDimension('n_scan', N_SCAN)
        group.createDimension('n_samples', N_SAMPLES)
        group.createDimension('n_channels', N_CHANNELS)
        group.createDimension('n_183', N_183)
        scan = group.createVariable('n_scan', "i4", dimensions=('n_scan',))
        scan[:] = np.arange(N_SCAN)
        samples = group.createVariable(
            'n_samples', "i4", dimensions=('n_samples',)
        )
        samples[:] = np.arange(N_SAMPLES)
        bt_a = group.createVariable(
            'bt_conversion_a', np.float32, dimensions=('n_channels',)
        )
        bt_a[:] = np.ones(N_CHANNELS)
        bt_b = group.createVariable(
            'bt_conversion_b', np.float32, dimensions=('n_channels',)
        )
        bt_b[:] = np.zeros(N_CHANNELS)
        cw = group.createVariable(
            'centre_wavenumber', np.float32, dimensions=('n_channels',)
        )
        cw[:] = np.array(
            [6.0] * 3 + [8.0] * 2 + [11.0] * 3 + [15.0] * 3 + [22.0] * 2
        )
        ici_radiance_183 = group.createVariable(
            'ici_radiance_183',
            np.float32,
            dimensions=('n_scan', 'n_samples', 'n_183'),
        )
        ici_radiance_183[:] = 0.08 * np.ones((N_SCAN, N_SAMPLES, N_183))


class TestIciL1bNCFileHandler:
    """Test the IciL1bNCFileHandler reader."""

    def test_start_time(self, reader):
        """Test start time."""
        assert reader.start_time == datetime(2000, 1, 2, 3, 4, 5)

    def test_end_time(self, reader):
        """Test end time."""
        assert reader.end_time == datetime(2000, 1, 2, 4, 5, 6)

    def test_sensor(self, reader):
        """Test sensor."""
        assert reader.sensor == "ICI"

    def test_platform_name(self, reader):
        """Test platform name."""
        assert reader.platform_name == "SGB"

    def test_ssp_lon(self, reader):
        """Test sub satellite path longitude."""
        assert reader.ssp_lon is None

    def test_longitude(self, reader):
        """Test longitude."""
        np.testing.assert_allclose(reader.longitude, 1, rtol=1e-3)

    def test_latitude(self, reader):
        """Test latitude."""
        np.testing.assert_allclose(reader.latitude, 2, rtol=1e-3)

    def test_solar_azimuth(self, reader):
        """Test solar azimuth."""
        np.testing.assert_allclose(reader.solar_azimuth, 3, rtol=1e-3)

    def test_solar_zenith(self, reader):
        """Test solar zenith."""
        np.testing.assert_allclose(reader.solar_zenith, 4, rtol=1e-3)

    def test_calibrate_raises_for_unknown_calibration_method(self, reader):
        """Test perform calibration raises for unknown calibration method."""
        variable = xr.DataArray(np.ones(3))
        dataset_info = {'calibration': 'unknown', 'name': 'radiance'}
        with pytest.raises(ValueError, match='Unknown calibration'):
            reader._calibrate(variable, dataset_info)

    @patch('satpy.readers.ici_l1b_nc.IciL1bNCFileHandler._calibrate_bt')
    def test_calibrate_does_not_call_calibrate_bt_if_not_needed(
        self,
        mocked_calibrate,
        reader,
    ):
        """Test calibrate does not call calibrate_bt if not needed."""
        variable = xr.DataArray(
            np.array([
                [0.060, 0.065, 0.070, 0.075],
                [0.080, 0.085, 0.090, 0.095],
            ]),
            dims=('n_scan', 'n_samples'),
        )
        dataset_info = {'calibration': 'radiance'}
        reader._calibrate(variable, dataset_info)
        mocked_calibrate.assert_not_called()

    @patch('satpy.readers.ici_l1b_nc.IciL1bNCFileHandler._calibrate_bt')
    def test_calibrate_calls_calibrate_bt(
        self,
        mocked_calibrate_bt,
        reader,
    ):
        """Test calibrate calls calibrate_bt."""
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
        reader._calibrate(variable, dataset_info)
        mocked_calibrate_bt.assert_called_once_with(
            variable,
            6.0,
            1.0,
            0.0,
        )

    def test_calibrate_bt(self, reader):
        """Test calibrate brightness temperature."""
        radiance = xr.DataArray(
            np.array([
                [0.060, 0.065, 0.070, 0.075],
                [0.080, 0.085, 0.090, 0.095],
            ])
        )
        cw = 6.1145
        a = 1.
        b = 0.0
        bt = reader._calibrate_bt(radiance, cw, a, b)
        expected_bt = np.array([
            [198.22929022, 214.38700287, 230.54437184, 246.70146465],
            [262.85833223, 279.01501371, 295.17153966, 311.32793429],
        ])
        np.testing.assert_allclose(bt, expected_bt)

    @pytest.mark.parametrize('dims', (
        ('n_scan', 'n_samples'),
        ('x', 'y'),
    ))
    def test_standardize_dims(self, reader, dims):
        """Test standardize dims."""
        variable = xr.DataArray(
            np.arange(6).reshape(2, 3),
            dims=dims,
        )
        standardized = reader._standardize_dims(variable)
        assert standardized.dims == ('y', 'x')

    @pytest.mark.parametrize('dims,data_info,expect', (
        (('y', 'x', 'n_horns'), {"n_horns": 1}, 1),
        (('y', 'x', 'n_183'), {"n_183": 2}, 2),
    ))
    def test_filter_variable(self, reader, dims, data_info, expect):
        """Test filter variable."""
        data = np.arange(24).reshape(2, 3, 4)
        variable = xr.DataArray(
            np.arange(24).reshape(2, 3, 4),
            dims=dims,
        )
        filtered = reader._filter_variable(variable, data_info)
        assert filtered.dims == ('y', 'x')
        assert (filtered == data[:, :, expect]).all()

    def test_drop_coords(self, reader):
        """Test drop coordinates."""
        coords = "dummy"
        data = xr.DataArray(
            np.ones(10),
            dims=('y'),
            coords={coords: 0},
        )
        assert coords in data.coords
        data = reader._drop_coords(data)
        assert coords not in data.coords

    def test_get_third_dimension_name(self, reader):
        """Test get third dimension name."""
        data = xr.DataArray(np.ones((1, 1, 1)), dims=('x', 'y', 'z'))
        assert reader._get_third_dimension_name(data) == 'z'

    def test_get_third_dimension_name_return_none_for_2d_data(self, reader):
        """Test get third dimension name return none for 2d data."""
        data = xr.DataArray(np.ones((1, 1)), dims=('x', 'y'))
        assert reader._get_third_dimension_name(data) is None

    def test_get_dataset_return_none_if_data_not_exist(self, reader):
        """Tes get dataset return none if data does not exist."""
        dataset_id = {'name': 'unknown'}
        dataset_info = {'file_key': 'non/existing/data'}
        dataset = reader.get_dataset(dataset_id, dataset_info)
        assert dataset is None

    @patch('satpy.readers.ici_l1b_nc.IciL1bNCFileHandler._calibrate_bt')
    def test_get_dataset_does_not_calibrate_if_not_desired(
        self,
        mocked_calibrate,
        reader,
        dataset_info,
    ):
        """Test get dataset does not calibrate if not desired."""
        dataset_id = {'name': '1'}
        dataset_info.pop('calibration')
        dataset = reader.get_dataset(dataset_id, dataset_info)
        assert dataset.dims == ('y', 'x')
        mocked_calibrate.assert_not_called()
        assert isinstance(dataset, xr.DataArray)

    def test_get_dataset_orthorectifies_if_orthorect_data_defined(
        self,
        reader,
    ):
        """Test get dataset orthorectifies if orthorect data is defined."""
        dataset_id = {'name': 'lon_pixels_horn_1'}
        dataset_info = {
            'name': 'lon_pixels_horn_1',
            'file_type': 'nc_ici_l1b_rad',
            'file_key': 'longitude',
            'orthorect_data': 'data/navigation_data/delta_longitude',
            'standard_name': 'longitude',
            'n_horns': 0,
            'modifiers': (),
        }
        dataset = reader.get_dataset(dataset_id, dataset_info)
        np.testing.assert_allclose(dataset, 1.009139, atol=1e-6)

    def test_get_dataset_handles_calibration(
        self,
        reader,
        dataset_info,
    ):
        """Test get dataset handles calibration."""
        dataset_id = {'name': '1'}
        dataset = reader.get_dataset(dataset_id, dataset_info)
        assert dataset.attrs["calibration"] == "brightness_temperature"
        np.testing.assert_allclose(dataset, 272.73734)

    def test_interpolate_returns_none_if_dataset_not_exist(self, reader):
        """Test interpolate returns none if dataset not exist."""
        azimuth, zenith = reader._interpolate(
            InterpolationType.OBSERVATION_ANGLES
        )
        assert azimuth is None and zenith is None

    @patch('satpy.readers.ici_l1b_nc.IciL1bNCFileHandler._interpolate_geo')
    def test_interpolate_calls_interpolate_geo(self, mock, reader):
        """Test interpolate calls interpolate_geo."""
        reader._interpolate(InterpolationType.LONLAT)
        mock.assert_called_once()

    @patch('satpy.readers.ici_l1b_nc.IciL1bNCFileHandler._interpolate_viewing_angle')  # noqa: E501
    def test_interpolate_calls_interpolate_viewing_angles(self, mock, reader):
        """Test interpolate calls interpolate viewing_angles."""
        reader._interpolate(InterpolationType.SOLAR_ANGLES)
        mock.assert_called_once()

    def test_interpolate_geo(self, reader):
        """Test interpolate geographic coordinates."""
        shape = (N_SCAN, N_SUBS, N_HORNS)
        dims = ('n_scan', 'n_subs', 'n_horns')
        longitude = xr.DataArray(
            2. * np.ones(shape),
            dims=dims,
            coords={
                'n_horns': np.arange(N_HORNS),
                'n_subs': np.arange(N_SUBS),
            },
        )
        latitude = xr.DataArray(np.ones(shape), dims=dims)
        lon, lat = reader._interpolate_geo(
            longitude,
            latitude,
            N_SAMPLES,
        )
        expect_shape = (N_SCAN, N_SAMPLES, N_HORNS)
        assert lon.shape == expect_shape
        assert lat.shape == expect_shape
        np.testing.assert_allclose(lon, 2.0)
        np.testing.assert_allclose(lat, 1.0)

    def test_interpolate_viewing_angle(self, reader):
        """Test interpolate viewing angle."""
        shape = (N_SCAN, N_SUBS, N_HORNS)
        dims = ('n_scan', 'n_subs', 'n_horns')
        azimuth = xr.DataArray(
            np.ones(shape),
            dims=dims,
            coords={
                'n_horns': np.arange(N_HORNS),
                'n_subs': np.arange(N_SUBS),
            },
        )
        zenith = xr.DataArray(100. * np.ones(shape), dims=dims)
        azimuth, zenith = reader._interpolate_viewing_angle(
            azimuth,
            zenith,
            N_SAMPLES,
        )
        expect_shape = (N_SCAN, N_SAMPLES, N_HORNS)
        assert azimuth.shape == expect_shape
        assert zenith.shape == expect_shape
        np.testing.assert_allclose(azimuth, 1.0)
        np.testing.assert_allclose(zenith, 100.0)

    def test_orthorectify(self, reader):
        """Test orthorectify."""
        variable = xr.DataArray(
            np.ones((N_SCAN, N_SAMPLES, N_HORNS)),
            dims=('y', 'x', 'n_horns'),
            coords={'n_horns': np.arange(N_HORNS)}
        )
        variable = variable.sel({'n_horns': 0})
        orthorect_data_name = 'data/navigation_data/delta_longitude'
        orthorectified = reader._orthorectify(
            variable,
            orthorect_data_name,
        )
        np.testing.assert_allclose(orthorectified, 1.009, rtol=1e-5)

    def test_get_global_attributes(self, reader):
        """Test get global attributes."""
        attributes = reader._get_global_attributes()
        assert attributes == {
            'filename': reader.filename,
            'start_time': datetime(2000, 1, 2, 3, 4, 5),
            'end_time': datetime(2000, 1, 2, 4, 5, 6),
            'spacecraft_name': 'SGB',
            'ssp_lon': None,
            'sensor': 'ICI',
            'filename_start_time': datetime(2000, 1, 1, 1, 0),
            'filename_end_time': datetime(2000, 1, 1, 2, 0),
            'platform_name': 'SGB',
            'quality_group': {
                'duration_of_product': np.array(1000., dtype=np.float32),
                'overall_quality_flag': 0,
            }
        }

    def test_get_quality_attributes(self, reader):
        """Test get quality attributes."""
        attributes = reader._get_quality_attributes()
        assert attributes == {
            'duration_of_product': np.array(1000., dtype=np.float32),
            'overall_quality_flag': 0,
        }

    @patch(
        'satpy.readers.ici_l1b_nc.IciL1bNCFileHandler._get_global_attributes',
        return_value={"mocked_global_attributes": True},
    )
    def test_manage_attributes(self, mock, reader):
        """Test manage attributes."""
        variable = xr.DataArray(
            np.ones(N_SCAN),
            attrs={"season": "summer"},
        )
        dataset_info = {'name': 'ici_1', 'units': 'K'}
        variable = reader._manage_attributes(variable, dataset_info)
        assert variable.attrs == {
            'season': 'summer',
            'units': 'K',
            'name': 'ici_1',
            'mocked_global_attributes': True,
        }
