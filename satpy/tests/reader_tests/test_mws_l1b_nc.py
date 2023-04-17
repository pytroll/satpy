# Copyright (c) 2022 Pytroll Developers

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""The mws_l1b_nc reader tests.

This module tests the reading of the MWS l1b netCDF format data as per version v4B issued 22 November 2021.

"""

import logging
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from netCDF4 import Dataset

from satpy.readers.mws_l1b import MWSL1BFile, get_channel_index_from_name

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - tmp_path
# - caplog

N_CHANNELS = 24
N_CHANNELS_OS = 2
N_SCANS = 2637
N_FOVS = 95
N_FOVS_CAL = 5
N_PRTS = 6


@pytest.fixture
def reader(fake_file):
    """Return reader of mws level-1b data."""
    return MWSL1BFile(
        filename=fake_file,
        filename_info={
            'start_time': (
                datetime.fromisoformat('2000-01-01T01:00:00')
            ),
            'end_time': (
                datetime.fromisoformat('2000-01-01T02:00:00')
            ),
            'creation_time': (
                datetime.fromisoformat('2000-01-01T03:00:00')
            ),
        },
        filetype_info={
            'longitude': 'data/navigation_data/mws_lon',
            'latitude': 'data/navigation_data/mws_lat',
            'solar_azimuth': 'data/navigation/mws_solar_azimuth_angle',
            'solar_zenith': 'data/navigation/mws_solar_zenith_angle',
            'satellite_azimuth': 'data/navigation/mws_satellite_azimuth_angle',
            'satellite_zenith': 'data/navigation/mws_satellite_zenith_angle',
        }
    )


@pytest.fixture
def fake_file(tmp_path):
    """Return file path to level-1b file."""
    file_path = tmp_path / 'test_file_mws_l1b.nc'
    writer = MWSL1BFakeFileWriter(file_path)
    writer.write()
    yield file_path


class MWSL1BFakeFileWriter:
    """Writer class of fake mws level-1b data."""

    def __init__(self, file_path):
        """Init."""
        self.file_path = file_path

    def write(self):
        """Write fake data to file."""
        with Dataset(self.file_path, 'w') as dataset:
            self._write_attributes(dataset)
            self._write_status_group(dataset)
            self._write_quality_group(dataset)
            data_group = dataset.createGroup('data')
            self._create_scan_dimensions(data_group)
            self._write_navigation_data_group(data_group)
            self._write_calibration_data_group(data_group)
            self._write_measurement_data_group(data_group)

    @staticmethod
    def _write_attributes(dataset):
        """Write attributes."""
        dataset.sensing_start_time_utc = "2000-01-02 03:04:05.000"
        dataset.sensing_end_time_utc = "2000-01-02 04:05:06.000"
        dataset.instrument = "MWS"
        dataset.spacecraft = "SGA1"

    @staticmethod
    def _write_status_group(dataset):
        """Write the status group."""
        group = dataset.createGroup('/status/satellite')
        subsat_latitude_start = group.createVariable(
            'subsat_latitude_start', "f4"
        )
        subsat_latitude_start[:] = 52.19

        subsat_longitude_start = group.createVariable(
            'subsat_longitude_start', "f4"
        )
        subsat_longitude_start[:] = 23.26

        subsat_latitude_end = group.createVariable(
            'subsat_latitude_end', "f4"
        )
        subsat_latitude_end[:] = 60.00

        subsat_longitude_end = group.createVariable(
            'subsat_longitude_end', "f4"
        )
        subsat_longitude_end[:] = 2.47

    @staticmethod
    def _write_quality_group(dataset):
        """Write the quality group."""
        group = dataset.createGroup('quality')
        group.overall_quality_flag = 0
        duration_of_product = group.createVariable(
            'duration_of_product', "f4"
        )
        duration_of_product[:] = 5944.

    @staticmethod
    def _write_navigation_data_group(dataset):
        """Write the navigation data group."""
        group = dataset.createGroup('navigation')
        dimensions = ('n_scans', 'n_fovs')
        shape = (N_SCANS, N_FOVS)
        longitude = group.createVariable(
            'mws_lon',
            np.int32,
            dimensions=dimensions,
        )
        longitude.scale_factor = 1.0E-4
        longitude.add_offset = 0.0
        longitude.missing_value = np.array((-2147483648), np.int32)
        longitude[:] = 35.7535 * np.ones(shape)

        latitude = group.createVariable(
            'mws_lat',
            np.float32,
            dimensions=dimensions,
        )
        latitude[:] = 2. * np.ones(shape)

        azimuth = group.createVariable(
            'mws_solar_azimuth_angle',
            np.float32,
            dimensions=dimensions,
        )
        azimuth[:] = 179. * np.ones(shape)

    @staticmethod
    def _create_scan_dimensions(dataset):
        """Create the scan/fovs dimensions."""
        dataset.createDimension('n_channels', N_CHANNELS)
        dataset.createDimension('n_channels_os', N_CHANNELS_OS)
        dataset.createDimension('n_scans', N_SCANS)
        dataset.createDimension('n_fovs', N_FOVS)
        dataset.createDimension('n_prts', N_PRTS)
        dataset.createDimension('n_fovs_cal', N_FOVS_CAL)

    @staticmethod
    def _write_calibration_data_group(dataset):
        """Write the calibration data group."""
        group = dataset.createGroup('calibration')
        toa_bt = group.createVariable(
            'mws_toa_brightness_temperature', np.float32, dimensions=('n_scans', 'n_fovs', 'n_channels',)
        )
        toa_bt.scale_factor = 1.0  # 1.0E-8
        toa_bt.add_offset = 0.0
        toa_bt.missing_value = -2147483648
        toa_bt[:] = 240.0 * np.ones((N_SCANS, N_FOVS, N_CHANNELS))

    @staticmethod
    def _write_measurement_data_group(dataset):
        """Write the measurement data group."""
        group = dataset.createGroup('measurement')
        counts = group.createVariable(
            'mws_earth_view_counts', np.int32, dimensions=('n_scans', 'n_fovs', 'n_channels',)
        )
        counts[:] = 24100 * np.ones((N_SCANS, N_FOVS, N_CHANNELS), dtype=np.int32)


class TestMwsL1bNCFileHandler:
    """Test the MWSL1BFile reader."""

    def test_start_time(self, reader):
        """Test acquiring the start time."""
        assert reader.start_time == datetime(2000, 1, 2, 3, 4, 5)

    def test_end_time(self, reader):
        """Test acquiring the end time."""
        assert reader.end_time == datetime(2000, 1, 2, 4, 5, 6)

    def test_sensor(self, reader):
        """Test sensor."""
        assert reader.sensor == "MWS"

    def test_platform_name(self, reader):
        """Test getting the platform name."""
        assert reader.platform_name == "Metop-SG-A1"

    def test_sub_satellite_longitude_start(self, reader):
        """Test getting the longitude of sub-satellite point at start of the product."""
        np.testing.assert_allclose(reader.sub_satellite_longitude_start, 23.26)

    def test_sub_satellite_latitude_start(self, reader):
        """Test getting the latitude of sub-satellite point at start of the product."""
        np.testing.assert_allclose(reader.sub_satellite_latitude_start, 52.19)

    def test_sub_satellite_longitude_end(self, reader):
        """Test getting the longitude of sub-satellite point at end of the product."""
        np.testing.assert_allclose(reader.sub_satellite_longitude_end, 2.47)

    def test_sub_satellite_latitude_end(self, reader):
        """Test getting the latitude of sub-satellite point at end of the product."""
        np.testing.assert_allclose(reader.sub_satellite_latitude_end, 60.0)

    def test_get_dataset_get_channeldata_counts(self, reader):
        """Test getting channel data."""
        dataset_id = {'name': '1', 'units': None,
                      'calibration': 'counts'}
        dataset_info = {'file_key': 'data/measurement/mws_earth_view_counts'}

        dataset = reader.get_dataset(dataset_id, dataset_info)
        expected_bt = np.array([[24100, 24100],
                                [24100, 24100]], dtype=np.int32)
        count = dataset[10:12, 12:14].data.compute()
        np.testing.assert_allclose(count, expected_bt)

    def test_get_dataset_get_channeldata_bts(self, reader):
        """Test getting channel data."""
        dataset_id = {'name': '1', 'units': 'K',
                      'calibration': 'brightness_temperature'}
        dataset_info = {'file_key': 'data/calibration/mws_toa_brightness_temperature'}

        dataset = reader.get_dataset(dataset_id, dataset_info)

        expected_bt = np.array([[240., 240., 240., 240., 240.],
                                [240., 240., 240., 240., 240.],
                                [240., 240., 240., 240., 240.],
                                [240., 240., 240., 240., 240.],
                                [240., 240., 240., 240., 240.]], dtype=np.float32)

        toa_bt = dataset[0:5, 0:5].data.compute()
        np.testing.assert_allclose(toa_bt, expected_bt)

    def test_get_dataset_return_none_if_data_not_exist(self, reader):
        """Test get dataset return none if data does not exist."""
        dataset_id = {'name': 'unknown'}
        dataset_info = {'file_key': 'non/existing/data'}
        dataset = reader.get_dataset(dataset_id, dataset_info)
        assert dataset is None

    def test_get_navigation_longitudes(self, caplog, fake_file, reader):
        """Test get the longitudes."""
        dataset_id = {'name': 'mws_lon'}
        dataset_info = {'file_key': 'data/navigation_data/mws_lon'}

        dataset = reader.get_dataset(dataset_id, dataset_info)

        expected_lons = np.array([[35.753498, 35.753498, 35.753498, 35.753498, 35.753498],
                                  [35.753498, 35.753498, 35.753498, 35.753498, 35.753498],
                                  [35.753498, 35.753498, 35.753498, 35.753498, 35.753498],
                                  [35.753498, 35.753498, 35.753498, 35.753498, 35.753498],
                                  [35.753498, 35.753498, 35.753498, 35.753498, 35.753498]], dtype=np.float32)

        longitudes = dataset[0:5, 0:5].data.compute()
        np.testing.assert_allclose(longitudes, expected_lons)

    def test_get_dataset_logs_debug_message(self, caplog, fake_file, reader):
        """Test get dataset return none if data does not exist."""
        dataset_id = {'name': 'mws_lon'}
        dataset_info = {'file_key': 'data/navigation_data/mws_lon'}

        with caplog.at_level(logging.DEBUG):
            _ = reader.get_dataset(dataset_id, dataset_info)

        log_output = "Reading mws_lon from {filename}".format(filename=str(fake_file))
        assert log_output in caplog.text

    def test_get_dataset_aux_data_not_supported(self, reader):
        """Test get auxillary dataset not supported."""
        dataset_id = {'name': 'scantime_utc'}
        dataset_info = {'file_key': 'non/existing'}

        with pytest.raises(NotImplementedError) as exec_info:
            _ = reader.get_dataset(dataset_id, dataset_info)

        assert str(exec_info.value) == "Dataset 'scantime_utc' not supported!"

    def test_get_dataset_aux_data_expected_data_missing(self, caplog, reader):
        """Test get auxillary dataset which is not present but supposed to be in file."""
        dataset_id = {'name': 'surface_type'}
        dataset_info = {'file_key': 'non/existing'}

        with caplog.at_level(logging.ERROR):
            with pytest.raises(KeyError) as exec_info:
                _ = reader.get_dataset(dataset_id, dataset_info)

        assert str(exec_info.value) == "'data/navigation/mws_surface_type'"

        log_output = ("Could not find key data/navigation/mws_surface_type in NetCDF file," +
                      " no valid Dataset created")
        assert log_output in caplog.text

    @pytest.mark.parametrize('dims', (
        ('n_scans', 'n_fovs'),
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

    @staticmethod
    def test_drop_coords(reader):
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

    def test_get_global_attributes(self, reader):
        """Test get global attributes."""
        attributes = reader._get_global_attributes()
        assert attributes == {
            'filename': reader.filename,
            'start_time': datetime(2000, 1, 2, 3, 4, 5),
            'end_time': datetime(2000, 1, 2, 4, 5, 6),
            'spacecraft_name': 'Metop-SG-A1',
            'sensor': 'MWS',
            'filename_start_time': datetime(2000, 1, 1, 1, 0),
            'filename_end_time': datetime(2000, 1, 1, 2, 0),
            'platform_name': 'Metop-SG-A1',
            'quality_group': {
                'duration_of_product': np.array(5944., dtype=np.float32),
                'overall_quality_flag': 0,
            }
        }

    @patch(
        'satpy.readers.mws_l1b.MWSL1BFile._get_global_attributes',
        return_value={"mocked_global_attributes": True},
    )
    def test_manage_attributes(self, mock, reader):
        """Test manage attributes."""
        variable = xr.DataArray(
            np.ones(N_SCANS),
            attrs={"season": "summer"},
        )
        dataset_info = {'name': '1', 'units': 'K'}
        variable = reader._manage_attributes(variable, dataset_info)
        assert variable.attrs == {
            'season': 'summer',
            'units': 'K',
            'name': '1',
            'mocked_global_attributes': True,
        }


@pytest.mark.parametrize("name, index", [('1', 0), ('2', 1), ('24', 23)])
def test_get_channel_index_from_name(name, index):
    """Test getting the MWS channel index from the channel name."""
    ch_idx = get_channel_index_from_name(name)
    assert ch_idx == index


def test_get_channel_index_from_name_throw_exception():
    """Test that an excpetion is thrown when getting the MWS channel index from an unsupported name."""
    with pytest.raises(Exception) as excinfo:
        _ = get_channel_index_from_name('channel 1')

    assert str(excinfo.value) == "Channel name 'channel 1' not supported"
    assert excinfo.type == AttributeError
