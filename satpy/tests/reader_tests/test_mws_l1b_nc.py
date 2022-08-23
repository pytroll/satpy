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

from datetime import datetime

import numpy as np
import pytest
from netCDF4 import Dataset

from satpy.readers.mws_l1b import MWSL1BFile

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
            'longitude': 'data/navigation_data/mws_lon',
            'latitude': 'data/navigation_data/mws_lat',
            'solar_azimuth': 'data/navigation/mws_solar_azimuth_angle',
            'solar_zenith': 'data/navigation/mws_solar_zenith_angle',
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
            data_group = dataset.createGroup('data')
            self._write_navigation_data_group(data_group)

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
    def _write_navigation_data_group(dataset):
        """Write the navigation data group."""
        group = dataset.createGroup('navigation')
        group.createDimension('n_schannels', N_CHANNELS)
        group.createDimension('n_schannels_os', N_CHANNELS_OS)
        group.createDimension('n_scans', N_SCANS)
        group.createDimension('n_fovs', N_FOVS)
        group.createDimension('n_prts', N_PRTS)
        group.createDimension('n_fovs_cal', N_FOVS_CAL)

        dimensions = ('n_scans', 'n_fovs')
        shape = (N_SCANS, N_FOVS)
        longitude = group.createVariable(
            'mws_lon',
            np.float32,
            dimensions=dimensions,
        )
        longitude[:] = np.ones(shape)
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
        azimuth[:] = 3. * np.ones(shape)


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
