#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018, 2022 Satpy developers
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
"""Module for testing the satpy.readers.ghrsst_l2 module."""

import unittest
from datetime import datetime
from unittest import mock
from unittest.mock import MagicMock, patch

import numpy as np
import xarray as xr

from satpy.readers.ghrsst_l2 import GHRSSTL2FileHandler


class TestGHRSSTL2Reader(unittest.TestCase):
    """Test Sentinel-3 SST L2 reader."""

    @mock.patch('satpy.readers.ghrsst_l2.xr')
    def setUp(self, xr_):
        """Create a fake osisaf ghrsst dataset."""
        self.base_data = np.array(([-32768, 1135, 1125], [1138, 1128, 1080]))
        self.sst = xr.DataArray(
            self.base_data,
            dims=('nj', 'ni'),
            attrs={'scale_factor': 0.01, 'add_offset': 273.15,
                   '_FillValue': -32768, 'units': 'kelvin',
                   }
        )
        self.fake_dataset = xr.Dataset(
            data_vars={
                'sea_surface_temperature': self.sst,
            },
            attrs={
                "start_time": "20220321T112640Z",
                "stop_time": "20220321T145711Z",
                "platform": 'NOAA20',
                "sensor": "VIIRS",
            },
        )

    @mock.patch('xarray.open_dataset')
    def test_instantiate(self, mocked_dataset):
        """Test initialization of file handlers."""
        filename_info = {}
        tmp = MagicMock(start_time='20191120T125002Z', stop_time='20191120T125002Z')
        tmp.rename.return_value = tmp
        xr.open_dataset.return_value = tmp
        GHRSSTL2FileHandler('somedir/somefile.nc', filename_info, None)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

        with patch('tarfile.open') as tf:
            tf.return_value.__enter__.return_value = MagicMock(getnames=lambda *a: ["GHRSST-SSTskin.nc"])
            GHRSSTL2FileHandler('somedir/somefile.tar', filename_info, None)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

    @mock.patch('xarray.open_dataset')
    def test_get_dataset(self, mocked_dataset):
        """Test retrieval of datasets."""
        filename_info = {}
        tmp = MagicMock(start_time='20191120T125002Z', stop_time='20191120T125002Z')
        tmp.rename.return_value = tmp
        xr.open_dataset.return_value = tmp
        test = GHRSSTL2FileHandler('somedir/somefile.nc', filename_info, None)
        test.nc = {'longitude': xr.Dataset(),
                   'latitude': xr.Dataset(),
                   'sea_surface_temperature': xr.Dataset(),
                   'sea_ice_fraction': xr.Dataset(),
                   }
        test.get_dataset('longitude', {'standard_name': 'longitude'})
        test.get_dataset('latitude', {'standard_name': 'latitude'})
        test.get_dataset('sea_surface_temperature', {'standard_name': 'sea_surface_temperature'})
        test.get_dataset('sea_ice_fraction', {'standard_name': 'sea_ice_fraction'})
        with self.assertRaises(KeyError):
            test.get_dataset('erroneous dataset', {'standard_name': 'erroneous dataset'})
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

    @mock.patch('xarray.open_dataset')
    def test_get_sensor(self, mocked_dataset):
        """Test retrieval of the sensor name from the netCDF file."""
        mocked_dataset.return_value = self.fake_dataset
        dt_valid = datetime(2022, 3, 21, 11, 26, 40)  # 202203211200Z
        filename_info = {'field_type': 'NARSST', 'generating_centre': 'FRA_',
                         'satid': 'NOAA20_', 'valid_time': dt_valid}

        test = GHRSSTL2FileHandler('somedir/somefile.nc', filename_info, None)
        assert test.sensor == 'viirs'

    @mock.patch('xarray.open_dataset')
    def test_get_start_and_end_times(self, mocked_dataset):
        """Test retrieval of the sensor name from the netCDF file."""
        mocked_dataset.return_value = self.fake_dataset
        dt_valid = datetime(2022, 3, 21, 11, 26, 40)  # 202203211200Z
        good_start_time = datetime(2022, 3, 21, 11, 26, 40)  # 20220321T112640Z
        good_stop_time = datetime(2022, 3, 21, 14, 57, 11)  # 20220321T145711Z

        filename_info = {'field_type': 'NARSST', 'generating_centre': 'FRA_',
                         'satid': 'NOAA20_', 'valid_time': dt_valid}

        test = GHRSSTL2FileHandler('somedir/somefile.nc', filename_info, None)

        assert test.start_time == good_start_time
        assert test.end_time == good_stop_time
