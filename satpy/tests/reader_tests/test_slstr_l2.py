#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 Satpy developers
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
"""Module for testing the satpy.readers.slstr_l2 module."""

import unittest
from unittest import mock
from unittest.mock import MagicMock
from unittest.mock import patch
import xarray as xr
from satpy.readers.slstr_l2 import SLSTRL2FileHandler


class TestSLSTRL2Reader(unittest.TestCase):
    """Test Sentinel-3 SST L2 reader."""

    @mock.patch('xarray.open_dataset')
    def test_instantiate(self, mocked_dataset):
        """Test initialization of file handlers."""
        filename_info = {}
        tmp = MagicMock(start_time='20191120T125002Z', stop_time='20191120T125002Z')
        tmp.rename.return_value = tmp
        xr.open_dataset.return_value = tmp
        SLSTRL2FileHandler('somedir/somefile.nc', filename_info, None)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

        with patch('tarfile.open') as tf:
            tf.return_value.__enter__.return_value = MagicMock(getnames=lambda *a: ["GHRSST-SSTskin.nc"])
            SLSTRL2FileHandler('somedir/somefile.tar', filename_info, None)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

    @mock.patch('xarray.open_dataset')
    def test_get_dataset(self, mocked_dataset):
        """Test retrieval of datasets."""
        filename_info = {}
        tmp = MagicMock(start_time='20191120T125002Z', stop_time='20191120T125002Z')
        tmp.rename.return_value = tmp
        xr.open_dataset.return_value = tmp
        test = SLSTRL2FileHandler('somedir/somefile.nc', filename_info, None)
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
