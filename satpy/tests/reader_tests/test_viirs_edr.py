#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2022 Satpy developers
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
"""Module for testing the satpy.readers.viirs_l2_jrr module.

Note: This is adapted from the test_slstr_l2.py code.
"""

from datetime import datetime
from unittest import mock
from unittest.mock import MagicMock

import pytest
import xarray as xr

from satpy.readers.viirs_edr import VIIRSJRRFileHandler


class TestVIIRSJRRReader:
    """Test the VIIRS JRR L2 reader."""

    @mock.patch('xarray.open_dataset')
    def test_instantiate(self, mocked_dataset):
        """Test initialization of file handlers."""
        filename_info = {'platform_shortname': 'npp'}
        tmp = MagicMock(start_time='20191120T125002Z', stop_time='20191120T125002Z')
        tmp.rename.return_value = tmp
        xr.open_dataset.return_value = tmp
        VIIRSJRRFileHandler('somedir/somefile.nc', filename_info, None)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

    @mock.patch('xarray.open_dataset')
    def test_get_dataset(self, mocked_dataset):
        """Test retrieval of datasets."""
        filename_info = {'platform_shortname': 'npp'}
        tmp = MagicMock(start_time='20191120T125002Z', stop_time='20191120T125002Z')
        xr.open_dataset.return_value = tmp
        test = VIIRSJRRFileHandler('somedir/somefile.nc', filename_info, None)
        test.nc = {'Longitude': xr.Dataset(),
                   'Latitude': xr.Dataset(),
                   'smoke_concentration': xr.Dataset(),
                   'fire_mask': xr.Dataset(),
                   'surf_refl_I01': xr.Dataset(),
                   'surf_refl_M05': xr.Dataset(),
                   }
        test.get_dataset('longitude', {'file_key': 'Longitude'})
        test.get_dataset('latitude', {'file_key': 'Latitude'})
        test.get_dataset('smoke_concentration', {'file_key': 'smoke_concentration'})
        test.get_dataset('fire_mask', {'file_key': 'fire_mask'})
        with pytest.raises(KeyError):
            test.get_dataset('erroneous dataset', {'file_key': 'erroneous dataset'})
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()
        test.get_dataset('surf_refl_I01', {'file_key': 'surf_refl_I01'})

    @mock.patch('xarray.open_dataset')
    def test_get_startend_times(self, mocked_dataset):
        """Test finding start and end times of granules."""
        filename_info = {'platform_shortname': 'npp',
                         'start_time': datetime(2021, 4, 3, 12, 0, 10),
                         'end_time': datetime(2021, 4, 3, 12, 4, 28)}
        tmp = MagicMock()
        tmp.rename.return_value = tmp
        xr.open_dataset.return_value = tmp
        hdl = VIIRSJRRFileHandler('somedir/somefile.nc', filename_info, None)
        assert hdl.start_time == datetime(2021, 4, 3, 12, 0, 10)
        assert hdl.end_time == datetime(2021, 4, 3, 12, 4, 28)

    @mock.patch('xarray.open_dataset')
    def test_get_platformname(self, mocked_dataset):
        """Test finding start and end times of granules."""
        tmp = MagicMock()
        tmp.rename.return_value = tmp
        xr.open_dataset.return_value = tmp
        hdl = VIIRSJRRFileHandler('somedir/somefile.nc', {'platform_shortname': 'npp'}, None)
        assert hdl.platform_name == 'Suomi-NPP'
        hdl = VIIRSJRRFileHandler('somedir/somefile.nc', {'platform_shortname': 'JPSS-1'}, None)
        assert hdl.platform_name == 'NOAA-20'
        hdl = VIIRSJRRFileHandler('somedir/somefile.nc', {'platform_shortname': 'J01'}, None)
        assert hdl.platform_name == 'NOAA-20'
