#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 Satpy developers
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
"""Module for testing the satpy.readers.safe_sar_l2_ocn module."""
import unittest
import unittest.mock as mock

import numpy as np
import xarray as xr

from satpy.tests.utils import make_dataid


class TestSAFENC(unittest.TestCase):
    """Test various SAFE SAR L2 OCN file handlers."""

    @mock.patch('satpy.readers.safe_sar_l2_ocn.xr')
    def setUp(self, xr_):
        """Set up the tests."""
        from satpy.readers.safe_sar_l2_ocn import SAFENC

        self.channels = ['owiWindSpeed', 'owiLon', 'owiLat', 'owiHs', 'owiNrcs', 'foo',
                         'owiPolarisationName', 'owiCalConstObsi']
        # Mock file access to return a fake dataset.
        self.dummy3d = np.zeros((2, 2, 1))
        self.dummy2d = np.zeros((2, 2))
        self.dummy1d = np.zeros((2))
        self.band = 1
        self.nc = xr.Dataset(
            {'owiWindSpeed': xr.DataArray(self.dummy2d, dims=('owiAzSize', 'owiRaSize'), attrs={'_FillValue': np.nan}),
             'owiLon': xr.DataArray(data=self.dummy2d, dims=('owiAzSize', 'owiRaSize')),
             'owiLat': xr.DataArray(data=self.dummy2d, dims=('owiAzSize', 'owiRaSize')),
             'owiHs': xr.DataArray(data=self.dummy3d, dims=('owiAzSize', 'owiRaSize', 'oswPartition')),
             'owiNrcs': xr.DataArray(data=self.dummy3d, dims=('owiAzSize', 'owiRaSize', 'oswPolarization')),
             'foo': xr.DataArray(self.dummy2d, dims=('owiAzSize', 'owiRaSize')),
             'owiPolarisationName': xr.DataArray(self.dummy1d, dims=('owiPolarisation')),
             'owiCalConstObsi': xr.DataArray(self.dummy1d, dims=('owiIncSize'))
             },
            attrs={'_FillValue': np.nan,
                   'missionName': 'S1A'})
        xr_.open_dataset.return_value = self.nc

        # Instantiate reader using the mocked open_dataset() method. Also, make
        # the reader believe all abstract methods have been implemented.
        self.reader = SAFENC(filename='dummy',
                             filename_info={'start_time': 0,
                                            'end_time': 0,
                                            'fstart_time': 0,
                                            'fend_time': 0,
                                            'polarization': 'vv'},
                             filetype_info={})

    def test_init(self):
        """Test reader initialization."""
        self.assertEqual(self.reader.start_time, 0)
        self.assertEqual(self.reader.end_time, 0)
        self.assertEqual(self.reader.fstart_time, 0)
        self.assertEqual(self.reader.fend_time, 0)

    def test_get_dataset(self):
        """Test getting a dataset."""
        for ch in self.channels:
            dt = self.reader.get_dataset(
                key=make_dataid(name=ch), info={})
            # ... this only compares the valid (unmasked) elements
            self.assertTrue(np.all(self.nc[ch] == dt.to_masked_array()),
                            msg='get_dataset() returns invalid data for '
                            'dataset {}'.format(ch))
