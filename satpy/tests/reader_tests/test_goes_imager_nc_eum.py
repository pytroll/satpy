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
"""Tests for the goes imager nc reader (EUMETSAT variant)."""

import unittest
from unittest import mock

import numpy as np
import xarray as xr

from satpy.readers.goes_imager_nc import is_vis_channel
from satpy.tests.utils import make_dataid


class GOESNCEUMFileHandlerRadianceTest(unittest.TestCase):
    """Tests for the radiances."""

    longMessage = True

    @mock.patch('satpy.readers.goes_imager_nc.xr')
    def setUp(self, xr_):
        """Set up the tests."""
        from satpy.readers.goes_imager_nc import CALIB_COEFS, GOESEUMNCFileHandler

        self.coefs = CALIB_COEFS['GOES-15']
        self.all_coefs = CALIB_COEFS
        self.channels = sorted(self.coefs.keys())
        self.ir_channels = sorted([ch for ch in self.channels
                                   if not is_vis_channel(ch)])
        self.vis_channels = sorted([ch for ch in self.channels
                                    if is_vis_channel(ch)])

        # Mock file access to return a fake dataset.
        nrows = ncols = 300
        self.radiance = np.ones((1, nrows, ncols))  # IR channels
        self.lon = np.zeros((nrows, ncols))  # Dummy
        self.lat = np.repeat(np.linspace(-150, 150, nrows), ncols).reshape(
            nrows, ncols)  # Includes invalid values to be masked

        xr_.open_dataset.return_value = xr.Dataset(
            {'data': xr.DataArray(data=self.radiance, dims=('time', 'yc', 'xc')),
             'time': xr.DataArray(data=np.array([0], dtype='datetime64[ms]'),
                                  dims=('time',)),
             'bands': xr.DataArray(data=np.array([1]))},
            attrs={'Satellite Sensor': 'G-15'})

        geo_data = xr.Dataset(
            {'lon': xr.DataArray(data=self.lon, dims=('yc', 'xc')),
             'lat': xr.DataArray(data=self.lat, dims=('yc', 'xc'))},
            attrs={'Satellite Sensor': 'G-15'})

        # Instantiate reader using the mocked open_dataset() method
        self.reader = GOESEUMNCFileHandler(filename='dummy', filename_info={},
                                           filetype_info={}, geo_data=geo_data)

    def test_get_dataset_radiance(self):
        """Test getting the radiances."""
        for ch in self.channels:
            if not is_vis_channel(ch):
                radiance = self.reader.get_dataset(
                    key=make_dataid(name=ch, calibration='radiance'), info={})
                # ... this only compares the valid (unmasked) elements
                self.assertTrue(np.all(self.radiance == radiance.to_masked_array()),
                                msg='get_dataset() returns invalid radiance for '
                                'channel {}'.format(ch))

    def test_calibrate(self):
        """Test whether the correct calibration methods are called."""
        for ch in self.channels:
            if not is_vis_channel(ch):
                calibs = {'brightness_temperature': '_calibrate_ir'}
                for calib, method in calibs.items():
                    with mock.patch.object(self.reader, method) as target_func:
                        self.reader.calibrate(data=self.reader.nc['data'],
                                              calibration=calib, channel=ch)
                        target_func.assert_called()

    def test_get_sector(self):
        """Test sector identification."""
        from satpy.readers.goes_imager_nc import (
            FULL_DISC,
            NORTH_HEMIS_EAST,
            NORTH_HEMIS_WEST,
            SOUTH_HEMIS_EAST,
            SOUTH_HEMIS_WEST,
            UNKNOWN_SECTOR,
        )
        shapes = {
            (2700, 5200): FULL_DISC,
            (1850, 3450): NORTH_HEMIS_EAST,
            (600, 3500): SOUTH_HEMIS_EAST,
            (1310, 3300): NORTH_HEMIS_WEST,
            (1099, 2800): SOUTH_HEMIS_WEST,
            (123, 456): UNKNOWN_SECTOR
        }
        for (nlines, ncols), sector_ref in shapes.items():
            for channel in ('00_7', '10_7'):
                sector = self.reader._get_sector(channel=channel, nlines=nlines,
                                                 ncols=ncols)
                self.assertEqual(sector, sector_ref,
                                 msg='Incorrect sector identification')


class GOESNCEUMFileHandlerReflectanceTest(unittest.TestCase):
    """Testing the reflectances."""

    longMessage = True

    @mock.patch('satpy.readers.goes_imager_nc.xr')
    def setUp(self, xr_):
        """Set up the tests."""
        from satpy.readers.goes_imager_nc import CALIB_COEFS, GOESEUMNCFileHandler

        self.coefs = CALIB_COEFS['GOES-15']
        self.all_coefs = CALIB_COEFS
        self.channels = sorted(self.coefs.keys())
        self.ir_channels = sorted([ch for ch in self.channels
                                   if not is_vis_channel(ch)])
        self.vis_channels = sorted([ch for ch in self.channels
                                    if is_vis_channel(ch)])

        # Mock file access to return a fake dataset.
        nrows = ncols = 300
        self.reflectance = 50 * np.ones((1, nrows, ncols))  # Vis channel
        self.lon = np.zeros((nrows, ncols))  # Dummy
        self.lat = np.repeat(np.linspace(-150, 150, nrows), ncols).reshape(
            nrows, ncols)  # Includes invalid values to be masked

        xr_.open_dataset.return_value = xr.Dataset(
            {'data': xr.DataArray(data=self.reflectance, dims=('time', 'yc', 'xc')),
             'time': xr.DataArray(data=np.array([0], dtype='datetime64[ms]'),
                                  dims=('time',)),
             'bands': xr.DataArray(data=np.array([1]))},
            attrs={'Satellite Sensor': 'G-15'})

        geo_data = xr.Dataset(
            {'lon': xr.DataArray(data=self.lon, dims=('yc', 'xc')),
             'lat': xr.DataArray(data=self.lat, dims=('yc', 'xc'))},
            attrs={'Satellite Sensor': 'G-15'})

        # Instantiate reader using the mocked open_dataset() method
        self.reader = GOESEUMNCFileHandler(filename='dummy', filename_info={},
                                           filetype_info={}, geo_data=geo_data)

    def test_get_dataset_reflectance(self):
        """Test getting the reflectance."""
        for ch in self.channels:
            if is_vis_channel(ch):
                refl = self.reader.get_dataset(
                    key=make_dataid(name=ch, calibration='reflectance'), info={})
                # ... this only compares the valid (unmasked) elements
                self.assertTrue(np.all(self.reflectance == refl.to_masked_array()),
                                msg='get_dataset() returns invalid reflectance for '
                                'channel {}'.format(ch))
