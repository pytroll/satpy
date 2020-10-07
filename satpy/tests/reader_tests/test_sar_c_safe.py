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
"""Module for testing the satpy.readers.sar-c_safe module."""
import unittest
import unittest.mock as mock

from satpy.dataset import DataQuery
import dask.array as da
import xarray as xr
import numpy as np


class TestSAFEGRD(unittest.TestCase):
    """Test the SAFE GRD file handler."""

    @mock.patch('rasterio.open')
    def setUp(self, mocked_rio_open):
        """Set up the test case."""
        from satpy.readers.sar_c_safe import SAFEGRD
        filename_info = {'mission_id': 'S1A', 'dataset_name': 'foo', 'start_time': 0, 'end_time': 0,
                         'polarization': 'vv'}
        filetype_info = 'bla'
        self.noisefh = mock.MagicMock()
        self.noisefh.get_noise_correction.return_value = xr.DataArray(np.zeros((2, 2)))
        self.calfh = mock.MagicMock()
        self.calfh.get_calibration_constant.return_value = 1
        self.calfh.get_calibration.return_value = xr.DataArray(np.ones((2, 2)))

        self.test_fh = SAFEGRD('S1A_IW_GRDH_1SDV_20190201T024655_20190201T024720_025730_02DC2A_AE07.SAFE/measurement/s1a-iw-grd'  # noqa
                               '-vv-20190201t024655-20190201t024720-025730-02dc2a-001.tiff',
                               filename_info, filetype_info, self.calfh, self.noisefh)
        self.mocked_rio_open = mocked_rio_open

    def test_instantiate(self):
        """Test initialization of file handlers."""
        assert(self.test_fh._polarization == 'vv')
        assert(self.test_fh.calibration == self.calfh)
        assert(self.test_fh.noise == self.noisefh)
        self.mocked_rio_open.assert_called()

    def test_read_calibrated_natural(self):
        """Test the calibration routines."""
        calibration = mock.MagicMock()
        calibration.name = "sigma_nought"
        with mock.patch.object(self.test_fh, 'read_band') as fake_read_band:
            fake_read_band.return_value = xr.DataArray(da.from_array(np.array([[0, 1], [2, 3]])))
            xarr = self.test_fh.get_dataset(DataQuery(name="measurement", polarization="vv",
                                                      calibration=calibration, quantity='natural'), info=dict())
            np.testing.assert_allclose(xarr, [[np.nan, 2], [5, 10]])

    def test_read_calibrated_dB(self):
        """Test the calibration routines."""
        calibration = mock.MagicMock()
        calibration.name = "sigma_nought"
        with mock.patch.object(self.test_fh, 'read_band') as fake_read_band:
            fake_read_band.return_value = xr.DataArray(da.from_array(np.array([[0, 1], [2, 3]])))
            xarr = self.test_fh.get_dataset(DataQuery(name="measurement", polarization="vv",
                                                      calibration=calibration, quantity='dB'), info=dict())
            np.testing.assert_allclose(xarr, [[np.nan, 3.0103], [6.9897, 10]])
