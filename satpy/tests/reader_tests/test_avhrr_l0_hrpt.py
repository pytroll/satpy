#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009-2021 Satpy developers
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
"""Tests for the hrpt reader."""

import os
import unittest
from contextlib import suppress
from tempfile import NamedTemporaryFile
from unittest import mock

import numpy as np
import xarray as xr

from satpy.readers.hrpt import HRPTFile, dtype
from satpy.tests.reader_tests.test_avhrr_l1b_gaclac import PygacPatcher
from satpy.tests.utils import make_dataid

NUMBER_OF_SCANS = 10
SWATH_WIDTH = 2048


class TestHRPTWithFile(unittest.TestCase):
    """Test base class with writing a fake file."""

    def setUp(self) -> None:
        """Set up the test case."""
        test_data = np.ones(NUMBER_OF_SCANS, dtype=dtype)
        # Channel 3a
        test_data["id"]["id"][:5] = 891
        # Channel 3b
        test_data["id"]["id"][5:] = 890
        with NamedTemporaryFile(mode='w+', suffix='.hmf', delete=False) as hrpt_file:
            self.filename = hrpt_file.name
            test_data.tofile(hrpt_file)

    def tearDown(self) -> None:
        """Tear down the test case."""
        with suppress(OSError):
            os.remove(self.filename)

    def _get_dataset(self, dataset_id):
        fh = HRPTFile(self.filename, {}, {})
        return fh.get_dataset(dataset_id, {})


class TestHRPTReading(TestHRPTWithFile):
    """Test case for reading hrpt data."""

    def test_reading(self):
        """Test that data is read."""
        fh = HRPTFile(self.filename, {}, {})
        assert fh._data is not None


class TestHRPTGetUncalibratedData(TestHRPTWithFile):
    """Test case for reading uncalibrated hrpt data."""

    def _get_channel_1_counts(self):
        return self._get_dataset(make_dataid(name='1', calibration='counts'))

    def test_get_dataset_returns_a_dataarray(self):
        """Test that get_dataset returns a dataarray."""
        result = self._get_channel_1_counts()
        assert isinstance(result, xr.DataArray)

    def test_platform_name(self):
        """Test that the platform name is correct."""
        result = self._get_channel_1_counts()
        assert result.attrs['platform_name'] == 'NOAA 19'

    def test_no_calibration_values_are_1(self):
        """Test that the values of non-calibrated data is 1."""
        result = self._get_channel_1_counts()
        assert (result.values == 1).all()


def fake_calibrate_solar(data, *args, **kwargs):
    """Fake calibration."""
    del args, kwargs
    return data * 25.43 + 3


def fake_calibrate_thermal(data, *args, **kwargs):
    """Fake calibration."""
    del args, kwargs
    return data * 35.43 + 3


class CalibratorPatcher(PygacPatcher):
    """Patch pygac."""

    def setUp(self) -> None:
        """Patch pygac's calibration."""
        super().setUp()

        # Import things to patch here to make them patchable. Otherwise another function
        # might import it first which would prevent a successful patch.
        from pygac.calibration import Calibrator, calibrate_solar, calibrate_thermal
        self.Calibrator = Calibrator
        self.calibrate_thermal = calibrate_thermal
        self.calibrate_thermal.side_effect = fake_calibrate_thermal
        self.calibrate_solar = calibrate_solar
        self.calibrate_solar.side_effect = fake_calibrate_solar


class TestHRPTWithPatchedCalibratorAndFile(CalibratorPatcher, TestHRPTWithFile):
    """Test case with patched calibration routines and a synthetic file."""

    def setUp(self) -> None:
        """Set up the test case."""
        CalibratorPatcher.setUp(self)
        TestHRPTWithFile.setUp(self)

    def tearDown(self):
        """Tear down the test case."""
        CalibratorPatcher.tearDown(self)
        TestHRPTWithFile.tearDown(self)


class TestHRPTGetCalibratedReflectances(TestHRPTWithPatchedCalibratorAndFile):
    """Test case for reading calibrated reflectances from hrpt data."""

    def _get_channel_1_reflectance(self):
        """Get the channel 1 reflectance."""
        dataset_id = make_dataid(name='1', calibration='reflectance')
        return self._get_dataset(dataset_id)

    def test_calibrated_reflectances_values(self):
        """Test the calibrated reflectance values."""
        result = self._get_channel_1_reflectance()
        np.testing.assert_allclose(result.values, 28.43)


class TestHRPTGetCalibratedBT(TestHRPTWithPatchedCalibratorAndFile):
    """Test case for reading calibrated brightness temperature from hrpt data."""

    def _get_channel_4_bt(self):
        """Get the channel 4 bt."""
        dataset_id = make_dataid(name='4', calibration='brightness_temperature')
        return self._get_dataset(dataset_id)

    def test_calibrated_bt_values(self):
        """Test the calibrated reflectance values."""
        result = self._get_channel_4_bt()
        np.testing.assert_allclose(result.values, 38.43)


class TestHRPTChannel3(TestHRPTWithPatchedCalibratorAndFile):
    """Test case for reading calibrated brightness temperature from hrpt data."""

    def _get_channel_3b_bt(self):
        """Get the channel 4 bt."""
        dataset_id = make_dataid(name='3b', calibration='brightness_temperature')
        return self._get_dataset(dataset_id)

    def _get_channel_3a_reflectance(self):
        """Get the channel 4 bt."""
        dataset_id = make_dataid(name='3a', calibration='reflectance')
        return self._get_dataset(dataset_id)

    def _get_channel_3a_counts(self):
        """Get the channel 4 bt."""
        dataset_id = make_dataid(name='3a', calibration='counts')
        return self._get_dataset(dataset_id)

    def test_channel_3b_masking(self):
        """Test that channel 3b is split correctly."""
        result = self._get_channel_3b_bt()
        assert np.isnan(result.values[:5]).all()
        assert np.isfinite(result.values[5:]).all()

    def test_channel_3a_masking(self):
        """Test that channel 3a is split correctly."""
        result = self._get_channel_3a_reflectance()
        assert np.isnan(result.values[5:]).all()
        assert np.isfinite(result.values[:5]).all()

    def test_uncalibrated_channel_3a_masking(self):
        """Test that channel 3a is split correctly."""
        result = self._get_channel_3a_counts()
        assert np.isnan(result.values[5:]).all()
        assert np.isfinite(result.values[:5]).all()


class TestHRPTNavigation(TestHRPTWithFile):
    """Test case for computing HRPT navigation."""

    def setUp(self) -> None:
        """Set up the test case."""
        super().setUp()
        self.fake_lons = np.ones((NUMBER_OF_SCANS, SWATH_WIDTH))
        self.fake_lats = np.ones((NUMBER_OF_SCANS, SWATH_WIDTH)) * 2

    def _prepare_mocks(self, Orbital, SatelliteInterpolator, get_lonlatalt):
        """Prepare the mocks."""
        Orbital.return_value.get_position.return_value = mock.MagicMock(), mock.MagicMock()
        get_lonlatalt.return_value = (mock.MagicMock(), mock.MagicMock(), mock.MagicMock())
        SatelliteInterpolator.return_value.interpolate.return_value = self.fake_lons, self.fake_lats

    @mock.patch.multiple('satpy.readers.hrpt',
                         Orbital=mock.DEFAULT,
                         compute_pixels=mock.DEFAULT,
                         get_lonlatalt=mock.DEFAULT,
                         SatelliteInterpolator=mock.DEFAULT)
    def test_longitudes_are_returned(self, Orbital, compute_pixels, get_lonlatalt, SatelliteInterpolator):
        """Check that latitudes are returned properly."""
        self._prepare_mocks(Orbital, SatelliteInterpolator, get_lonlatalt)
        dataset_id = make_dataid(name='longitude')
        result = self._get_dataset(dataset_id)
        assert (result == self.fake_lons).all()

    @mock.patch.multiple('satpy.readers.hrpt',
                         Orbital=mock.DEFAULT,
                         compute_pixels=mock.DEFAULT,
                         get_lonlatalt=mock.DEFAULT,
                         SatelliteInterpolator=mock.DEFAULT)
    def test_latitudes_are_returned(self, Orbital, compute_pixels, get_lonlatalt, SatelliteInterpolator):
        """Check that latitudes are returned properly."""
        self._prepare_mocks(Orbital, SatelliteInterpolator, get_lonlatalt)
        dataset_id = make_dataid(name='latitude')
        result = self._get_dataset(dataset_id)
        assert (result == self.fake_lats).all()
