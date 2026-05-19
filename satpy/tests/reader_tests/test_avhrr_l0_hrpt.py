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

from datetime import datetime, timedelta
from unittest import mock

import numpy as np
import pytest
import xarray as xr

from satpy.readers import hrpt
from satpy.readers.hrpt import HRPTFile, scanline_dtype, time_seconds
from satpy.tests.utils import make_dataid

NUMBER_OF_SCANS = 10
SWATH_WIDTH = 2048

COUNTS = (np.arange(5 * NUMBER_OF_SCANS * SWATH_WIDTH) % 500 + 450).reshape((NUMBER_OF_SCANS, SWATH_WIDTH, 5))
LONS = np.ones((NUMBER_OF_SCANS, SWATH_WIDTH))
LATS = np.ones((NUMBER_OF_SCANS, SWATH_WIDTH)) * 2
START_TIME = datetime(2009, 6, 9, 9, 45)


@pytest.fixture
def hrpt_file(tmp_path):
        """Set up the test case."""
        test_data = np.ones(NUMBER_OF_SCANS, dtype=scanline_dtype)
        time_code = []
        offset = timedelta(seconds=1/6)
        for line in range(NUMBER_OF_SCANS):
            time_code.append(to_timecode(START_TIME + offset * line)[0])
        test_data["timecode"] = time_code
        test_data["image_data"] = COUNTS
        test_data["telemetry"]["PRT"] = 250
        test_data["telemetry"]["PRT"][::5] = 0
        # Channel 3a
        test_data["id"]["id"][:5] = 891
        test_data[:5]["space_data"] = [38, 39, 38, 991, 995]
        test_data[:5]["back_scan"] = [40, 445, 420]
        # Channel 3b
        test_data["id"]["id"][5:] = 890
        test_data[5:]["space_data"] = [38, 39, 980, 991, 995]
        test_data[5:]["back_scan"] = [440, 445, 420]
        filename = tmp_path / "20250609094500_noaa19.hmf"
        test_data.tofile(filename)
        return filename


@pytest.fixture
def hrpt_filename_info():
    """Create expected filename info without actually parsing the filename."""
    return {"start_time": START_TIME}


@pytest.fixture
def hrpt_fh(hrpt_file, hrpt_filename_info):
    """Open the file handler."""
    return HRPTFile(hrpt_file, hrpt_filename_info, {})


class TestHRPTReading:
    """Test case for reading hrpt data."""

    def test_reading(self, hrpt_file, hrpt_filename_info):
        """Test that data is read."""
        fh = HRPTFile(hrpt_file, hrpt_filename_info, {})
        assert fh._data is not None


class TestHRPTGetUncalibratedData:
    """Test case for reading uncalibrated hrpt data."""

    def test_get_dataset_returns_a_dataarray(self, hrpt_fh):
        """Test that get_dataset returns a data array."""
        result = hrpt_fh.get_dataset(make_dataid(name="1", calibration="counts"), {})
        assert isinstance(result, xr.DataArray)

    def test_platform_name(self, hrpt_fh):
        """Test that the platform name is correct."""
        result = hrpt_fh.get_dataset(make_dataid(name="1", calibration="counts"), {})
        assert result.attrs["platform_name"] == "NOAA 19"

    def test_no_calibration_values_are_raw(self, hrpt_fh):
        """Test that the values of uncalibrated data is the raw data."""
        result = hrpt_fh.get_dataset(make_dataid(name="1", calibration="counts"), {})
        assert (result.values == COUNTS[:, :, 0]).all()


class TestHRPTGetCalibratedReflectances:
    """Test case for reading calibrated reflectances from hrpt data."""

    def test_calibrated_reflectances_values(self, hrpt_fh):
        """Test the calibrated reflectance values."""
        result = hrpt_fh.get_dataset(make_dataid(name="1", calibration="reflectance"), {})
        np.testing.assert_allclose(result.values.mean(), 57.772733)


class TestHRPTGetCalibratedBT:
    """Test case for reading calibrated brightness temperature from hrpt data."""

    def test_calibrated_bt_values(self, hrpt_fh):
        """Test the calibrated reflectance values."""
        result = hrpt_fh.get_dataset(make_dataid(name="4", calibration="brightness_temperature"), {})
        np.testing.assert_allclose(result.values.mean(), 249.52884)


class TestHRPTChannel3:
    """Test case for reading calibrated brightness temperature from hrpt data."""

    def test_channel_3b_masking(self, hrpt_fh):
        """Test that channel 3b is split correctly."""
        result = hrpt_fh.get_dataset(make_dataid(name="3b", calibration="brightness_temperature"), {})
        assert np.isnan(result.values[:5]).all()
        assert np.isfinite(result.values[5:]).all()

    def test_channel_3a_masking(self, hrpt_fh):
        """Test that channel 3a is split correctly."""
        result = hrpt_fh.get_dataset(make_dataid(name="3a", calibration="reflectance"), {})
        assert np.isnan(result.values[5:]).all()
        assert np.isfinite(result.values[:5]).all()

    def test_uncalibrated_channel_3a_masking(self, hrpt_fh):
        """Test that channel 3a is split correctly."""
        result = hrpt_fh.get_dataset(make_dataid(name="3a", calibration="counts"), {})
        assert np.isnan(result.values[5:]).all()
        assert np.isfinite(result.values[:5]).all()


@pytest.fixture
def mock_nav(monkeypatch):
        """Prepare the mocks."""
        monkeypatch.setattr(hrpt, "compute_pixels", mock.Mock())
        Orbital = mock.Mock()
        Orbital.return_value.get_position.return_value = mock.MagicMock(), mock.MagicMock()
        monkeypatch.setattr(hrpt, "Orbital", Orbital)
        get_lonlatalt = mock.Mock()
        get_lonlatalt.return_value = (mock.MagicMock(), mock.MagicMock(), mock.MagicMock())
        monkeypatch.setattr(hrpt, "get_lonlatalt", get_lonlatalt)
        SatelliteInterpolator = mock.Mock()
        SatelliteInterpolator.return_value.interpolate.return_value = LONS, LATS
        monkeypatch.setattr(hrpt, "SatelliteInterpolator", SatelliteInterpolator)


class TestHRPTNavigation:
    """Test case for computing HRPT navigation."""

    def test_longitudes_are_returned(self, hrpt_fh, mock_nav):
        """Check that latitudes are returned properly."""
        dataset_id = make_dataid(name="longitude")
        result = hrpt_fh.get_dataset(dataset_id, {})
        assert (result == LONS).all()

    def test_latitudes_are_returned(self, hrpt_fh, mock_nav):
        """Check that latitudes are returned properly."""
        dataset_id = make_dataid(name="latitude")
        result = hrpt_fh.get_dataset(dataset_id, {})
        assert (result == LATS).all()


def to_timecode(dt_time):
    """Convert a datetime to timecode for hrpt scans."""
    year = dt_time.year
    delta = dt_time - datetime(year, 1, 1)
    days = delta.days + 1
    timecode = np.array([0, 0, 0, 0])
    timecode[0] = days << 1
    msecs = int(delta.seconds * 1000 + delta.microseconds / 1000)
    timecode[1] = (msecs >> 20) & 127
    timecode[2] = (msecs >> 10) & 1023
    timecode[3] = msecs & 1023
    return timecode, year


def test_time_seconds():
    """Test conversion of timecode to datetime64."""
    current = datetime.now()
    current = current.replace(microsecond=round(current.microsecond, -3))
    timecode, year = to_timecode(current)
    assert time_seconds(np.array([timecode], "u2"), year) == np.datetime64(current)
