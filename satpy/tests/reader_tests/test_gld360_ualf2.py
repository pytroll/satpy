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
"""Tests for the Vaisala GLD360 UALF2-reader."""

import datetime as dt

import numpy as np
import pytest

from satpy.readers.gld360_ualf2 import UALF2_COLUMN_NAMES, VaisalaGld360Ualf2FileHandler
from satpy.tests.utils import make_dataid

TEST_START_TIME = dt.datetime(2021, 1, 4, 8, 0)
TEST_END_TIME = TEST_START_TIME + dt.timedelta(hours=1)


@pytest.fixture
def fake_file(tmp_path):
    """Create UALF2 file for the tests."""
    fname = tmp_path / "2021.01.04.08.00.txt"
    with open(fname, "w") as fid:
        fid.write(
            u"2\t3\t2021\t1\t4\t8\t0\t1\t51\t-20.8001\t-158.3439\t0\t0\t10\t0\t0\t1\t3\t3\t9.47\t1.91\t1.59\t"
            "0.19\t11.4\t8.8\t0.0\t1\t1\t0\t1\n"
            "2\t3\t2021\t1\t4\t8\t0\t1\t864782486\t0.4381\t-0.8500\t0\t0\t-20\t0\t1\t0\t4\t5\t24.99\t1.95\t1.53\t"
            "1.53\t14.0\t12.9\t-0.0\t0\t1\t0\t1\n"
            "2\t3\t2021\t1\t4\t8\t0\t1\t864782486\t0.4381\t-0.8500\t0\t0\t-20\t0\t1\t0\t4\t5\t24.99\t1.95\t1.53\t"
            "1.53\t14.0\t12.9\t-0.0\t0\t1\t0\t1\n"
            "2\t3\t2021\t1\t4\t8\t0\t1\t897014133\t66.8166\t42.4914\t0\t0\t15\t0\t0\t1\t5\t7\t103.87\t4.33\t1.46\t"
            "0.48\t22.0\t12.3\t0.0\t1\t1\t0\t1"
        )

    return fname


@pytest.fixture
def fake_filehandler(fake_file):
    """Create FileHandler for the tests."""
    filename_info = {}
    filetype_info = {}

    return VaisalaGld360Ualf2FileHandler(
        fake_file, filename_info, filetype_info
    )


def test_ualf2_record_type(fake_filehandler):
    """Test ualf record type."""
    expected = np.array([2, 2, 2])
    dataset_id = make_dataid(name="ualf_record_type")
    dataset_info = {}
    actual = fake_filehandler.get_dataset(dataset_id, dataset_info).values
    np.testing.assert_array_equal(actual, expected)


def test_network_type(fake_filehandler):
    """Test network type."""
    expected = np.array([3, 3, 3])
    dataset_id = make_dataid(name="network_type")
    dataset_info = {}
    actual = fake_filehandler.get_dataset(dataset_id, dataset_info).values
    np.testing.assert_array_equal(actual, expected)


def test_time(fake_filehandler):
    """Test time."""
    expected = np.array(["2021-01-04T08:00:01.000000051", "2021-01-04T08:00:01.864782486",
                         "2021-01-04T08:00:01.897014133"], dtype="datetime64[ns]")
    dataset_id = make_dataid(name="time")
    dataset_info = {}
    actual = fake_filehandler.get_dataset(dataset_id, dataset_info).values
    np.testing.assert_array_equal(actual, expected)


def test_latitude(fake_filehandler):
    """Test latitude."""
    expected = np.array([-20.8001, 0.4381, 66.8166])
    dataset_id = make_dataid(name="latitude")
    dataset_info = {}
    actual = fake_filehandler.get_dataset(dataset_id, dataset_info).values
    np.testing.assert_allclose(actual, expected, rtol=1e-05)


def test_longitude(fake_filehandler):
    """Test longitude."""
    expected = np.array([-158.3439, -0.85, 42.4914])
    dataset_id = make_dataid(name="longitude")
    dataset_info = {}
    actual = fake_filehandler.get_dataset(dataset_id, dataset_info).values
    np.testing.assert_allclose(actual, expected, rtol=1e-05)


def test_altitude(fake_filehandler):
    """Test altitude."""
    expected = np.array([0, 0, 0])
    dataset_id = make_dataid(name="altitude")
    dataset_info = {}
    actual = fake_filehandler.get_dataset(dataset_id, dataset_info).values
    np.testing.assert_array_equal(actual, expected)


def test_altitude_uncertainty(fake_filehandler):
    """Test altitude uncertainty."""
    expected = np.array([0, 0, 0])
    dataset_id = make_dataid(name="altitude_uncertainty")
    dataset_info = {}
    actual = fake_filehandler.get_dataset(dataset_id, dataset_info).values
    np.testing.assert_array_equal(actual, expected)


def test_peak_current(fake_filehandler):
    """Test peak current."""
    expected = np.array([10, -20, 15])
    dataset_id = make_dataid(name="peak_current")
    dataset_info = {}
    actual = fake_filehandler.get_dataset(dataset_id, dataset_info).values
    np.testing.assert_array_equal(actual, expected)


def test_vhf_range(fake_filehandler):
    """Test vhf range."""
    expected = np.array([0, 0, 0])
    dataset_id = make_dataid(name="vhf_range")
    dataset_info = {}
    actual = fake_filehandler.get_dataset(dataset_id, dataset_info).values
    np.testing.assert_array_equal(actual, expected)


def test_multiplicity_flash(fake_filehandler):
    """Test multiplicity flash."""
    expected = np.array([0, 1, 0])
    dataset_id = make_dataid(name="multiplicity_flash")
    dataset_info = {}
    actual = fake_filehandler.get_dataset(dataset_id, dataset_info).values
    np.testing.assert_array_equal(actual, expected)


def test_cloud_pulse_count(fake_filehandler):
    """Test cloud pulse count."""
    expected = np.array([1, 0, 1])
    dataset_id = make_dataid(name="cloud_pulse_count")
    dataset_info = {}
    actual = fake_filehandler.get_dataset(dataset_id, dataset_info).values
    np.testing.assert_array_equal(actual, expected)


def test_number_of_sensors(fake_filehandler):
    """Test number of sensors."""
    expected = np.array([3, 4, 5])
    dataset_id = make_dataid(name="number_of_sensors")
    dataset_info = {}
    actual = fake_filehandler.get_dataset(dataset_id, dataset_info).values
    np.testing.assert_array_equal(actual, expected)


def test_degree_freedom_for_location(fake_filehandler):
    """Test degree freedom for location."""
    expected = np.array([3, 5, 7])
    dataset_id = make_dataid(name="degree_freedom_for_location")
    dataset_info = {}
    actual = fake_filehandler.get_dataset(dataset_id, dataset_info).values
    np.testing.assert_array_equal(actual, expected)


def test_error_ellipse_angle(fake_filehandler):
    """Test error ellipse angle."""
    expected = np.array([9.47, 24.99, 103.87])
    dataset_id = make_dataid(name="error_ellipse_angle")
    dataset_info = {}
    actual = fake_filehandler.get_dataset(dataset_id, dataset_info).values
    np.testing.assert_allclose(actual, expected, rtol=1e-05)


def test_error_ellipse_max_axis_length(fake_filehandler):
    """Test error ellipse max axis length."""
    expected = np.array([1.91, 1.95, 4.33])
    dataset_id = make_dataid(name="error_ellipse_max_axis_length")
    dataset_info = {}
    actual = fake_filehandler.get_dataset(dataset_id, dataset_info).values
    np.testing.assert_allclose(actual, expected, rtol=1e-05)


def test_error_ellipse_min_axis_length(fake_filehandler):
    """Test error ellipse min axis length."""
    expected = np.array([1.59, 1.53, 1.46])
    dataset_id = make_dataid(name="error_ellipse_min_axis_length")
    dataset_info = {}
    actual = fake_filehandler.get_dataset(dataset_id, dataset_info).values
    np.testing.assert_allclose(actual, expected, rtol=1e-05)


def test_chi_squared_value_location_optimization(fake_filehandler):
    """Test chi squared value location optimization."""
    expected = np.array([0.19, 1.53, 0.48])
    dataset_id = make_dataid(name="chi_squared_value_location_optimization")
    dataset_info = {}
    actual = fake_filehandler.get_dataset(dataset_id, dataset_info).values
    np.testing.assert_allclose(actual, expected, rtol=1e-05)


def test_wave_form_rise_time(fake_filehandler):
    """Test wave form rise time."""
    expected = np.array([11.4, 14., 22.])
    dataset_id = make_dataid(name="wave_form_rise_time")
    dataset_info = {}
    actual = fake_filehandler.get_dataset(dataset_id, dataset_info).values
    np.testing.assert_allclose(actual, expected, rtol=1e-05)


def test_wave_form_peak_to_zero_time(fake_filehandler):
    """Test wave form peak to zero time."""
    expected = np.array([8.8, 12.9, 12.3])
    dataset_id = make_dataid(name="wave_form_peak_to_zero_time")
    dataset_info = {}
    actual = fake_filehandler.get_dataset(dataset_id, dataset_info).values
    np.testing.assert_allclose(actual, expected, rtol=1e-05)


def test_wave_form_max_rate_of_rise(fake_filehandler):
    """Test wave form max rate of rise."""
    expected = np.array([0, 0, 0])
    dataset_id = make_dataid(name="wave_form_max_rate_of_rise")
    dataset_info = {}
    actual = fake_filehandler.get_dataset(dataset_id, dataset_info).values
    np.testing.assert_array_equal(actual, expected)


def test_cloud_indicator(fake_filehandler):
    """Test cloud indicator."""
    expected = np.array([1, 0, 1])
    dataset_id = make_dataid(name="cloud_indicator")
    dataset_info = {}
    actual = fake_filehandler.get_dataset(dataset_id, dataset_info).values
    np.testing.assert_array_equal(actual, expected)


def test_angle_indicator(fake_filehandler):
    """Test angle indicator."""
    expected = np.array([1, 1, 1])
    dataset_id = make_dataid(name="angle_indicator")
    dataset_info = {}
    actual = fake_filehandler.get_dataset(dataset_id, dataset_info).values
    np.testing.assert_array_equal(actual, expected)


def test_signal_indicator(fake_filehandler):
    """Test signal indicator."""
    expected = np.array([0, 0, 0])
    dataset_id = make_dataid(name="signal_indicator")
    dataset_info = {}
    actual = fake_filehandler.get_dataset(dataset_id, dataset_info).values
    np.testing.assert_array_equal(actual, expected)


def test_timing_indicator(fake_filehandler):
    """Test timing indicator."""
    expected = np.array([1, 1, 1])
    dataset_id = make_dataid(name="timing_indicator")
    dataset_info = {}
    actual = fake_filehandler.get_dataset(dataset_id, dataset_info).values
    np.testing.assert_array_equal(actual, expected)


def test_pad_nanoseconds(fake_filehandler):
    """Test pad nanoseconds."""
    expected = "000000013"
    actual = fake_filehandler.pad_nanoseconds(13)
    np.testing.assert_string_equal(actual, expected)


def test_nanoseconds_index():
    """Test nanosecond column being after seconds."""
    expected = UALF2_COLUMN_NAMES.index("nanosecond")
    actual = UALF2_COLUMN_NAMES.index("second") + 1
    np.testing.assert_array_equal(actual, expected)


def test_column_names_length():
    """Test correct number of column names."""
    expected = 30
    actual = len(UALF2_COLUMN_NAMES)
    np.testing.assert_equal(actual, expected)


@pytest.fixture
def fake_scn(fake_file):
    """Create fake file for tests."""
    from satpy import Scene
    scn = Scene(reader="gld360_ualf2", filenames=[fake_file])
    return scn


def test_scene_attributes(fake_scn):
    """Test for correct start and end times."""
    np.testing.assert_equal(fake_scn.start_time, TEST_START_TIME)
    np.testing.assert_equal(fake_scn.end_time, TEST_END_TIME)


def test_scene_load(fake_scn):
    """Test data loading through Scene-object."""
    fake_scn.load(["time", "latitude", "longitude"])
    assert "time" in fake_scn
    assert "latitude" in fake_scn
    assert "longitude" in fake_scn


def test_area_(fake_scn):
    """Test correct area instance type."""
    from pyresample.geometry import SwathDefinition
    fake_scn.load(["time"])
    assert isinstance(fake_scn["time"].attrs["area"], SwathDefinition)
