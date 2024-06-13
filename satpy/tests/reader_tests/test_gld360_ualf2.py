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

import tempfile
import unittest

import numpy as np

from satpy.readers.gld360_ualf2 import VaisalaGld360Ualf2FileHandler
from satpy.tests.utils import make_dataid


class TestVaisalaGld360Ualf2FileHandler(unittest.TestCase):
    """Test class for the FileHandler."""

    def test_vaisala_gld360(self):
        """Test basic functionality for vaisala file handler."""
        expected_ualf_record_type = np.array([2, 2, 2])
        expected_network_type = np.array([3, 3, 3])
        expected_time = np.array(["2021-01-04T08:00:01.000000051", "2021-01-04T08:00:01.864782486",
                                  "2021-01-04T08:00:01.897014133"], dtype="datetime64[ns]")
        expected_latitude = np.array([-20.8001, 0.4381, 66.8166])
        expected_longitude = np.array([-158.3439, -0.85, 42.4914])
        expected_altitude = np.array([0, 0, 0])
        expected_altitude_uncertainty = np.array([0, 0, 0])
        expected_peak_current = np.array([10, -20, 15])
        expected_vhf_range = np.array([0, 0, 0])
        expected_multiplicity_flash = np.array([0, 1, 0])
        expected_cloud_pulse_count = np.array([1, 0, 1])
        expected_number_of_sensors = np.array([3, 4, 5])
        expected_degree_freedom_for_location = np.array([3, 5, 7])
        expected_error_ellipse_angle = np.array([9.47, 24.99, 103.87])
        expected_error_ellipse_max_axis_length = np.array([1.91, 1.95, 4.33])
        expected_error_ellipse_min_axis_length = np.array([1.59, 1.53, 1.46])
        expected_chi_squared_value_location_optimization = np.array([0.19, 1.53, 0.48])
        expected_wave_form_rise_time = np.array([11.4, 14., 22.])
        expected_wave_form_peak_to_zero_time = np.array([8.8, 12.9, 12.3])
        expected_wave_form_max_rate_of_rise = np.array([0, 0, 0])
        expected_cloud_indicator = np.array([1, 0, 1])
        expected_angle_indicator = np.array([1, 1, 1])
        expected_signal_indicator = np.array([0, 0, 0])
        expected_timing_indicator = np.array([1, 1, 1])

        with tempfile.NamedTemporaryFile(mode="w") as t:
            t.write(
                u"2\t3\t2021\t1\t4\t8\t0\t1\t51\t-20.8001\t-158.3439\t0\t0\t10\t0\t0\t1\t3\t3\t9.47\t1.91\t1.59\t"
                "0.19\t11.4\t8.8\t0.0\t1\t1\t0\t1\n"
                "2\t3\t2021\t1\t4\t8\t0\t1\t864782486\t0.4381\t-0.8500\t0\t0\t-20\t0\t1\t0\t4\t5\t24.99\t1.95\t1.53\t"
                "1.53\t14.0\t12.9\t-0.0\t0\t1\t0\t1\n"
                "2\t3\t2021\t1\t4\t8\t0\t1\t864782486\t0.4381\t-0.8500\t0\t0\t-20\t0\t1\t0\t4\t5\t24.99\t1.95\t1.53\t"
                "1.53\t14.0\t12.9\t-0.0\t0\t1\t0\t1\n"
                "2\t3\t2021\t1\t4\t8\t0\t1\t897014133\t66.8166\t42.4914\t0\t0\t15\t0\t0\t1\t5\t7\t103.87\t4.33\t1.46\t"
                "0.48\t22.0\t12.3\t0.0\t1\t1\t0\t1"
                )

            t.seek(0)
            filename_info = {}
            filetype_info = {}

            self.handler = VaisalaGld360Ualf2FileHandler(
                t.name, filename_info, filetype_info
                )

            # Test ualf record type.
            dataset_id = make_dataid(name="ualf_record_type")
            dataset_info = {}
            result = self.handler.get_dataset(dataset_id, dataset_info).values
            np.testing.assert_array_equal(result, expected_ualf_record_type)

            # Test network type.
            dataset_id = make_dataid(name="network_type")
            dataset_info = {}
            result = self.handler.get_dataset(dataset_id, dataset_info).values
            np.testing.assert_array_equal(result, expected_network_type)

            # Test time.
            dataset_id = make_dataid(name="time")
            dataset_info = {}
            result = self.handler.get_dataset(dataset_id, dataset_info).values
            np.testing.assert_array_equal(result, expected_time)

            # Test latitude.
            dataset_id = make_dataid(name="latitude")
            dataset_info = {}
            result = self.handler.get_dataset(dataset_id, dataset_info).values
            np.testing.assert_allclose(result, expected_latitude, rtol=1e-05)

            # Test longitude.
            dataset_id = make_dataid(name="longitude")
            dataset_info = {}
            result = self.handler.get_dataset(dataset_id, dataset_info).values
            np.testing.assert_allclose(result, expected_longitude, rtol=1e-05)

            # Test altitude.
            dataset_id = make_dataid(name="altitude")
            dataset_info = {}
            result = self.handler.get_dataset(dataset_id, dataset_info).values
            np.testing.assert_array_equal(result, expected_altitude)

            # Test altitude uncertainty.
            dataset_id = make_dataid(name="altitude_uncertainty")
            dataset_info = {}
            result = self.handler.get_dataset(dataset_id, dataset_info).values
            np.testing.assert_array_equal(result, expected_altitude_uncertainty)

            # Test peak current.
            dataset_id = make_dataid(name="peak_current")
            dataset_info = {}
            result = self.handler.get_dataset(dataset_id, dataset_info).values
            np.testing.assert_array_equal(result, expected_peak_current)

            # Test vhf range.
            dataset_id = make_dataid(name="vhf_range")
            dataset_info = {}
            result = self.handler.get_dataset(dataset_id, dataset_info).values
            np.testing.assert_array_equal(result, expected_vhf_range)

            # Test multiplicity flash.
            dataset_id = make_dataid(name="multiplicity_flash")
            dataset_info = {}
            result = self.handler.get_dataset(dataset_id, dataset_info).values
            np.testing.assert_array_equal(result, expected_multiplicity_flash)

            # Test cloud pulse count.
            dataset_id = make_dataid(name="cloud_pulse_count")
            dataset_info = {}
            result = self.handler.get_dataset(dataset_id, dataset_info).values
            np.testing.assert_array_equal(result, expected_cloud_pulse_count)

            # Test number of sensors.
            dataset_id = make_dataid(name="number_of_sensors")
            dataset_info = {}
            result = self.handler.get_dataset(dataset_id, dataset_info).values
            np.testing.assert_array_equal(result, expected_number_of_sensors)

            # Test degree freedom for location.
            dataset_id = make_dataid(name="degree_freedom_for_location")
            dataset_info = {}
            result = self.handler.get_dataset(dataset_id, dataset_info).values
            np.testing.assert_array_equal(result, expected_degree_freedom_for_location)

            # Test error ellipse angle.
            dataset_id = make_dataid(name="error_ellipse_angle")
            dataset_info = {}
            result = self.handler.get_dataset(dataset_id, dataset_info).values
            np.testing.assert_allclose(result, expected_error_ellipse_angle, rtol=1e-05)

            # Test error ellipse max axis length.
            dataset_id = make_dataid(name="error_ellipse_max_axis_length")
            dataset_info = {}
            result = self.handler.get_dataset(dataset_id, dataset_info).values
            np.testing.assert_allclose(result, expected_error_ellipse_max_axis_length, rtol=1e-05)

            # Test error ellipse min axis length.
            dataset_id = make_dataid(name="error_ellipse_min_axis_length")
            dataset_info = {}
            result = self.handler.get_dataset(dataset_id, dataset_info).values
            np.testing.assert_allclose(result, expected_error_ellipse_min_axis_length, rtol=1e-05)

            # Test chi squared value location optimization.
            dataset_id = make_dataid(name="chi_squared_value_location_optimization")
            dataset_info = {}
            result = self.handler.get_dataset(dataset_id, dataset_info).values
            np.testing.assert_allclose(result, expected_chi_squared_value_location_optimization, rtol=1e-05)

            # Test wave form rise time.
            dataset_id = make_dataid(name="wave_form_rise_time")
            dataset_info = {}
            result = self.handler.get_dataset(dataset_id, dataset_info).values
            np.testing.assert_allclose(result, expected_wave_form_rise_time, rtol=1e-05)

            # Test wave form peak to zero time.
            dataset_id = make_dataid(name="wave_form_peak_to_zero_time")
            dataset_info = {}
            result = self.handler.get_dataset(dataset_id, dataset_info).values
            np.testing.assert_allclose(result, expected_wave_form_peak_to_zero_time, rtol=1e-05)

            # Test wave form max rate of rise.
            dataset_id = make_dataid(name="wave_form_max_rate_of_rise")
            dataset_info = {}
            result = self.handler.get_dataset(dataset_id, dataset_info).values
            np.testing.assert_array_equal(result, expected_wave_form_max_rate_of_rise)

            # Test cloud indicator.
            dataset_id = make_dataid(name="cloud_indicator")
            dataset_info = {}
            result = self.handler.get_dataset(dataset_id, dataset_info).values
            np.testing.assert_array_equal(result, expected_cloud_indicator)

            # Test angle indicator.
            dataset_id = make_dataid(name="angle_indicator")
            dataset_info = {}
            result = self.handler.get_dataset(dataset_id, dataset_info).values
            np.testing.assert_array_equal(result, expected_angle_indicator)

            # Test signal indicator.
            dataset_id = make_dataid(name="signal_indicator")
            dataset_info = {}
            result = self.handler.get_dataset(dataset_id, dataset_info).values
            np.testing.assert_array_equal(result, expected_signal_indicator)

            # Test timing indicator.
            dataset_id = make_dataid(name="timing_indicator")
            dataset_info = {}
            result = self.handler.get_dataset(dataset_id, dataset_info).values
            np.testing.assert_array_equal(result, expected_timing_indicator)

            t.close()
