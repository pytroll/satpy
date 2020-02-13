#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Satpy developers
#
# satpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# satpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with satpy.  If not, see <http://www.gnu.org/licenses/>.

"""The vii_utils reader tests package."""

import numpy as np

import satpy.readers.vii_utils

import unittest

# Constants to be tested
C1 = 1.191062e+8
C2 = 1.4387863e+4
TIE_POINTS_FACTOR = 8
SCAN_ALT_TIE_POINTS = 4
MEAN_EARTH_RADIUS = 6371008.7714

TEST_VALID_ALT_TIE_POINTS = SCAN_ALT_TIE_POINTS * 5
TEST_INVALID_ALT_TIE_POINTS = SCAN_ALT_TIE_POINTS * 5 + 1
TEST_ACT_TIE_POINTS = 10


class Test_vii_utils(unittest.TestCase):
    """Test the vii_utils module."""

    def setUp(self):
        """Set up the test."""
        # Create two arrays for the interpolation test
        # The first has a valid number of n_tie_alt points (multiple of SCAN_ALT_TIE_POINTS)
        self.valid_data_for_interpolation = np.array(
            range(TEST_VALID_ALT_TIE_POINTS * TEST_ACT_TIE_POINTS)
            ).reshape(TEST_ACT_TIE_POINTS, TEST_VALID_ALT_TIE_POINTS)
        # The first has an invalid number of n_tie_alt points (not multiple of SCAN_ALT_TIE_POINTS)
        self.invalid_data_for_interpolation = np.array(
            range(TEST_INVALID_ALT_TIE_POINTS * TEST_ACT_TIE_POINTS)
            ).reshape(TEST_ACT_TIE_POINTS, TEST_INVALID_ALT_TIE_POINTS)

    def tearDown(self):
        """Tear down the test."""
        # Nothing to do
        pass

    def test_constants(self):
        """Test the constant values."""
        # Test the value of the constants
        self.assertEqual(satpy.readers.vii_utils.C1, C1)
        self.assertEqual(satpy.readers.vii_utils.C2, C2)
        self.assertEqual(satpy.readers.vii_utils.TIE_POINTS_FACTOR, TIE_POINTS_FACTOR)
        self.assertEqual(satpy.readers.vii_utils.SCAN_ALT_TIE_POINTS, SCAN_ALT_TIE_POINTS)
        self.assertEqual(satpy.readers.vii_utils.MEAN_EARTH_RADIUS, MEAN_EARTH_RADIUS)

    def test_interpolation(self):
        """# Test the interpolation routine with valid and invalid input."""
        # Test the interpolation routine with valid input
        result_valid = satpy.readers.vii_utils.tie_points_interpolation(self.valid_data_for_interpolation)

        act_points_interp = (TEST_ACT_TIE_POINTS - 1) * TIE_POINTS_FACTOR
        num_scans = TEST_VALID_ALT_TIE_POINTS // SCAN_ALT_TIE_POINTS
        scan_alt_points_interp = (SCAN_ALT_TIE_POINTS - 1) * TIE_POINTS_FACTOR

        # It is easier to check the delta between interpolated points, which must be 1/8 of the original delta
        # Across the track, it is possible to check the delta on the entire array
        delta_axis_0 = 1.0 * TEST_VALID_ALT_TIE_POINTS / TIE_POINTS_FACTOR
        self.assertTrue(np.allclose(
            np.diff(result_valid, axis=0),
            np.ones((act_points_interp - 1, num_scans * scan_alt_points_interp)) * delta_axis_0
            ))

        delta_axis_1 = 1.0 / TIE_POINTS_FACTOR
        # Along the track, it is necessary to check the delta on each scan separately
        for i in range(num_scans):
            first_index = i*(SCAN_ALT_TIE_POINTS-1)*TIE_POINTS_FACTOR
            last_index = (i+1)*(SCAN_ALT_TIE_POINTS-1)*TIE_POINTS_FACTOR
            result_per_scan = result_valid[:, first_index:last_index]
            self.assertTrue(np.allclose(
                np.diff(result_per_scan, axis=1),
                np.ones((act_points_interp, (SCAN_ALT_TIE_POINTS-1)*TIE_POINTS_FACTOR - 1)) * delta_axis_1
                ))

        # Test the interpolation routine with invalid input
        with self.assertRaises(ValueError):
            satpy.readers.vii_utils.tie_points_interpolation(self.invalid_data_for_interpolation)


def suite():
    """Build the test suite for test_scene."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()

    mysuite.addTest(loader.loadTestsFromTestCase(Test_vii_utils))

    return mysuite


if __name__ == '__main__':
    unittest.main()
