#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Satpy developers
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

import unittest

import satpy.readers.core.vii

# Constants to be tested
C1 = 1.191062e+8
C2 = 1.4387863e+4
TIE_POINTS_FACTOR = 8
SCAN_ALT_TIE_POINTS = 4
MEAN_EARTH_RADIUS = 6371008.7714


class TestViiUtils(unittest.TestCase):
    """Test the vii_utils module."""

    def test_constants(self):
        """Test the constant values."""
        # Test the value of the constants
        assert satpy.readers.core.vii.C1 == C1
        assert satpy.readers.core.vii.C2 == C2
        assert satpy.readers.core.vii.TIE_POINTS_FACTOR == TIE_POINTS_FACTOR
        assert satpy.readers.core.vii.SCAN_ALT_TIE_POINTS == SCAN_ALT_TIE_POINTS
        assert satpy.readers.core.vii.MEAN_EARTH_RADIUS == MEAN_EARTH_RADIUS
