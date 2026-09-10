
"""The vii_utils reader tests package."""

import unittest

import satpy.readers.core.metimage

# Constants to be tested
C1 = 1.191062e+8
C2 = 1.4387863e+4
TIE_POINTS_FACTOR = 8
SCAN_ALT_TIE_POINTS = 4
MEAN_EARTH_RADIUS = 6371008.7714


class TestMETimageUtils(unittest.TestCase):
    """Test the vii_utils module."""

    def test_constants(self):
        """Test the constant values."""
        # Test the value of the constants
        assert satpy.readers.core.metimage.C1 == C1
        assert satpy.readers.core.metimage.C2 == C2
        assert satpy.readers.core.metimage.TIE_POINTS_FACTOR == TIE_POINTS_FACTOR
        assert satpy.readers.core.metimage.SCAN_ALT_TIE_POINTS == SCAN_ALT_TIE_POINTS
        assert satpy.readers.core.metimage.MEAN_EARTH_RADIUS == MEAN_EARTH_RADIUS
