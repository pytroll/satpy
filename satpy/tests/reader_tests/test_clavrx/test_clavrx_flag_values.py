import numpy as np
import pytest
from satpy.readers.clavrx import _CLAVRxHelper

class TestVerifyFlagValues:
    """Test _verify_flag_values validation logic."""

    @pytest.mark.parametrize(
        ("flag_values", "expected_invalid"),
        [
            # Valid cases: strictly increasing
            ([0, 1, 2, 3], False),
            (np.array([0, 1, 2, 3]), False),
            
            # Invalid cases: repeated/corrupted values (e.g., 0b, 0b, 0b)
            ([0, 0, 0, 0], True),
            (np.array([0, 0, 0, 0]), True),
            ([0, 1, 1, 2], True),      # non-strictly increasing
            ([3, 2, 1, 0], True),      # decreasing
            
            # Edge cases: None or single value
            (None, False),
            ([None], False),
            ([0], False),
        ]
    )
    def test_verify_flag_values(self, flag_values, expected_invalid):
        """Test detection of invalid flag_values arrays."""
        assert _CLAVRxHelper._verify_flag_values(flag_values) == expected_invalid
