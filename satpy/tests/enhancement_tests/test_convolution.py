
"""Unit testing the convolution enhancements functions."""

import numpy as np

from .utils import create_ch1, run_and_check_enhancement


def test_three_d_effect():
    """Test the three_d_effect enhancement function."""
    from satpy.enhancements.convolution import three_d_effect

    ch1 = create_ch1()
    expected = np.array([[
        [np.nan, np.nan, -389.5, -294.5, 826.5],
        [np.nan, np.nan, 85.5, 180.5, 1301.5]]])
    run_and_check_enhancement(three_d_effect, ch1, expected)
