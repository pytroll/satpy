# Copyright (c) 2017-2025 Satpy developers
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
