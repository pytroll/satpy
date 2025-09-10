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
"""Unit testing the stretching enhancements functions."""

import numpy as np
import pytest

from .utils import create_ch1, create_ch2, create_rgb, run_and_check_enhancement, run_and_check_enhancement_with_dtype


class TestEnhancementsStretching:
    """Class for testing enhancements in satpy.enhancements.contrast module."""

    def setup_method(self):
        """Create test data used by every test."""
        self.ch1 = create_ch1()
        self.ch2 = create_ch2()
        self.rgb = create_rgb()

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_cira_stretch(self, dtype):
        """Test applying the cira_stretch."""
        from satpy.enhancements.contrast import cira_stretch

        expected = np.array([[
            [np.nan, -7.04045974, -7.04045974, 0.79630132, 0.95947296],
            [1.05181359, 1.11651012, 1.16635571, 1.20691137, 1.24110186]]], dtype=dtype)
        run_and_check_enhancement_with_dtype(cira_stretch, self.ch1.astype(dtype), expected)

    def test_reinhard(self):
        """Test the reinhard algorithm."""
        from satpy.enhancements.contrast import reinhard_to_srgb
        expected = np.array([[[np.nan, 0., 0., 0.93333793, 1.29432402],
                              [1.55428709, 1.76572249, 1.94738635, 2.10848544, 2.25432809]],

                             [[np.nan, 0., 0., 0.93333793, 1.29432402],
                              [1.55428709, 1.76572249, 1.94738635, 2.10848544, 2.25432809]],

                             [[np.nan, 0., 0., 0.93333793, 1.29432402],
                              [1.55428709, 1.76572249, 1.94738635, 2.10848544, 2.25432809]]])
        run_and_check_enhancement(reinhard_to_srgb, self.rgb, expected)

    def test_piecewise_linear_stretch(self):
        """Test the piecewise_linear_stretch enhancement function."""
        from satpy.enhancements.contrast import piecewise_linear_stretch
        expected = np.array([[
            [np.nan, 0., 0., 0.44378, 0.631734],
            [0.737562, 0.825041, 0.912521, 1., 1.]]])
        run_and_check_enhancement(piecewise_linear_stretch,
                                  self.ch2 / 100.0,
                                  expected,
                                  xp=[0., 25., 55., 100., 255.],
                                  fp=[0., 90., 140., 175., 255.],
                                  reference_scale_factor=255,
                                  )

    def test_btemp_threshold(self):
        """Test applying the cira_stretch."""
        from satpy.enhancements.contrast import btemp_threshold

        expected = np.array([[
            [np.nan, 0.946207, 0.892695, 0.839184, 0.785672],
            [0.73216, 0.595869, 0.158745, -0.278379, -0.715503]]])
        run_and_check_enhancement(btemp_threshold, self.ch1, expected,
                                  min_in=-200, max_in=500, threshold=350)

    def tearDown(self):
        """Clean up."""
