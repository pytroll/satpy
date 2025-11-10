#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 Satpy developers
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
"""Unit testing for the HSAF NC enhancement."""

import numpy as np
import xarray as xr
from satpy.enhancements.enhancer import get_enhanced_image, Enhancer


class TestHSAFEnhancement:
    """Test that the HSAF rain rate enhancement is correctly defined and usable."""

    def setup_method(self):
        """Create a small synthetic rain-rate dataset."""
        self.data = xr.DataArray(
            np.array([[0.0, 0.1, 0.2], [5.0, 10.0, 20.0]]),
            dims=("y", "x"),
            name="rr",
            attrs={"standard_name": "rain_rate", "units": "mm/h"}
        )

    # def test_detect_hsaf(self):
    #     enh = Enhancer()
    #     enh.add_sensor_enhancements(["hsaf"])
    #     enh.enhancement_tree.print_tree()

    def test_hsaf_rr_enhancement_colormap_applied(self):
        enh = Enhancer()
        enh.add_sensor_enhancements(["hsaf"])

        img = get_enhanced_image(self.data, enhance=enh)
        enhanced = img.data

        # Result must be an xarray.DataArray
        assert isinstance(enhanced, xr.DataArray)

        # Shape must be (4, y, x) because RGBA colormap applied
        assert enhanced.ndim == 3 and enhanced.shape[0] == 4, (
            f"Expected RGBA image, got shape {enhanced.shape}"
        )

        # Check transparency handling (lowest values → alpha=0)
        alpha_channel = enhanced[3, :, :]  # bands-first: 4th band = alpha
        min_alpha = alpha_channel.min().compute().item()
        max_alpha = alpha_channel.max().compute().item()
        assert min_alpha == 0 and max_alpha == 1, "Alpha channel not correctly mapped"
