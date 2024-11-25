#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2018 Satpy developers
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
"""Common functionality for FCI data readers."""
from __future__ import annotations


def calculate_area_extent(area_dict):
        """Calculate the area extent seen by MTG FCI instrument.

        Since the center of the FCI grids is located at the interface between the pixels, there are equally many
        pixels (e.g. 5568/2 = 2784 for 2km grid) in each direction from the center points. Hence, the area extent
        can be easily computed by simply adding and subtracting half the width and height from teh centre point (=0).

        Args:
            area_dict: A dictionary containing the required parameters
                ncols: number of pixels in east-west direction
                nlines: number of pixels in south-north direction
                column_step: Pixel resulution in meters in east-west direction
                line_step: Pixel resulution in meters in south-north direction
        Returns:
            tuple: An area extent for the scene defined by the lower left and
                   upper right corners

        """
        ncols = area_dict["ncols"]
        nlines = area_dict["nlines"]
        column_step = area_dict["column_step"]
        line_step = area_dict["line_step"]

        ll_c = (0 - ncols / 2.) * column_step
        ll_l = (0 + nlines / 2.) * line_step
        ur_c = (0 + ncols / 2.) * column_step
        ur_l = (0 - nlines / 2.) * line_step

        return (ll_c, ll_l, ur_c, ur_l)
