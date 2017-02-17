#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Composite classes for the VIIRS instrument.
"""

import logging

from satpy.composites import RGBCompositor
from satpy.dataset import combine_info

LOG = logging.getLogger(__name__)


def overlay(top, bottom):
    """Blending two layers.

    from: https://docs.gimp.org/en/gimp-concepts-layer-modes.html
    """
    maxval = max(top.max(), bottom.max())

    res = 2 * top
    res /= maxval
    res -= 1
    res *= bottom
    res += 2 * top
    res *= bottom
    res /= maxval

    return res


class SARIce(RGBCompositor):
    """The SAR Ice composite."""

    def __call__(self, projectables, *args, **kwargs):
        """Create the SAR Ice composite."""
        (mhh, mhv) = projectables
        green = overlay(mhh, mhv)
        green.info = combine_info(mhh, mhv)

        return super(SARIce, self).__call__((mhv, green, mhh), *args, **kwargs)
