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

from pyresample.geometry import AreaDefinition
from satpy.composites import IncompatibleAreas, RGBCompositor
from satpy.projectable import Projectable, combine_info
from satpy.readers import DatasetID

LOG = logging.getLogger(__name__)


def overlay(top, bottom):
    """Blending two layers.

    from: https://docs.gimp.org/en/gimp-concepts-layer-modes.html
    """
    mx = max(top.max(), bottom.max())

    res = 2 * top
    res /= mx
    res -= 1
    res *= bottom
    res += 2 * top
    res *= bottom
    res /= mx

    return res


class SARIce(RGBCompositor):
    """Corrector of the AHI green band to compensate for the deficit of
    chlorophyl signal.
    """

    def __call__(self, projectables, *args, **kwargs):
        """Create the SAR Ice composite."""
        (hh, hv) = projectables
        green = overlay(hh, hv)

        return super(SARIce, self).__call__((hv, green, hh), *args, **kwargs)
