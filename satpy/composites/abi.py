#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2015-2017

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
"""Composite classes for the AHI instrument.
"""

import logging
import numpy as np

from satpy.composites import RGBCompositor

LOG = logging.getLogger(__name__)


def four_element_average(d):
    """Average every 4 elements (2x2) in a 2D array"""
    rows, cols = d.shape
    new_shape = (int(rows / 2.), 2, int(cols / 2.), 2)
    return np.ma.mean(d.reshape(new_shape), axis=(1, 3))


class TrueColor2km(RGBCompositor):
    """True Color ABI compositor assuming all bands are the same resolution"""

    def __call__(self, projectables, **info):

        c01, c02, c03 = projectables

        r = c02
        b = c01
        g = (c01 + c02) / 2 * 0.93 + 0.07 * c03

        return super(TrueColor2km, self).__call__((r, g, b), **info)


class TrueColor(RGBCompositor):
    """Ratio sharpened full resolution true color"""

    def __call__(self, projectables, **info):
        c01, c02, c03 = projectables
        r = c02
        b = np.repeat(np.repeat(c01, 2, axis=0), 2, axis=1)
        c03_high = np.repeat(np.repeat(c03, 2, axis=0), 2, axis=1)
        g = (b + r) / 2 * 0.93 + 0.07 * c03_high

        low_res_red = four_element_average(r)
        low_res_red = np.repeat(np.repeat(low_res_red, 2, axis=0), 2, axis=1)
        ratio = r / low_res_red

        # make sure metadata is copied over
        # copy red channel area to get correct resolution
        g *= ratio
        g.info = c03.info.copy()
        g.info['area'] = r.info['area']
        b *= ratio
        b.info = c01.info.copy()
        b.info['area'] = r.info['area']
        return super(TrueColor, self).__call__((r, g, b), **info)
