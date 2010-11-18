#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
#   Martin Raspaud <martin.raspaud@smhi.se>
#   Adam Dybbroe <adam.dybbroe@smhi.se>

# This file is part of the mpop.

# mpop is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# mpop is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with mpop.  If not, see <http://www.gnu.org/licenses/>.

"""Miscellaneous image processing tools.
"""
import numpy


def gamma_correction(arr, gamma):
    """Perform gamma correction *g* to an array *arr*, which is assumed
    to be in the range [0.0,1.0], and return the resulting array (same
    range). 
    """
    return arr ** (1.0 / gamma)

def crude_stretch(arr, norm = 1, amin = None, amax = None):
    """Perform simple linear stretching (without any cutoff) and normalize."""

    if(amin is None):
        amin = arr.min()
    if(amax is None):
        amax = arr.max()

    res = (arr - amin) * (norm * 1.0) / (amax - amin)
    res = numpy.where(res > norm, norm, res)
    res = numpy.where(res < 0, 0, res)

    return res
