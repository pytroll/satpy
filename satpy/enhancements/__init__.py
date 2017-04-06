#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017.

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

# This file is part of satpy.

# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.

"""Enhancements."""

import numpy as np


def stretch(img, **kwargs):
    """Perform stretch."""
    img.stretch(**kwargs)


def gamma(img, **kwargs):
    """Perform gamma correction."""
    img.gamma(**kwargs)


def invert(img, *args):
    """Perform inversion."""
    img.invert(*args)


def cira_stretch(img, **kwargs):
    """Logarithmic stretch adapted to human vision."""
    for chn in img.channels:
        chn /= 100
        np.ma.log10(chn, out=chn)
        chn -= np.log10(0.0223)
        chn /= 1.0 - np.log10(0.0223)
        chn /= 0.75
