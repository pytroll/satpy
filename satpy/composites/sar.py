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

import numpy as np
from satpy.composites import GenericCompositor
from satpy.dataset import combine_metadata

LOG = logging.getLogger(__name__)


def overlay(top, bottom, maxval=None):
    """Blending two layers.

    from: https://docs.gimp.org/en/gimp-concepts-layer-modes.html
    """
    if maxval is None:
        maxval = np.maximum(top.max(), bottom.max())

    res = ((2 * top / maxval - 1) * bottom + 2 * top) * bottom / maxval
    return res.clip(min=0)


class SARIce(GenericCompositor):
    """The SAR Ice composite."""

    def __call__(self, projectables, *args, **kwargs):
        """Create the SAR Ice composite."""
        (mhh, mhv) = projectables
        ch1attrs = mhh.attrs
        ch2attrs = mhv.attrs
        mhh = np.sqrt(mhh ** 2 + 0.002) - 0.04
        mhv = np.sqrt(mhv ** 2 + 0.002) - 0.04
        mhh.attrs = ch1attrs
        mhv.attrs = ch2attrs
        green = overlay(mhh, mhv, 30) * 1000
        green.attrs = combine_metadata(mhh, mhv)

        return super(SARIce, self).__call__((mhv, green, mhh), *args, **kwargs)


class SARIceLegacy(GenericCompositor):
    """The SAR Ice composite, legacy version with dynamic stretching."""

    def __call__(self, projectables, *args, **kwargs):
        """Create the SAR RGB composite."""

        (mhh, mhv) = projectables
        green = overlay(mhh, mhv)
        green.attrs = combine_metadata(mhh, mhv)

        return super(SARIceLegacy, self).__call__((mhv, green, mhh), *args, **kwargs)


class SARRGB(GenericCompositor):
    """The SAR RGB composite."""

    def __call__(self, projectables, *args, **kwargs):
        """Create the SAR RGB composite."""

        (mhh, mhv) = projectables
        green = overlay(mhh, mhv)
        green.attrs = combine_metadata(mhh, mhv)

        return super(SARRGB, self).__call__((-mhv, -green, -mhh), *args, **kwargs)
        # (mhh, mhv) = projectables
        # green = 1 - (overlay(mhh, mhv) / .0044)
        # red = 1 - (mhv / .223)
        # blue = 1 - (mhh / .596)
        # import xarray as xr
        # import xarray.ufuncs as xu
        # from functools import reduce
        #
        # mask1 = reduce(xu.logical_and,
        #                [abs(green - blue) < 10 / 255.,
        #                 red - blue >= 0,
        #                 xu.maximum(green, blue) < 200 / 255.])
        #
        # mask2 = xu.logical_and(abs(green - blue) < 40 / 255.,
        #                        red - blue > 40 / 255.)
        #
        # mask3 = xu.logical_and(red - blue > 10 / 255.,
        #                        xu.maximum(green, blue) < 120 / 255.)
        #
        # mask4 = reduce(xu.logical_and,
        #                [red < 70 / 255.,
        #                 green < 60 / 255.,
        #                 blue < 60 / 255.])
        #
        # mask5 = reduce(xu.logical_and,
        #                [red < 80 / 255.,
        #                 green < 80 / 255.,
        #                 blue < 80 / 255.,
        #                 xu.minimum(xu.minimum(red, green), blue) < 30 / 255.])
        #
        # mask6 = reduce(xu.logical_and,
        #                [red < 110 / 255.,
        #                 green < 110 / 255.,
        #                 blue < 110 / 255.,
        #                 xu.minimum(red, green) < 10 / 255.])
        #
        # mask = reduce(xu.logical_or, [mask1, mask2, mask3, mask4, mask5, mask6])
        #
        # red = xr.where(mask, 230 / 255. - red, red).clip(min=0)
        # green = xr.where(mask, 1 - green, green)
        # blue = xr.where(mask, 1 - blue, blue)
        #
        # attrs = combine_metadata(mhh, mhv)
        # green.attrs = attrs
        # red.attrs = attrs
        # blue.attrs = attrs
        #
        # return super(SARRGB, self).__call__((mhv, green, mhh), *args, **kwargs)


class SARQuickLook(GenericCompositor):
    """The SAR QuickLook composite."""

    def __call__(self, projectables, *args, **kwargs):
        """Create the SAR QuickLook composite."""
        (mhh, mhv) = projectables

        blue = mhv / mhh
        blue.attrs = combine_metadata(mhh, mhv)

        return super(SARQuickLook, self).__call__((mhh, mhv, blue), *args, **kwargs)
