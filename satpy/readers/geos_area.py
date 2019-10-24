#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 Satpy developers
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
"""Geostationary Projection / Area computations
***************************

This module computes properties and area definitions for geostationary
satellites. It is designed to be a common module that can be called by
all geostationary satellite readers and uses commonly-included parameters
such as the CFAC/LFAC values, satellite position, etc, to compute the
correct area definition.
"""

import numpy as np
from pyresample import geometry


def get_xy_from_linecol(self, line, col, offsets, factors):
    """Get the intermediate coordinates from line & col.

    Intermediate coordinates are actually the instruments scanning angles.
    """
    loff, coff = offsets
    lfac, cfac = factors
    x__ = (col - coff) / cfac * 2**16
    y__ = (line - loff) / lfac * 2**16

    return x__, y__


def get_area_extent(self, size, offsets, factors, platform_height):
    """Get the area extent of the file."""
    nlines, ncols = size
    h = platform_height

    # count starts at 1
    cols = 1 - 0.5
    lines = 1 - 0.5
    ll_x, ll_y = self.get_xy_from_linecol(lines, cols, offsets, factors)

    cols += ncols
    lines += nlines
    ur_x, ur_y = self.get_xy_from_linecol(lines, cols, offsets, factors)

    return (np.deg2rad(ll_x) * h, np.deg2rad(ll_y) * h,
            np.deg2rad(ur_x) * h, np.deg2rad(ur_y) * h)


def get_area_def(self, dsid):
    """Get the area definition of the band."""
    cfac = np.int32(self.mda['cfac'])
    lfac = np.int32(self.mda['lfac'])
    coff = np.float32(self.mda['coff'])
    loff = np.float32(self.mda['loff'])

    a = self.mda['projection_parameters']['a']
    b = self.mda['projection_parameters']['b']
    h = self.mda['projection_parameters']['h']
    lon_0 = self.mda['projection_parameters']['SSP_longitude']
    nlines = int(self.mda['number_of_lines'])
    ncols = int(self.mda['number_of_columns'])

    area_extent = self.get_area_extent((nlines, ncols),
                                       (loff, coff),
                                       (lfac, cfac),
                                       h)

    proj_dict = {'a': float(a),
                 'b': float(b),
                 'lon_0': float(lon_0),
                 'h': float(h),
                 'proj': 'geos',
                 'units': 'm'}

    area = geometry.AreaDefinition(
        'some_area_name',
        "On-the-fly area",
        'geosmsg',
        proj_dict,
        ncols,
        nlines,
        area_extent)

    self.area = area
    return area
