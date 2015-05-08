
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2014, 2015.
#
# Author(s):
#
#   Panu Lahtinen <panu.lahtinen@fmi.fi>
#   Martin Raspaud <martin.raspaud@smhi.se>
#
# This file is part of mpop.
#
# mpop is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# mpop is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# mpop.  If not, see <http://www.gnu.org/licenses/>.

'''Helper functions for area extent calculations.
'''

import numpy as np
from mpop.projector import get_area_def
# from pyresample.utils import AreaNotFound

import pyresample

import logging
from pyproj import Proj

LOGGER = logging.getLogger(__name__)


def area_def_names_to_extent(area_def_names, proj4_str,
                             default_extent=(-5567248.07, -5570248.48,
                                             5570248.48, 5567248.07)):
    '''Convert a list of *area_def_names* to maximal area extent in destination
    projection defined by *proj4_str*. *default_extent* gives the extreme
    values.  Default value is MSG3 extents at lat0=0.0. If a boundary of one of
    the area_defs is entirely invalid, the *default_extent* is taken.
    '''

    if type(area_def_names) is not list:
        area_def_names = [area_def_names]

    maximum_extent = None

    for name in area_def_names:

        try:
            boundaries = get_area_def(name).get_boundary_lonlats()
        except pyresample.utils.AreaNotFound:
            LOGGER.warning('Area definition not found ' + name)
            continue
        except AttributeError:
            boundaries = name.get_boundary_lonlats()

        if (all(boundaries[0].side1 > 1e20) or
                all(boundaries[0].side2 > 1e20) or
                all(boundaries[0].side3 > 1e20) or
                all(boundaries[0].side4 > 1e20)):
            maximum_extent = list(default_extent)
            continue

        lon_sides = (boundaries[0].side1, boundaries[0].side2,
                     boundaries[0].side3, boundaries[0].side4)
        lat_sides = (boundaries[1].side1, boundaries[1].side2,
                     boundaries[1].side3, boundaries[1].side4)

        maximum_extent = boundaries_to_extent(proj4_str, maximum_extent,
                                              default_extent,
                                              lon_sides, lat_sides)

    maximum_extent[0] -= 10000
    maximum_extent[1] -= 10000
    maximum_extent[2] += 10000
    maximum_extent[3] += 10000

    return maximum_extent


def boundaries_to_extent(proj4_str, maximum_extent, default_extent,
                         lon_sides, lat_sides):
    '''Get area extent from given boundaries.
    '''

    # proj4-ify the projection string
    if '+' not in proj4_str:
        proj4_str = proj4_str.split(' ')
        proj4_str = '+' + ' +'.join(proj4_str)

    pro = Proj(proj4_str)

    # extents for edges
    x_dir, y_dir = pro(np.concatenate(lon_sides),
                       np.concatenate(lat_sides))

    # replace invalid values with NaN
    x_dir[np.abs(x_dir) > 1e20] = np.nan
    y_dir[np.abs(y_dir) > 1e20] = np.nan

    # Get the maximum needed extent from different corners.
    extent = [np.nanmin(x_dir),
              np.nanmin(y_dir),
              np.nanmax(x_dir),
              np.nanmax(y_dir)]

    # Replace "infinity" values with default extent
    for i in range(4):
        if extent[i] is np.nan:
            extent[i] = default_extent[i]

    # update maximum extent
    if maximum_extent is None:
        maximum_extent = extent
    else:
        if maximum_extent[0] > extent[0]:
            maximum_extent[0] = extent[0]
        if maximum_extent[1] > extent[1]:
            maximum_extent[1] = extent[1]
        if maximum_extent[2] < extent[2]:
            maximum_extent[2] = extent[2]
        if maximum_extent[3] < extent[3]:
            maximum_extent[3] = extent[3]

    # Replace "infinity" values with default extent
    for i in range(4):
        if not np.isfinite(maximum_extent[i]):
            maximum_extent[i] = default_extent[i]

    return maximum_extent
