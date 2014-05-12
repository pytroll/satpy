
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2014.
# 
# Author(s):
#  
#   Panu Lahtinen <panu.lahtinen@fmi.fi>
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
from pyresample.geometry import Boundary
# from pyresample.utils import AreaNotFound

import pyresample

import logging
from pyproj import Proj

LOGGER = logging.getLogger(__name__)


def area_def_names_to_extent(area_def_names, proj4_str,
                             default_extent=(-5567248.07, -5570248.48,
                                              5570248.48, 5567248.07)):
    '''Convert a list of *area_def_names* to maximal area extent in
    destination projection defined by *proj4_str*. *default_extent*
    gives the extreme values.  Default value is MSG3 extents at
    lat0=0.0.
    '''

    if type(area_def_names) is not list:
        area_def_names = [area_def_names]

    # proj4-ify the projection string
    if '+' not in proj4_str:
        global_proj4_str = proj4_str.split(' ')
        global_proj4_str = '+' + ' +'.join(global_proj4_str)

    pro = Proj(global_proj4_str)

    maximum_area_extent = None

    for name in area_def_names:

        try:
            boundaries = get_area_def(name).get_boundary_lonlats()
        except pyresample.utils.AreaNotFound:
            LOGGER.warning('Area definition not found ' + name)
            continue
        except AttributeError:
            boundaries = name.get_boundary_lonlats()

        # extents for edges
        _, up_y = pro(boundaries[0].side1, boundaries[1].side1)
        right_x, _ = pro(boundaries[0].side2, boundaries[1].side2)
        _, down_y = pro(boundaries[0].side3, boundaries[1].side3)
        left_x, _ = pro(boundaries[0].side4, boundaries[1].side4)

        # replace invalid values with NaN
        up_y[np.abs(up_y) > 1e20] = np.nan
        right_x[np.abs(right_x) > 1e20] = np.nan
        down_y[np.abs(down_y) > 1e20] = np.nan
        left_x[np.abs(left_x) > 1e20] = np.nan

        # Get the maximum needed extent from different corners.
        extent = [np.nanmin(left_x),
                  np.nanmin(down_y),
                  np.nanmax(right_x),
                  np.nanmax(up_y)]

        # Replace "infinity" values with default extent
        for i in range(4):
            if extent[i] is np.nan:
                extent[i] = default_extent[i]

        # update maximum extent
        if maximum_area_extent is None:
            maximum_area_extent = extent
        else:
            if maximum_area_extent[0] > extent[0]:
                maximum_area_extent[0] = extent[0]
            if maximum_area_extent[1] > extent[1]:
                maximum_area_extent[1] = extent[1]
            if maximum_area_extent[2] < extent[2]:
                maximum_area_extent[2] = extent[2]
            if maximum_area_extent[3] < extent[3]:
                maximum_area_extent[3] = extent[3]

        # Replace "infinity" values with default extent
        for i in range(4):
            if not np.isfinite(maximum_area_extent[i]):
                maximum_area_extent[i] = default_extent[i]


    return maximum_area_extent

