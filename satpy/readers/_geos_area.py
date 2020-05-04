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
"""Geostationary Projection / Area computations.

This module computes properties and area definitions for geostationary
satellites. It is designed to be a common module that can be called by
all geostationary satellite readers and uses commonly-included parameters
such as the CFAC/LFAC values, satellite position, etc, to compute the
correct area definition.
"""

import numpy as np
from pyresample import geometry


def get_xy_from_linecol(line, col, offsets, factors):
    """Get the intermediate coordinates from line & col.

    Intermediate coordinates are actually the instruments scanning angles.
    """
    loff, coff = offsets
    lfac, cfac = factors
    x__ = float(col - coff) / (float(cfac) / 2 ** 16)
    y__ = float(line - loff) / (float(lfac) / 2 ** 16)

    return x__, y__


def make_ext(ll_x, ur_x, ll_y, ur_y, h):
    """Create the area extent from computed ll and ur.

    Args:
        ll_x: The lower left x coordinate (m)
        ur_x: The upper right x coordinate (m)
        ll_y: The lower left y coordinate (m)
        ur_y: The upper right y coordinate (m)
        h: The satellite altitude above the Earth's surface
    Returns:
        aex: An area extent for the scene

    """
    aex = (np.deg2rad(ll_x) * h, np.deg2rad(ll_y) * h,
           np.deg2rad(ur_x) * h, np.deg2rad(ur_y) * h)

    return aex


def get_area_extent(pdict):
    """Get the area extent seen by a geostationary satellite.

    Args:
        pdict: A dictionary containing common parameters:
            nlines: Number of lines in image
            ncols: Number of columns in image
            cfac: Column scaling factor
            lfac: Line scaling factor
            coff: Column offset factor
            loff: Line offset factor
            scandir: 'N2S' for standard (NW->SE), 'S2N' for inverse (SE->NW)
    Returns:
        aex: An area extent for the scene


    The general relation between (line, col) and (y, x) is:

        y = (line - loff) / lfac
        x = (col - coff) / cfac

    There are two common scanning directions:

    1) North-West to South-East:

                         y
                         ^
                         |
               0.5       |
            0.5 ---------|-------------> cols
                |        |         |
                |        |         |
                |        |         |
        ----------------------------------------> x
                |        |         |
                |        |         |
                |        |         |
                |-------------------
                |        |          \
                V        |          image
              lines      |

    Here the "x" and "cols" axes are parallel, but the "y" and "lines" axes
    point into opposite directions. That means cfac must be positive and lfac
    must be negative.

    2) South-East to North-West: In this case just invert the x and y axes. Then
    the "x" and "cols" axes point into opposite directions while the "y" and "lines"
    axes are parallel. That means cfac must be negative and lfac must be positive.

    Note: lfac and cfac in the data might have different signs than what is expected
    here (e.g. both negative for SEVIRI or both positive for AHI). In that case this
    method will change the sign as explained above.

    Once the signs of lfac and cfac have been determined, use the above formulas
    to compute the x/y coordinates at the lower left and upper right corner of the
    image:

        lower left:  (line, col) = (nlines + 0.5, 0.5)
        upper right: (line, col) = (0.5, ncols + 0.5)

    By convention the center of the upper left pixel is (line, col) = (1, 1), so the
    upper left corner is (0.5, 0.5).
    """
    # Set sign of lfac and cfac dependening on the scanning direction
    lfac_sign = -1 if pdict['scandir'] == 'N2S' else 1
    lfac = np.fabs(pdict['lfac']) * lfac_sign
    cfac_sign = 1 if pdict['scandir'] == 'N2S' else -1
    cfac = np.fabs(pdict['cfac']) * cfac_sign

    factors = (lfac, cfac)
    offsets = (pdict['loff'], pdict['coff'])

    # Lower left corner of the image
    col = 0.5
    line = pdict['nlines'] + 0.5
    ll_x, ll_y = get_xy_from_linecol(line, col, offsets, factors)

    # Upper right corner of the image
    col = pdict['ncols'] + 0.5
    line = 0.5
    ur_x, ur_y = get_xy_from_linecol(line, col, offsets, factors)

    return make_ext(ll_x=ll_x, ll_y=ll_y, ur_x=ur_x, ur_y=ur_y, h=pdict['h'])


def get_area_definition(pdict, a_ext):
    """Get the area definition for a geo-sat.

    Args:
        pdict: A dictionary containing common parameters:
            nlines: Number of lines in image
            ncols: Number of columns in image
            ssp_lon: Subsatellite point longitude (deg)
            a: Earth equatorial radius (m)
            b: Earth polar radius (m)
            h: Platform height (m)
            a_name: Area name
            a_desc: Area description
            p_id: Projection id
        a_ext: A four element tuple containing the area extent (scan angle)
               for the scene in radians
    Returns:
        a_def: An area definition for the scene

    """
    proj_dict = {'a': float(pdict['a']),
                 'b': float(pdict['b']),
                 'lon_0': float(pdict['ssp_lon']),
                 'h': float(pdict['h']),
                 'proj': 'geos',
                 'units': 'm'}

    a_def = geometry.AreaDefinition(
        pdict['a_name'],
        pdict['a_desc'],
        pdict['p_id'],
        proj_dict,
        int(pdict['ncols']),
        int(pdict['nlines']),
        a_ext)

    return a_def
