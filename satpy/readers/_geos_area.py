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
            scandir: 'N2S' for standard (N->S), 'S2N' for inverse (S->N)
    Returns:
        aex: An area extent for the scene

    """
    # count starts at 1
    cols = 1 - 0.5

    if pdict['scandir'] == 'S2N':
        lines = 0.5 - 1
        scanmult = -1
    else:
        lines = 1 - 0.5
        scanmult = 1
    # Lower left x, y scanning angles in degrees
    ll_x, ll_y = get_xy_from_linecol(lines * scanmult,
                                     cols,
                                     (pdict['loff'], pdict['coff']),
                                     (pdict['lfac'], pdict['cfac']))

    cols += pdict['ncols']
    lines += pdict['nlines']
    # Upper right x, y scanning angles in degrees
    ur_x, ur_y = get_xy_from_linecol(lines * scanmult,
                                     cols,
                                     (pdict['loff'], pdict['coff']),
                                     (pdict['lfac'], pdict['cfac']))
    if pdict['scandir'] == 'S2N':
        ll_y *= -1
        ur_y *= -1

    # Convert degrees to radians and create area extent
    aex = make_ext(ll_x=ll_x, ur_x=ur_x, ll_y=ll_y, ur_y=ur_y, h=pdict['h'])

    return aex


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
