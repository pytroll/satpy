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

    .. note::

        The AreaDefinition `proj_id` attribute is being deprecated.

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


def sampling_to_lfac_cfac(sampling):
    """Convert angular sampling to line/column scaling factor (aka LFAC/CFAC).

    Reference: `MSG Ground Segment LRIT HRIT Mission Specific Implementation`_,
    Appendix E.2.


    .. _MSG Ground Segment LRIT HRIT Mission Specific Implementation:
    https://www-cdn.eumetsat.int/files/2020-04/pdf_ten_05057_spe_msg_lrit_hri.pdf

    Args:
        sampling: float
            Angular sampling (rad)

    Returns:
        Line/column scaling factor (deg-1)
    """
    return 2.0 ** 16 / np.rad2deg(sampling)


def get_geos_area_naming(input_dict):
    """Get a dictionary containing formatted AreaDefinition naming.

    Args:
        input_dict: dict
            Dictionary with keys `platform_name`, `instrument_name`, `service_name`, `service_desc`, `resolution` .
            The resolution is expected in meters.
    Returns:
        area_naming_dict with `area_id`, `description`  keys, values are strings.

    .. note::

        The AreaDefinition `proj_id` attribute is being deprecated and is therefore not formatted here.
        An empty string is to be used until the attribute is fully removed.

    """
    area_naming_dict = {}

    resolution_strings = get_resolution_and_unit_strings(input_dict['resolution'])

    area_naming_dict['area_id'] = '{}_{}_{}_{}{}'.format(input_dict['platform_name'].lower(),
                                                         input_dict['instrument_name'].lower(),
                                                         input_dict['service_name'].lower(),
                                                         resolution_strings['value'],
                                                         resolution_strings['unit']
                                                         )

    area_naming_dict['description'] = '{} {} {} area definition ' \
                                      'with {} {} resolution'.format(input_dict['platform_name'].upper(),
                                                                     input_dict['instrument_name'].upper(),
                                                                     input_dict['service_desc'],
                                                                     resolution_strings['value'],
                                                                     resolution_strings['unit']
                                                                     )

    return area_naming_dict


def get_resolution_and_unit_strings(resolution):
    """Get the resolution value and unit as strings.

    If the resolution is larger than 1000 m, use kilometer as unit. If lower, use meter.

    Args:
        resolution: scalar
            Resolution in meters.

    Returns:
        Dictionary with `value` and `unit` keys, values are strings.
    """
    if resolution >= 1000:
        return {'value': '{:.0f}'.format(resolution*1e-3),
                'unit': 'km'}

    return {'value': '{:.0f}'.format(resolution),
            'unit': 'm'}
