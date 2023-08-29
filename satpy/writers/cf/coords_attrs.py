#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Set CF-compliant attributes to x and y spatial dimensions."""

import logging

from satpy.writers.cf.crs import _is_projected

logger = logging.getLogger(__name__)


def add_xy_coords_attrs(dataarray):
    """Add relevant attributes to x, y coordinates."""
    # If there are no coords, return dataarray
    if not dataarray.coords.keys() & {"x", "y", "crs"}:
        return dataarray
    # If projected area
    if _is_projected(dataarray):
        dataarray = _add_xy_projected_coords_attrs(dataarray)
    else:
        dataarray = _add_xy_geographic_coords_attrs(dataarray)
    if 'crs' in dataarray.coords:
        dataarray = dataarray.drop_vars('crs')
    return dataarray


def _add_xy_projected_coords_attrs(dataarray, x='x', y='y'):
    """Add relevant attributes to x, y coordinates of a projected CRS."""
    if x in dataarray.coords:
        dataarray[x].attrs['standard_name'] = 'projection_x_coordinate'
        dataarray[x].attrs['units'] = 'm'
    if y in dataarray.coords:
        dataarray[y].attrs['standard_name'] = 'projection_y_coordinate'
        dataarray[y].attrs['units'] = 'm'
    return dataarray


def _add_xy_geographic_coords_attrs(dataarray, x='x', y='y'):
    """Add relevant attributes to x, y coordinates of a geographic CRS."""
    if x in dataarray.coords:
        dataarray[x].attrs['standard_name'] = 'longitude'
        dataarray[x].attrs['units'] = 'degrees_east'
    if y in dataarray.coords:
        dataarray[y].attrs['standard_name'] = 'latitude'
        dataarray[y].attrs['units'] = 'degrees_north'
    return dataarray
