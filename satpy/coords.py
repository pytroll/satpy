#!/usr/bin/env python
# Copyright (c) 2015-2025 Satpy developers
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
"""Utility functions for coordinates."""

import logging

import xarray as xr

LOG = logging.getLogger(__name__)


def add_xy_coords(data_arr, area, crs=None):
    """Assign x/y coordinates to DataArray from provided area.

    If 'x' and 'y' coordinates already exist then they will not be added.

    Args:
        data_arr (xarray.DataArray): data object to add x/y coordinates to
        area (pyresample.geometry.AreaDefinition): area providing the
            coordinate data.
        crs (pyproj.crs.CRS or None): CRS providing additional information
            about the area's coordinate reference system if available.
            Requires pyproj 2.0+.

    Returns (xarray.DataArray): Updated DataArray object

    """
    if "x" in data_arr.coords and "y" in data_arr.coords:
        # x/y coords already provided
        return data_arr
    if "x" not in data_arr.dims or "y" not in data_arr.dims:
        # no defined x and y dimensions
        return data_arr
    if not hasattr(area, "get_proj_vectors"):
        return data_arr
    x, y = area.get_proj_vectors()

    # convert to DataArrays
    y_attrs = {}
    x_attrs = {}

    _check_crs_units(crs, x_attrs, y_attrs)

    y = xr.DataArray(y, dims=("y",), attrs=y_attrs)
    x = xr.DataArray(x, dims=("x",), attrs=x_attrs)
    return data_arr.assign_coords(y=y, x=x)


def _check_crs_units(crs, x_attrs, y_attrs):
    if crs is None:
        return
    units = crs.axis_info[0].unit_name
    # fix udunits/CF standard units
    units = units.replace("metre", "meter")
    if units == "degree":
        y_attrs["units"] = "degrees_north"
        x_attrs["units"] = "degrees_east"
    else:
        y_attrs["units"] = units
        x_attrs["units"] = units


def add_crs_xy_coords(data_arr, area):
    """Add :class:`pyproj.crs.CRS` and x/y or lons/lats to coordinates.

    For SwathDefinition or GridDefinition areas this will add a
    `crs` coordinate and coordinates for the 2D arrays of `lons` and `lats`.

    For AreaDefinition areas this will add a `crs` coordinate and the
    1-dimensional `x` and `y` coordinate variables.

    Args:
        data_arr (xarray.DataArray): DataArray to add the 'crs'
            coordinate.
        area (pyresample.geometry.AreaDefinition): Area to get CRS
            information from.

    """
    from pyresample.geometry import SwathDefinition

    crs, data_arr = _add_crs(area, data_arr)

    # Add x/y coordinates if possible
    if isinstance(area, SwathDefinition):
        # add lon/lat arrays for swath definitions
        # SwathDefinitions created by Satpy should be assigning DataArray
        # objects as the lons/lats attributes so use those directly to
        # maintain original .attrs metadata (instead of converting to dask
        # array).
        lons = area.lons
        lats = area.lats
        lons.attrs.setdefault("standard_name", "longitude")
        lons.attrs.setdefault("long_name", "longitude")
        lons.attrs.setdefault("units", "degrees_east")
        lats.attrs.setdefault("standard_name", "latitude")
        lats.attrs.setdefault("long_name", "latitude")
        lats.attrs.setdefault("units", "degrees_north")
        # See https://github.com/pydata/xarray/issues/3068
        # data_arr = data_arr.assign_coords(longitude=lons, latitude=lats)
    else:
        # Gridded data (AreaDefinition/StackedAreaDefinition)
        data_arr = add_xy_coords(data_arr, area, crs=crs)
    return data_arr


def _add_crs(area, data_arr):
    from pyproj import CRS

    if hasattr(area, "crs"):
        crs = area.crs
    else:
        # default lat/lon projection
        latlon_proj = "+proj=latlong +datum=WGS84 +ellps=WGS84"
        proj_str = getattr(area, "proj_str", latlon_proj)
        crs = CRS.from_string(proj_str)
    data_arr = data_arr.assign_coords(crs=crs)
    return crs, data_arr
