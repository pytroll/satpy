#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Set CF-compliant spatial and temporal coordinates."""

import logging
from contextlib import suppress

import numpy as np
import xarray as xr
from pyresample.geometry import AreaDefinition, SwathDefinition

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


def _is_projected(dataarray):
    """Guess whether data are projected or not."""
    crs = _try_to_get_crs(dataarray)
    if crs:
        return crs.is_projected
    units = _try_get_units_from_coords(dataarray)
    if units:
        if units.endswith("m"):
            return True
        if units.startswith("degrees"):
            return False
    logger.warning("Failed to tell if data are projected. Assuming yes.")
    return True


def _try_to_get_crs(dataarray):
    """Try to get a CRS from attributes."""
    if "area" in dataarray.attrs:
        if isinstance(dataarray.attrs["area"], AreaDefinition):
            return dataarray.attrs["area"].crs
        if not isinstance(dataarray.attrs["area"], SwathDefinition):
            logger.warning(
                f"Could not tell CRS from area of type {type(dataarray.attrs['area']).__name__:s}. "
                "Assuming projected CRS.")
    if "crs" in dataarray.coords:
        return dataarray.coords["crs"].item()


def _try_get_units_from_coords(dataarray):
    """Try to retrieve coordinate x/y units."""
    for c in ["x", "y"]:
        with suppress(KeyError):
            # If the data has only 1 dimension, it has only one of x or y coords
            if "units" in dataarray.coords[c].attrs:
                return dataarray.coords[c].attrs["units"]


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


def add_time_bounds_dimension(ds, time="time"):
    """Add time bound dimension to xr.Dataset."""
    start_times = []
    end_times = []
    for _var_name, data_array in ds.items():
        start_times.append(data_array.attrs.get("start_time", None))
        end_times.append(data_array.attrs.get("end_time", None))

    start_time = min(start_time for start_time in start_times
                     if start_time is not None)
    end_time = min(end_time for end_time in end_times
                   if end_time is not None)
    ds['time_bnds'] = xr.DataArray([[np.datetime64(start_time),
                                     np.datetime64(end_time)]],
                                   dims=['time', 'bnds_1d'])
    ds[time].attrs['bounds'] = "time_bnds"
    ds[time].attrs['standard_name'] = "time"
    return ds


def process_time_coord(dataarray, epoch):
    """Process the 'time' coordinate, if existing.

    It expand the DataArray with a time dimension if does not yet exists.

    The function assumes

        - that x and y dimensions have at least shape > 1
        - the time coordinate has size 1

    """
    if 'time' in dataarray.coords:
        dataarray['time'].encoding['units'] = epoch
        dataarray['time'].attrs['standard_name'] = 'time'
        dataarray['time'].attrs.pop('bounds', None)

        if 'time' not in dataarray.dims and dataarray["time"].size not in dataarray.shape:
            dataarray = dataarray.expand_dims('time')

    return dataarray
