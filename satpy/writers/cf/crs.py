#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CRS utility."""

import logging
from contextlib import suppress

from pyresample.geometry import AreaDefinition, SwathDefinition

logger = logging.getLogger(__name__)


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
