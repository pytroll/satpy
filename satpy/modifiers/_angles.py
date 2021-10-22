#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2021 Satpy developers
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
"""Utilties for getting various angles for a dataset.."""

import dask.array as da
import numpy as np
import xarray as xr
from satpy.utils import get_satpos
from pyorbital.orbital import get_observer_look
from pyorbital.astronomy import get_alt_az, sun_zenith_angle


def get_angles(vis):
    """Get sun and satellite angles to use in crefl calculations."""
    lons, lats = _get_valid_lonlats(vis)
    sun_angles = _get_sun_angles(vis, lons, lats)
    sat_angles = _get_sensor_angles(vis, lons, lats)
    # sata, satz, suna, sunz
    return sat_angles + sun_angles


def get_satellite_zenith_angle(data_arr: xr.DataArray) -> xr.DataArray:
    """Generate satellite zenith angle for the provided data."""
    lons, lats = _get_valid_lonlats(data_arr)
    satz = _get_sensor_angles(data_arr, lons, lats)[1]
    return satz


def _get_valid_lonlats(vis):
    lons, lats = vis.attrs['area'].get_lonlats(chunks=vis.data.chunks)
    lons = da.where(lons >= 1e30, np.nan, lons)
    lats = da.where(lats >= 1e30, np.nan, lats)
    return lons, lats


def _get_sun_angles(vis, lons, lats):
    suna = get_alt_az(vis.attrs['start_time'], lons, lats)[1]
    suna = np.rad2deg(suna)
    sunz = sun_zenith_angle(vis.attrs['start_time'], lons, lats)
    return suna, sunz


def _get_sensor_angles(vis, lons, lats):
    sat_lon, sat_lat, sat_alt = get_satpos(vis)
    sata, satel = get_observer_look(
        sat_lon,
        sat_lat,
        sat_alt / 1000.0,  # km
        vis.attrs['start_time'],
        lons, lats, 0)
    satz = 90 - satel
    return sata, satz
