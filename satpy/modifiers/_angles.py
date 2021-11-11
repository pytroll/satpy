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
from __future__ import annotations

import os
import hashlib
from functools import update_wrapper
from typing import Callable, Any, Optional

import dask.array as da
import numpy as np
import xarray as xr
from satpy.utils import get_satpos
from pyorbital.orbital import get_observer_look
from pyorbital.astronomy import get_alt_az, sun_zenith_angle

import satpy


def cache_to_zarr(num_results: int = 1) -> Callable:
    """Decorate a function and cache the results as a zarr array on disk.

    Note: Only the dask array is cached. Currently the metadata and coordinate
    information is not stored.

    """
    # TODO: Remove num_results, just determine it when it is returned
    # TODO: Use satpy.config to get the default behavior
    # TODO: Use dask to_zarr with return_stored=True
    def _decorator(func: Callable) -> Callable:
        def _wrapper(data_arr: xr.DataArray, *args, cache_dir: Optional[str] = None) -> Any:
            arg_hash = _hash_args(*args)
            should_cache = satpy.config.get("cache_angles", False)
            if cache_dir is None:
                cache_dir = satpy.config.get("cache_dir")
            if cache_dir is None:
                should_cache = False
            zarr_fn = f"{func.__name__}_{arg_hash}" + "{}.zarr"
            # XXX: Can we use zarr groups to save us from having multiple zarr arrays
            zarr_paths = [os.path.join(cache_dir, zarr_fn).format(result_idx) for result_idx in range(num_results)]
            if not should_cache or not os.path.exists(zarr_paths[0]):
                res = func(data_arr, *args)
            elif should_cache and os.path.exists(zarr_paths[0]):
                res = tuple(da.from_zarr(zarr_path) for zarr_path in zarr_paths)
            if should_cache and not os.path.exists(zarr_paths[0]):
                os.makedirs(cache_dir, exist_ok=True)
                new_res = []
                for sub_res, zarr_path in zip(res, zarr_paths):
                    new_sub_res = sub_res.to_zarr(zarr_path,
                                                  return_stored=True,
                                                  compute=False)
                    new_res.append(new_sub_res)
                # actually compute the storage to zarr
                # this returns the numpy arrays
                # FIXME: Find a way to compute all the zarrs at the same time, but still work from that zarr array
                da.compute(new_res)
                res = tuple(new_res)
            return res
        wrapper = update_wrapper(_wrapper, func)
        return wrapper
    return _decorator


def _hash_args(*args):
    import json
    hashable_args = []
    for arg in args:
        if not isinstance(arg, (xr.DataArray, da.Array)):
            hashable_args.append(arg)
            continue
        # TODO: Be smarter
        hashables = [arg.shape]
        if isinstance(arg, xr.DataArray):
            hashables.append(arg.attrs.get("start_time"))
            hashables.append(arg.dims)

        hashable_args.append(tuple(hashables))
    hash = hashlib.sha1()
    hash.update(json.dumps(tuple(hashable_args)).encode('utf8'))
    return hash.hexdigest()


def get_angles(data_arr: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Get sun and satellite angles to use in crefl calculations."""
    # TODO: Restructure so lonlats are loaded in each function
    # and that only start_time or area are passed between instead of the whole DataArray
    lons, lats = _get_valid_lonlats(data_arr)
    sun_angles = _get_sun_angles(data_arr, lons, lats)
    sat_angles = _get_sensor_angles(data_arr, lons, lats)
    # sata, satz, suna, sunz
    return sat_angles + sun_angles


def get_satellite_zenith_angle(data_arr: xr.DataArray) -> xr.DataArray:
    """Generate satellite zenith angle for the provided data."""
    lons, lats = _get_valid_lonlats(data_arr)
    satz = _get_sensor_angles(data_arr, lons, lats)[1]
    return satz


def _get_valid_lonlats(data_arr: xr.DataArray) -> tuple[da.Array, da.Array]:
    lons, lats = data_arr.attrs['area'].get_lonlats(chunks=data_arr.data.chunks)
    lons = da.where(lons >= 1e30, np.nan, lons)
    lats = da.where(lats >= 1e30, np.nan, lats)
    return lons, lats


def _get_sun_angles(data_arr: xr.DataArray, lons: da.Array, lats: da.Array) -> tuple[xr.DataArray, xr.DataArray]:
    suna = get_alt_az(data_arr.attrs['start_time'], lons, lats)[1]
    suna = np.rad2deg(suna)
    sunz = sun_zenith_angle(data_arr.attrs['start_time'], lons, lats)
    return suna, sunz


@cache_to_zarr(num_results=2)
def _get_sensor_angles(data_arr: xr.DataArray, lons: da.Array, lats: da.Array) -> tuple[xr.DataArray, xr.DataArray]:
    sat_lon, sat_lat, sat_alt = get_satpos(data_arr)
    sata, satel = get_observer_look(
        sat_lon,
        sat_lat,
        sat_alt / 1000.0,  # km
        data_arr.attrs['start_time'],
        lons, lats, 0)
    satz = 90 - satel
    return sata, satz
