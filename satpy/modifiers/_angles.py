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
from datetime import datetime
from functools import update_wrapper
from glob import glob
from typing import Callable, Any, Optional, Union
import shutil

import dask
import dask.array as da
import numpy as np
import xarray as xr
from satpy.utils import get_satpos, ignore_invalid_float_warnings
from pyorbital.orbital import get_observer_look
from pyorbital.astronomy import get_alt_az, sun_zenith_angle
from pyresample.geometry import SwathDefinition, AreaDefinition

import satpy

PRGeometry = Union[SwathDefinition, AreaDefinition]

# Arbitrary time used when computing sensor angles that is passed to
# pyorbital's get_observer_look function.
# The difference is on the order of 1e-10 at most as time changes so we force
# it to a single time for easier caching. It is *only* used if caching.
STATIC_EARTH_INERTIAL_DATETIME = datetime(2020, 1, 1, 0, 0, 0)
# simple integer for future proofing function changes that would invalidate
# existing files in a user's cache
_CACHE_VERSION = 1


class ZarrCacheHelper:
    """Helper for caching function results to on-disk zarr arrays."""

    def __init__(self,
                 func: Callable,
                 cache_config_key: str,
                 uncacheable_arg_types=(SwathDefinition, xr.DataArray, da.Array),
                 sanitize_args_func: Callable = None,
                 ):
        """Hold on to provided arguments for future use."""
        self._func = func
        self._cache_config_key = cache_config_key
        self._uncacheable_arg_types = uncacheable_arg_types
        self._sanitize_args_func = sanitize_args_func

    def cache_clear(self, cache_dir: Optional[str] = None):
        """Remove all on-disk files associated with this function.

        Intended to mimic the :func:`functools.cache` behavior.
        """
        if cache_dir is None:
            cache_dir = satpy.config.get("cache_dir")
        if cache_dir is None:
            raise RuntimeError("No 'cache_dir' configured.")
        zarr_pattern = self._zarr_pattern("*", cache_version="*").format("*")
        for zarr_dir in glob(os.path.join(cache_dir, zarr_pattern)):
            try:
                shutil.rmtree(zarr_dir)
            except OSError:
                continue

    def _zarr_pattern(self, arg_hash, cache_version: Union[int, str] = _CACHE_VERSION) -> str:
        return f"{self._func.__name__}_v{cache_version}" + "_{}_" + f"{arg_hash}.zarr"

    def __call__(self, *args, cache_dir: Optional[str] = None) -> Any:
        """Call the decorated function."""
        new_args = self._sanitize_args_func(*args) if self._sanitize_args_func is not None else args
        arg_hash = _hash_args(*new_args)
        should_cache, cache_dir = self._get_should_cache_and_cache_dir(new_args, cache_dir)
        zarr_fn = self._zarr_pattern(arg_hash)
        zarr_format = os.path.join(cache_dir, zarr_fn)
        zarr_paths = glob(zarr_format.format("*"))
        if not should_cache or not zarr_paths:
            # use sanitized arguments if we are caching, otherwise use original arguments
            args = new_args if should_cache else args
            res = self._func(*args)
            if should_cache and not zarr_paths:
                self._cache_results(res, zarr_format)
        # if we did any caching, let's load from the zarr files
        if should_cache:
            # re-calculate the cached paths
            zarr_paths = glob(zarr_format.format("*"))
            if not zarr_paths:
                raise RuntimeError("Data was cached to disk but no files were found")
            res = tuple(da.from_zarr(zarr_path) for zarr_path in zarr_paths)
        return res

    def _get_should_cache_and_cache_dir(self, args, cache_dir):
        should_cache = satpy.config.get(self._cache_config_key, False)
        can_cache = not any(isinstance(arg, self._uncacheable_arg_types) for arg in args)
        should_cache = should_cache and can_cache
        if cache_dir is None:
            cache_dir = satpy.config.get("cache_dir")
        if cache_dir is None:
            should_cache = False
        return should_cache, cache_dir

    def _cache_results(self, res, zarr_format):
        os.makedirs(os.path.dirname(zarr_format), exist_ok=True)
        new_res = []
        for idx, sub_res in enumerate(res):
            zarr_path = zarr_format.format(idx)
            # See https://github.com/dask/dask/issues/8380
            with dask.config.set({"optimization.fuse.active": False}):
                new_sub_res = sub_res.to_zarr(zarr_path,
                                              return_stored=True,
                                              compute=False)
            new_res.append(new_sub_res)
        # actually compute the storage to zarr
        da.compute(new_res)


def cache_to_zarr(
        cache_config_key: str,
        uncacheable_arg_types=(SwathDefinition, xr.DataArray, da.Array),
        sanitize_args_func: Callable = None,
) -> Callable:
    """Decorate a function and cache the results as a zarr array on disk.

    Note: Only the dask array is cached. Currently the metadata and coordinate
    information is not stored.

    """
    # TODO: Consider cache management (LRU)
    def _decorator(func: Callable) -> Callable:
        zarr_cacher = ZarrCacheHelper(func,
                                      cache_config_key,
                                      uncacheable_arg_types,
                                      sanitize_args_func)
        wrapper = update_wrapper(zarr_cacher, func)
        return wrapper
    return _decorator


def _hash_args(*args):
    import json
    hashable_args = []
    for arg in args:
        if isinstance(arg, (xr.DataArray, da.Array, SwathDefinition)):
            continue
        if isinstance(arg, AreaDefinition):
            arg = hash(arg)
        elif isinstance(arg, datetime):
            arg = arg.isoformat(" ")
        hashable_args.append(arg)
    arg_hash = hashlib.sha1()
    arg_hash.update(json.dumps(tuple(hashable_args)).encode('utf8'))
    return arg_hash.hexdigest()


def _sanitize_observer_look_args(*args):
    new_args = []
    for arg in args:
        if isinstance(arg, datetime):
            new_args.append(STATIC_EARTH_INERTIAL_DATETIME)
        elif isinstance(arg, (float, np.float64, np.float32)):
            # round floating point numbers to nearest tenth
            new_args.append(round(arg, 1))
        else:
            new_args.append(arg)
    return new_args


def get_angles(data_arr: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Get sun and satellite angles to use in crefl calculations."""
    sun_angles = _get_sun_angles(data_arr)
    sat_angles = _get_sensor_angles(data_arr)
    # sata, satz, suna, sunz
    # FIXME: These are actually dask arrays so...?
    return sat_angles + sun_angles


def get_satellite_zenith_angle(data_arr: xr.DataArray) -> xr.DataArray:
    """Generate satellite zenith angle for the provided data."""
    satz = _get_sensor_angles(data_arr)[1]
    return satz


@cache_to_zarr("cache_lonlats")
def _get_valid_lonlats(area: PRGeometry, chunks: int = "auto") -> tuple[da.Array, da.Array]:
    with ignore_invalid_float_warnings():
        lons, lats = area.get_lonlats(chunks=chunks)
        lons = da.where(lons >= 1e30, np.nan, lons)
        lats = da.where(lats >= 1e30, np.nan, lats)
    return lons, lats


def _get_sun_angles(data_arr: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    lons, lats = _get_valid_lonlats(data_arr.attrs["area"], data_arr.data.chunks)
    with ignore_invalid_float_warnings():
        suna = get_alt_az(data_arr.attrs['start_time'], lons, lats)[1]
        suna = np.rad2deg(suna)
        sunz = sun_zenith_angle(data_arr.attrs['start_time'], lons, lats)
    return suna, sunz


def _get_sensor_angles(data_arr: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    sat_lon, sat_lat, sat_alt = get_satpos(data_arr)
    area_def = data_arr.attrs["area"]
    return _get_sensor_angles_from_sat_pos(sat_lon, sat_lat, sat_alt,
                                           data_arr.attrs["start_time"],
                                           area_def, data_arr.data.chunks)


@cache_to_zarr("cache_sensor_angles", sanitize_args_func=_sanitize_observer_look_args)
def _get_sensor_angles_from_sat_pos(sat_lon, sat_lat, sat_alt, start_time, area_def, chunks):
    lons, lats = _get_valid_lonlats(area_def, chunks)
    res = da.map_blocks(_get_sensor_angles_wrapper, lons, lats, start_time, sat_lon, sat_lat, sat_alt,
                        dtype=lons.dtype, meta=np.array((), dtype=lons.dtype), new_axis=[0],
                        chunks=(2,) + lons.chunks)
    return res[0], res[1]


def _get_sensor_angles_wrapper(lons, lats, start_time, sat_lon, sat_lat, sat_alt):
    with ignore_invalid_float_warnings():
        sata, satel = get_observer_look(
            sat_lon,
            sat_lat,
            sat_alt / 1000.0,  # km
            start_time,
            lons, lats, 0)
        satz = 90 - satel
        return np.stack([sata, satz])
