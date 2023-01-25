#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2021-2023 Satpy developers
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

import hashlib
import os
import shutil
import warnings
from datetime import datetime
from functools import update_wrapper
from glob import glob
from typing import Any, Callable, Optional, Union

import dask
import numpy as np
import xarray as xr
from dask import array as da
from pyorbital.astronomy import cos_zen as pyob_cos_zen
from pyorbital.astronomy import get_alt_az
from pyorbital.orbital import get_observer_look
from pyresample.geometry import AreaDefinition, StackedAreaDefinition, SwathDefinition

import satpy
from satpy.utils import PerformanceWarning, get_satpos, ignore_invalid_float_warnings

PRGeometry = Union[SwathDefinition, AreaDefinition, StackedAreaDefinition]

# Arbitrary time used when computing sensor angles that is passed to
# pyorbital's get_observer_look function.
# The difference is on the order of 1e-10 at most as time changes so we force
# it to a single time for easier caching. It is *only* used if caching.
STATIC_EARTH_INERTIAL_DATETIME = datetime(2000, 1, 1, 12, 0, 0)
DEFAULT_UNCACHE_TYPES = (SwathDefinition, xr.DataArray, da.Array)
HASHABLE_GEOMETRIES = (AreaDefinition, StackedAreaDefinition)


class ZarrCacheHelper:
    """Helper for caching function results to on-disk zarr arrays.

    It is recommended to use this class through the :func:`cache_to_zarr_if`
    decorator rather than using it directly.

    Currently the cache does not perform any limiting or removal of cache
    content. That is left up to the user to manage. Caching is based on
    arguments passed to the decorated function but will only be performed
    if the arguments are of a certain type (see ``uncacheable_arg_types``).
    The cache value to use is purely based on the hash value of all of the
    provided arguments along with the "cache version" (see below).

    Note that the zarr format requires regular chunking of data. That is,
    chunks must be all the same size per dimension except for the last chunk.
    To work around this limitation, this class will determine a good regular
    chunking based on the existing chunking scheme, rechunk the input
    arguments, and then rechunk the results before returning them to the user.
    This rechunking is only done if caching is enabled.

    Args:
        func: Function that will be called to generate the value to cache.
        cache_config_key: Name of the boolean ``satpy.config`` parameter to
            use to determine if caching should be done.
        uncacheable_arg_types: Types that if present in the passed arguments
            should trigger caching to *not* happen. By default this includes
            ``SwathDefinition``, ``xr.DataArray``, and ``da.Array`` objects.
        sanitize_args_func: Optional function to call to sanitize provided
            arguments before they are considered for caching. This can be used
            to make arguments more "cacheable" by replacing them with similar
            values that will result in more cache hits. Note that the sanitized
            arguments are only passed to the underlying function if caching
            will be performed, otherwise the original arguments are passed.
        cache_version: Version number used to distinguish one version of a
            decorated function from future versions.

    Notes:
        * Caching only supports dask array values.

        * This helper allows for an additional ``cache_dir`` parameter to
          override the use of the ``satpy.config`` ``cache_dir`` parameter.

    Examples:
        To use through the :func:`cache_to_zarr_if` decorator::

            @cache_to_zarr_if("cache_my_stuff")
            def generate_my_stuff(area_def: AreaDefinition, some_factor: int) -> da.Array:
                # Generate
                return my_dask_arr

        To use the decorated function::

            with satpy.config.set(cache_my_stuff=True):
                my_stuff = generate_my_stuff(area_def, 5)

    """

    def __init__(self,
                 func: Callable,
                 cache_config_key: str,
                 uncacheable_arg_types=DEFAULT_UNCACHE_TYPES,
                 sanitize_args_func: Optional[Callable] = None,
                 cache_version: int = 1,
                 ):
        """Hold on to provided arguments for future use."""
        self._func = func
        self._cache_config_key = cache_config_key
        self._uncacheable_arg_types = uncacheable_arg_types
        self._sanitize_args_func = sanitize_args_func
        self._cache_version = cache_version

    def cache_clear(self, cache_dir: Optional[str] = None):
        """Remove all on-disk files associated with this function.

        Intended to mimic the :func:`functools.cache` behavior.
        """
        cache_dir = self._get_cache_dir_from_config(cache_dir)
        zarr_pattern = self._zarr_pattern("*", cache_version="*").format("*")
        for zarr_dir in glob(os.path.join(cache_dir, zarr_pattern)):
            shutil.rmtree(zarr_dir, ignore_errors=True)

    def _zarr_pattern(self, arg_hash, cache_version: Union[None, int, str] = None) -> str:
        if cache_version is None:
            cache_version = self._cache_version
        return f"{self._func.__name__}_v{cache_version}" + "_{}_" + f"{arg_hash}.zarr"

    def __call__(self, *args, cache_dir: Optional[str] = None) -> Any:
        """Call the decorated function."""
        new_args = self._sanitize_args_func(*args) if self._sanitize_args_func is not None else args
        arg_hash = _hash_args(*new_args, unhashable_types=self._uncacheable_arg_types)
        should_cache, cache_dir = self._get_should_cache_and_cache_dir(new_args, cache_dir)
        zarr_fn = self._zarr_pattern(arg_hash)
        zarr_format = os.path.join(cache_dir, zarr_fn)
        zarr_paths = glob(zarr_format.format("*"))
        if not should_cache or not zarr_paths:
            # use sanitized arguments if we are caching, otherwise use original arguments
            args_to_use = new_args if should_cache else args
            res = self._func(*args_to_use)
            if should_cache and not zarr_paths:
                self._warn_if_irregular_input_chunks(args, args_to_use)
                self._cache_results(res, zarr_format)
        # if we did any caching, let's load from the zarr files
        if should_cache:
            # re-calculate the cached paths
            zarr_paths = sorted(glob(zarr_format.format("*")))
            if not zarr_paths:
                raise RuntimeError("Data was cached to disk but no files were found")
            new_chunks = _get_output_chunks_from_func_arguments(args)
            res = tuple(da.from_zarr(zarr_path, chunks=new_chunks) for zarr_path in zarr_paths)
        return res

    def _get_should_cache_and_cache_dir(self, args, cache_dir: Optional[str]) -> tuple[bool, str]:
        should_cache: bool = satpy.config.get(self._cache_config_key, False)
        can_cache = not any(isinstance(arg, self._uncacheable_arg_types) for arg in args)
        should_cache = should_cache and can_cache
        cache_dir = self._get_cache_dir_from_config(cache_dir)
        return should_cache, cache_dir

    @staticmethod
    def _get_cache_dir_from_config(cache_dir: Optional[str]) -> str:
        cache_dir = cache_dir or satpy.config.get("cache_dir")
        if cache_dir is None:
            raise RuntimeError("Can't use zarr caching. No 'cache_dir' configured.")
        return cache_dir

    @staticmethod
    def _warn_if_irregular_input_chunks(args, modified_args):
        arg_chunks = _get_output_chunks_from_func_arguments(args)
        new_chunks = _get_output_chunks_from_func_arguments(modified_args)
        if _chunks_are_irregular(arg_chunks):
            warnings.warn(
                "Calling cached function with irregular dask chunks. The data "
                "has been rechunked for caching, but this is not optimal for "
                "future calculations. "
                f"Original chunks: {arg_chunks}; New chunks: {new_chunks}",
                PerformanceWarning)

    def _cache_results(self, res, zarr_format):
        os.makedirs(os.path.dirname(zarr_format), exist_ok=True)
        new_res = []
        for idx, sub_res in enumerate(res):
            if not isinstance(sub_res, da.Array):
                raise ValueError("Zarr caching currently only supports dask "
                                 f"arrays. Got {type(sub_res)}")
            zarr_path = zarr_format.format(idx)
            # See https://github.com/dask/dask/issues/8380
            with dask.config.set({"optimization.fuse.active": False}):
                new_sub_res = sub_res.to_zarr(zarr_path, compute=False)
            new_res.append(new_sub_res)
        # actually compute the storage to zarr
        da.compute(new_res)


def _get_output_chunks_from_func_arguments(args):
    """Determine what the desired output chunks are.

    It is assumed a tuple of tuples of integers is defining chunk sizes. If
    a tuple like this is not found then arguments are checked for array-like
    objects with a ``.chunks`` attribute.

    """
    chunked_args = [arg for arg in args if hasattr(arg, "chunks")]
    tuple_args = [arg for arg in args if _is_chunk_tuple(arg)]
    if not tuple_args and not chunked_args:
        raise RuntimeError("Cannot determine desired output chunksize for cached function.")
    new_chunks = tuple_args[-1] if tuple_args else chunked_args[0].chunks
    return new_chunks


def cache_to_zarr_if(
        cache_config_key: str,
        uncacheable_arg_types=DEFAULT_UNCACHE_TYPES,
        sanitize_args_func: Optional[Callable] = None,
) -> Callable:
    """Decorate a function and cache the results as a zarr array on disk.

    This only happens if the ``satpy.config`` boolean value for the provided
    key is ``True`` as well as some other conditions. See
    :class:`ZarrCacheHelper` for more information. Most importantly, this
    decorator does not limit how many items can be cached and does not clear
    out old entries. It is up to the user to manage the size of the cache.

    """
    def _decorator(func: Callable) -> Callable:
        zarr_cacher = ZarrCacheHelper(func,
                                      cache_config_key,
                                      uncacheable_arg_types,
                                      sanitize_args_func)
        wrapper = update_wrapper(zarr_cacher, func)
        return wrapper
    return _decorator


def _hash_args(*args, unhashable_types=DEFAULT_UNCACHE_TYPES):
    import json
    hashable_args = []
    for arg in args:
        if isinstance(arg, unhashable_types):
            continue
        if isinstance(arg, HASHABLE_GEOMETRIES):
            arg = hash(arg)
        elif isinstance(arg, datetime):
            arg = arg.isoformat(" ")
        hashable_args.append(arg)
    arg_hash = hashlib.sha1()  # nosec
    arg_hash.update(json.dumps(tuple(hashable_args)).encode('utf8'))
    return arg_hash.hexdigest()


def _sanitize_observer_look_args(*args):
    new_args = []
    for arg in args:
        if isinstance(arg, datetime):
            new_args.append(STATIC_EARTH_INERTIAL_DATETIME)
        elif isinstance(arg, (float, np.float64, np.float32)):
            # Round floating point numbers to nearest tenth. Numpy types don't
            # serialize into JSON which is needed for hashing, thus the casting
            # to float here:
            new_args.append(float(round(arg, 1)))
        elif _is_chunk_tuple(arg) and _chunks_are_irregular(arg):
            new_chunks = _regular_chunks_from_irregular_chunks(arg)
            new_args.append(new_chunks)
        else:
            new_args.append(arg)
    return new_args


def _sanitize_args_with_chunks(*args):
    new_args = []
    for arg in args:
        if _is_chunk_tuple(arg) and _chunks_are_irregular(arg):
            new_chunks = _regular_chunks_from_irregular_chunks(arg)
            new_args.append(new_chunks)
        else:
            new_args.append(arg)
    return new_args


def _is_chunk_tuple(some_obj: Any) -> bool:
    if not isinstance(some_obj, tuple):
        return False
    if not all(isinstance(sub_obj, tuple) for sub_obj in some_obj):
        return False
    sub_elements = [sub_obj_elem for sub_obj in some_obj for sub_obj_elem in sub_obj]
    return all(isinstance(sub_obj_elem, int) for sub_obj_elem in sub_elements)


def _regular_chunks_from_irregular_chunks(
        old_chunks: tuple[tuple[int, ...], ...]
) -> tuple[tuple[int, ...], ...]:
    shape = tuple(sum(dim_chunks) for dim_chunks in old_chunks)
    new_dim_chunks = tuple(max(dim_chunks) for dim_chunks in old_chunks)
    return da.core.normalize_chunks(new_dim_chunks, shape=shape)


def _chunks_are_irregular(chunks_tuple: tuple) -> bool:
    """Determine if an array is irregularly chunked.

    Zarr does not support saving data in irregular chunks. Regular chunking
    is when all chunks are the same size (except for the last one).

    """
    return any(len(set(chunks[:-1])) > 1 for chunks in chunks_tuple)


def _geo_dask_to_data_array(arr: da.Array) -> xr.DataArray:
    return xr.DataArray(arr, dims=('y', 'x'))


def compute_relative_azimuth(sat_azi: xr.DataArray, sun_azi: xr.DataArray) -> xr.DataArray:
    """Compute the relative azimuth angle.

    Args:
        sat_azi: DataArray for the satellite azimuth angles, typically in 0-360 degree range.
        sun_azi: DataArray for the solar azimuth angles, should be in same range as sat_azi.
    Returns:
        A DataArray containing the relative azimuth angle in the 0-180 degree range.

    NOTE: Relative azimuth is defined such that:
    Relative azimuth is 0 when sun and satellite are aligned on one side of a pixel (back scatter).
    Relative azimuth is 180 when sun and satellite are directly opposite each other (forward scatter).
    """
    ssadiff = np.absolute(sun_azi - sat_azi)
    ssadiff = np.minimum(ssadiff, 360 - ssadiff)

    return ssadiff


def get_angles(data_arr: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Get sun and satellite azimuth and zenith angles.

    Note that this function can benefit from the ``satpy.config`` parameters
    :ref:`cache_lonlats <config_cache_lonlats_setting>` and
    :ref:`cache_sensor_angles <config_cache_sensor_angles_setting>`
    being set to ``True``.

    Args:
        data_arr: DataArray to get angles for. Information extracted from this
            object are ``.attrs["area"]``,``.attrs["start_time"]``, and
            ``.attrs["orbital_parameters"]``. See :func:`satpy.utils.get_satpos`
            and :ref:`dataset_metadata` for more information.
            Additionally, the dask array chunk size is used when generating
            new arrays. The actual data of the object is not used.

    Returns:
        Four DataArrays representing sensor azimuth angle, sensor zenith angle,
        solar azimuth angle, and solar zenith angle. All values are in degrees.
        Sensor angles are provided in the [0, 360] degree range.
        Solar angles are provided in the [-180, 180] degree range.

    """
    sata, satz = _get_sensor_angles(data_arr)
    suna, sunz = _get_sun_angles(data_arr)
    return sata, satz, suna, sunz


def get_satellite_zenith_angle(data_arr: xr.DataArray) -> xr.DataArray:
    """Generate satellite zenith angle for the provided data.

    Note that this function can benefit from the ``satpy.config`` parameters
    :ref:`cache_lonlats <config_cache_lonlats_setting>` and
    :ref:`cache_sensor_angles <config_cache_sensor_angles_setting>`
    being set to ``True``. Values are in degrees.

    """
    satz = _get_sensor_angles(data_arr)[1]
    return satz


def get_cos_sza(data_arr: xr.DataArray) -> xr.DataArray:
    """Generate the cosine of the solar zenith angle for the provided data.

    Returns:
        DataArray with the same shape as ``data_arr``.

    """
    chunks = _geo_chunks_from_data_arr(data_arr)
    lons, lats = _get_valid_lonlats(data_arr.attrs["area"], chunks)
    cos_sza = _get_cos_sza(data_arr.attrs["start_time"], lons, lats)
    return _geo_dask_to_data_array(cos_sza)


@cache_to_zarr_if("cache_lonlats", sanitize_args_func=_sanitize_args_with_chunks)
def _get_valid_lonlats(area: PRGeometry, chunks: Union[int, str, tuple] = "auto") -> tuple[da.Array, da.Array]:
    with ignore_invalid_float_warnings():
        lons, lats = area.get_lonlats(chunks=chunks)
        lons = da.where(lons >= 1e30, np.nan, lons)
        lats = da.where(lats >= 1e30, np.nan, lats)
    return lons, lats


def _get_sun_angles(data_arr: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    chunks = _geo_chunks_from_data_arr(data_arr)
    lons, lats = _get_valid_lonlats(data_arr.attrs["area"], chunks)
    suna = da.map_blocks(_get_sun_azimuth_ndarray, lons, lats,
                         data_arr.attrs["start_time"],
                         dtype=lons.dtype, meta=np.array((), dtype=lons.dtype),
                         chunks=lons.chunks)
    cos_sza = _get_cos_sza(data_arr.attrs["start_time"], lons, lats)
    sunz = np.rad2deg(np.arccos(cos_sza))
    suna = _geo_dask_to_data_array(suna)
    sunz = _geo_dask_to_data_array(sunz)
    return suna, sunz


def _get_cos_sza(utc_time, lons, lats):
    cos_sza = da.map_blocks(_cos_zen_ndarray,
                            lons, lats, utc_time,
                            meta=np.array((), dtype=lons.dtype),
                            dtype=lons.dtype,
                            chunks=lons.chunks)
    return cos_sza


def _cos_zen_ndarray(lons, lats, utc_time):
    with ignore_invalid_float_warnings():
        return pyob_cos_zen(utc_time, lons, lats)


def _get_sun_azimuth_ndarray(lons: np.ndarray, lats: np.ndarray, start_time: datetime) -> np.ndarray:
    with ignore_invalid_float_warnings():
        suna = get_alt_az(start_time, lons, lats)[1]
        suna = np.rad2deg(suna)

        # The get_alt_az function returns values in the range -180 to 180 degrees.
        # Satpy expects values in the 0 - 360 range, which is what is returned for the
        # satellite azimuth angles.
        # Here this is corrected so both sun and sat azimuths are in the same range.
        suna = suna % 360.
    return suna


def _get_sensor_angles(data_arr: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    preference = satpy.config.get('sensor_angles_position_preference', 'actual')
    sat_lon, sat_lat, sat_alt = get_satpos(data_arr, preference=preference)
    area_def = data_arr.attrs["area"]
    chunks = _geo_chunks_from_data_arr(data_arr)

    sata, satz = _get_sensor_angles_from_sat_pos(sat_lon, sat_lat, sat_alt,
                                                 data_arr.attrs["start_time"],
                                                 area_def, chunks)
    sata = _geo_dask_to_data_array(sata)
    satz = _geo_dask_to_data_array(satz)
    return sata, satz


def _geo_chunks_from_data_arr(data_arr: xr.DataArray) -> tuple:
    x_dim_index = _dim_index_with_default(data_arr.dims, "x", -1)
    y_dim_index = _dim_index_with_default(data_arr.dims, "y", -2)
    chunks = (data_arr.chunks[y_dim_index], data_arr.chunks[x_dim_index])
    return chunks


def _dim_index_with_default(dims: tuple, dim_name: str, default: int) -> int:
    try:
        return dims.index(dim_name)
    except ValueError:
        return default


@cache_to_zarr_if("cache_sensor_angles", sanitize_args_func=_sanitize_observer_look_args)
def _get_sensor_angles_from_sat_pos(sat_lon, sat_lat, sat_alt, start_time, area_def, chunks):
    lons, lats = _get_valid_lonlats(area_def, chunks)
    res = da.map_blocks(_get_sensor_angles_ndarray, lons, lats, start_time, sat_lon, sat_lat, sat_alt,
                        dtype=lons.dtype, meta=np.array((), dtype=lons.dtype), new_axis=[0],
                        chunks=(2,) + lons.chunks)
    return res[0], res[1]


def _get_sensor_angles_ndarray(lons, lats, start_time, sat_lon, sat_lat, sat_alt) -> np.ndarray:
    with ignore_invalid_float_warnings():
        sata, satel = get_observer_look(
            sat_lon,
            sat_lat,
            sat_alt / 1000.0,  # km
            start_time,
            lons, lats, 0)
        satz = 90 - satel
        return np.stack([sata, satz])


def sunzen_corr_cos(data: da.Array,
                    cos_zen: da.Array,
                    limit: float = 88.,
                    max_sza: Optional[float] = 95.) -> da.Array:
    """Perform Sun zenith angle correction.

    The correction is based on the provided cosine of the zenith
    angle (``cos_zen``).  The correction is limited
    to ``limit`` degrees (default: 88.0 degrees).  For larger zenith
    angles, the correction is the same as at the ``limit`` if ``max_sza``
    is `None`. The default behavior is to gradually reduce the correction
    past ``limit`` degrees up to ``max_sza`` where the correction becomes
    0. Both ``data`` and ``cos_zen`` should be 2D arrays of the same shape.

    """
    return da.map_blocks(_sunzen_corr_cos_ndarray,
                         data, cos_zen, limit, max_sza,
                         meta=np.array((), dtype=data.dtype),
                         chunks=data.chunks)


def _sunzen_corr_cos_ndarray(data: np.ndarray,
                             cos_zen: np.ndarray,
                             limit: float,
                             max_sza: Optional[float]) -> np.ndarray:
    # Convert the zenith angle limit to cosine of zenith angle
    limit_rad = np.deg2rad(limit)
    limit_cos = np.cos(limit_rad)
    max_sza_rad = np.deg2rad(max_sza) if max_sza is not None else max_sza

    # Cosine correction
    corr = 1. / cos_zen
    if max_sza is not None:
        # gradually fall off for larger zenith angle
        grad_factor = (np.arccos(cos_zen) - limit_rad) / (max_sza_rad - limit_rad)
        # invert the factor so maximum correction is done at `limit` and falls off later
        with np.errstate(invalid='ignore'):  # we expect space pixels to be invalid
            grad_factor = 1. - np.log(grad_factor + 1) / np.log(2)
        # make sure we don't make anything negative
        grad_factor = grad_factor.clip(0.)
    else:
        # Use constant value (the limit) for larger zenith angles
        grad_factor = 1.
    corr = np.where(cos_zen > limit_cos, corr, grad_factor / limit_cos)
    # Force "night" pixels to 0 (where SZA is invalid)
    corr[np.isnan(cos_zen)] = 0
    return data * corr
