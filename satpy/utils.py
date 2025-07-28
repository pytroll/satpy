# Copyright (c) 2009-2023 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# satpy is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Module defining various utilities."""

from __future__ import annotations

import contextlib
import datetime
import importlib.metadata
import logging
import os
import pathlib
import platform
import warnings
from collections.abc import Mapping, MutableMapping
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Literal, Optional
from urllib.parse import urlparse

import dask.utils
import numpy as np
import xarray as xr
import yaml
from yaml import BaseLoader, UnsafeLoader

from satpy._compat import ArrayLike, DTypeLike

_is_logging_on = False
TRACE_LEVEL = 5

logger = logging.getLogger(__name__)


class PerformanceWarning(Warning):
    """Warning raised when there is a possible performance impact."""


def debug_on(deprecation_warnings=True):
    """Turn debugging logging on.

    Sets up a StreamHandler to to `sys.stderr` at debug level for all
    loggers, such that all debug messages (and log messages with higher
    severity) are logged to the standard error stream.

    By default, since Satpy 0.26, this also enables the global visibility
    of deprecation warnings.  This can be suppressed by passing a false
    value.

    Args:
        deprecation_warnings (Optional[bool]): Switch on deprecation warnings.
            Defaults to True.

    Returns:
        None
    """
    logging_on(logging.DEBUG)
    if deprecation_warnings:
        deprecation_warnings_on()


def debug_off():
    """Turn debugging logging off.

    This disables both debugging logging and the global visibility of
    deprecation warnings.
    """
    logging_off()
    deprecation_warnings_off()


@contextlib.contextmanager
def debug(deprecation_warnings=True):
    """Context manager to temporarily set debugging on.

    Example::

        >>> with satpy.utils.debug():
        ...     code_here()

    Args:
        deprecation_warnings (Optional[bool]): Switch on deprecation warnings.
            Defaults to True.
    """
    debug_on(deprecation_warnings=deprecation_warnings)
    yield
    debug_off()


def trace_on():
    """Turn trace logging on."""
    logging_on(TRACE_LEVEL)


class _WarningManager:
    """Class to handle switching warnings on and off."""

    filt = None


_warning_manager = _WarningManager()


def deprecation_warnings_on():
    """Switch on deprecation warnings."""
    warnings.filterwarnings("default", category=DeprecationWarning)
    _warning_manager.filt = warnings.filters[0]


def deprecation_warnings_off():
    """Switch off deprecation warnings."""
    if _warning_manager.filt in warnings.filters:
        warnings.filters.remove(_warning_manager.filt)


def logging_on(level=logging.WARNING):
    """Turn logging on."""
    global _is_logging_on

    if not _is_logging_on:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("[%(levelname)s: %(asctime)s :"
                                               " %(name)s] %(message)s",
                                               "%Y-%m-%d %H:%M:%S"))
        console.setLevel(level)
        logging.getLogger("").addHandler(console)
        _is_logging_on = True

    log = logging.getLogger("")
    log.setLevel(level)
    for h in log.handlers:
        h.setLevel(level)


def logging_off():
    """Turn logging off."""
    logging.getLogger("").handlers = [logging.NullHandler()]


def get_logger(name):
    """Return logger with null handler added if needed."""
    if not hasattr(logging.Logger, "trace"):
        logging.addLevelName(TRACE_LEVEL, "TRACE")

        def trace(self, message, *args, **kwargs):
            if self.isEnabledFor(TRACE_LEVEL):
                # Yes, logger takes its '*args' as 'args'.
                self._log(TRACE_LEVEL, message, args, **kwargs)

        logging.Logger.trace = trace

    log = logging.getLogger(name)
    return log


def in_ipynb():
    """Check if we are in a jupyter notebook."""
    try:
        return "ZMQ" in get_ipython().__class__.__name__
    except NameError:
        return False


# Spherical conversions


def lonlat2xyz(
        lon: ArrayLike,
        lat: ArrayLike,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Convert lon lat to cartesian.

    For a sphere with unit radius, convert the spherical coordinates
    longitude and latitude to cartesian coordinates.

    Args:
        lon: Longitude in °.
        lat: Latitude in °.

    Returns:
        (x, y, z) Cartesian coordinates [1]
    """
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return x, y, z


def xyz2lonlat(
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        asin: bool = False,
) -> tuple[ArrayLike, ArrayLike]:
    """Convert cartesian to lon lat.

    For a sphere with unit radius, convert cartesian coordinates to spherical
    coordinates longitude and latitude.

    Args:
        x: x-coordinate, unitless
        y: y-coordinate, unitless
        z: z-coordinate, unitless
        asin: If true, use arcsin for calculations.
            If false, use arctan2 for calculations.

    Returns:
        Longitude and latitude in °.
    """
    lon = np.rad2deg(np.arctan2(y, x))
    if asin:
        lat = np.rad2deg(np.arcsin(z))
    else:
        lat = np.rad2deg(np.arctan2(z, np.sqrt(x ** 2 + y ** 2)))
    return lon, lat


def angle2xyz(azi, zen):
    """Convert azimuth and zenith to cartesian."""
    azi = np.deg2rad(azi)
    zen = np.deg2rad(zen)
    x = np.sin(zen) * np.sin(azi)
    y = np.sin(zen) * np.cos(azi)
    z = np.cos(zen)
    return x, y, z


def xyz2angle(x, y, z, acos=False):
    """Convert cartesian to azimuth and zenith."""
    azi = np.rad2deg(np.arctan2(x, y))
    if acos:
        zen = np.rad2deg(np.arccos(z))
    else:
        zen = 90 - np.rad2deg(np.arctan2(z, np.sqrt(x ** 2 + y ** 2)))
    return azi, zen


def proj_units_to_meters(proj_str):
    """Convert projection units from kilometers to meters."""
    proj_parts = proj_str.split()
    new_parts = []
    for itm in proj_parts:
        key, val = itm.split("=")
        key = key.strip("+")
        if key in ["a", "b", "h"]:
            val = float(val)
            if val < 6e6:
                val *= 1000.
                val = "%.3f" % val

        if key == "units" and val == "km":
            continue

        new_parts.append("+%s=%s" % (key, val))

    return " ".join(new_parts)


def _get_sunz_corr_li_and_shibata(cos_zen):
    return 24.35 / (2. * cos_zen + np.sqrt(498.5225 * cos_zen**2 + 1))


def atmospheric_path_length_correction(data, cos_zen, limit=88., max_sza=95.):
    """Perform Sun zenith angle correction.

    This function uses the correction method proposed by
    Li and Shibata (2006): https://doi.org/10.1175/JAS3682.1

    The correction is limited to ``limit`` degrees (default: 88.0 degrees). For
    larger zenith angles, the correction is the same as at the ``limit`` if
    ``max_sza`` is `None`. The default behavior is to gradually reduce the
    correction past ``limit`` degrees up to ``max_sza`` where the correction
    becomes 0. Both ``data`` and ``cos_zen`` should be 2D arrays of the same
    shape.

    """
    # Convert the zenith angle limit to cosine of zenith angle
    limit_rad = np.deg2rad(limit)
    limit_cos = np.cos(limit_rad)
    max_sza_rad = np.deg2rad(max_sza) if max_sza is not None else max_sza

    # Cosine correction
    corr = _get_sunz_corr_li_and_shibata(cos_zen)
    # Use constant value (the limit) for larger zenith angles
    corr_lim = _get_sunz_corr_li_and_shibata(limit_cos)

    if max_sza is not None:
        # gradually fall off for larger zenith angle
        grad_factor = (np.arccos(cos_zen) - limit_rad) / (max_sza_rad - limit_rad)
        # invert the factor so maximum correction is done at `limit` and falls off later
        grad_factor = 1. - np.log(grad_factor + 1) / np.log(2)
        # make sure we don't make anything negative
        grad_factor = grad_factor.clip(0.)
    else:
        # Use constant value (the limit) for larger zenith angles
        grad_factor = 1.
    corr = corr.where(cos_zen > limit_cos, grad_factor * corr_lim)
    # Force "night" pixels to 0 (where SZA is invalid)
    corr = corr.where(cos_zen.notnull(), 0)

    return data * corr


def get_satpos(
        data_arr: xr.DataArray,
        preference: Optional[str] = None,
        use_tle: bool = False
) -> tuple[float, float, float]:
    """Get satellite position from dataset attributes.

    Args:
        data_arr: DataArray object to access ``.attrs`` metadata
            from.
        preference: Optional preference for one of the available types of
            position information. If not provided or ``None`` then the default
            preference is:

            * Longitude & Latitude: nadir, actual, nominal, projection
            * Altitude: actual, nominal, projection

            The provided ``preference`` can be any one of these individual
            strings (nadir, actual, nominal, projection). If the
            preference is not available then the original preference list is
            used. A warning is issued when projection values have to be used because
            nothing else is available and it wasn't provided as the ``preference``.
        use_tle: If true, try to obtain position via satellite name
            and TLE if it can't be determined otherwise.  This requires pyorbital, skyfield,
            and astropy to be installed and may need network access to obtain the TLE.
            Note that even if ``use_tle`` is true, the TLE will not be used if
            the dataset metadata contain the satellite position directly.

    Returns:
        Geodetic longitude, latitude, altitude [km]

    """
    if preference is not None and preference not in ("nadir", "actual", "nominal", "projection"):
        raise ValueError(f"Unrecognized satellite coordinate preference: {preference}")
    lonlat_prefixes = ("nadir_", "satellite_actual_", "satellite_nominal_", "projection_")
    alt_prefixes = _get_prefix_order_by_preference(lonlat_prefixes[1:], preference)
    lonlat_prefixes = _get_prefix_order_by_preference(lonlat_prefixes, preference)
    try:
        lon, lat = _get_sat_lonlat(data_arr, lonlat_prefixes)
        alt = _get_sat_altitude(data_arr, alt_prefixes)
    except KeyError:
        if use_tle:
            logger.warning(
                    "Orbital parameters missing from metadata.  "
                    "Calculating from TLE using skyfield and astropy.")
            return _get_satpos_from_platform_name(data_arr)
        raise KeyError("Unable to determine satellite position. Either the "
                       "reader doesn't provide that information or "
                       "geolocation datasets were not available.")
    return lon, lat, alt


def _get_prefix_order_by_preference(prefixes, preference):
    preferred_prefixes = [prefix for prefix in prefixes if preference and preference in prefix]
    nonpreferred_prefixes = [prefix for prefix in prefixes if not preference or preference not in prefix]
    if nonpreferred_prefixes[-1] == "projection_":
        # remove projection as a prefix as it is our fallback
        nonpreferred_prefixes = nonpreferred_prefixes[:-1]
    return preferred_prefixes + nonpreferred_prefixes


def _get_sat_altitude(data_arr, key_prefixes):
    orb_params = data_arr.attrs["orbital_parameters"]
    alt_keys = [prefix + "altitude" for prefix in key_prefixes]
    try:
        alt = _get_first_available_item(orb_params, alt_keys)
    except KeyError:
        alt = orb_params["projection_altitude"]
        warnings.warn(
            "Actual satellite altitude not available, using projection altitude instead.",
            stacklevel=3
        )
    return alt


def _get_sat_lonlat(data_arr, key_prefixes):
    orb_params = data_arr.attrs["orbital_parameters"]
    lon_keys = [prefix + "longitude" for prefix in key_prefixes]
    lat_keys = [prefix + "latitude" for prefix in key_prefixes]
    try:
        lon = _get_first_available_item(orb_params, lon_keys)
        lat = _get_first_available_item(orb_params, lat_keys)
    except KeyError:
        lon = orb_params["projection_longitude"]
        lat = orb_params["projection_latitude"]
        warnings.warn(
            "Actual satellite lon/lat not available, using projection center instead.",
            stacklevel=3
        )
    return lon, lat


def _get_satpos_from_platform_name(cth_dataset):
    """Get satellite position if no orbital parameters in metadata.

    Some cloud top height datasets lack orbital parameter information in
    metadata.  Here, orbital parameters are calculated based on the platform
    name and start time, via Two Line Element (TLE) information.

    Needs pyorbital, skyfield, and astropy to be installed.
    """
    from pyorbital.orbital import tlefile
    from skyfield.api import EarthSatellite, load
    from skyfield.toposlib import wgs84

    name = cth_dataset.attrs["platform_name"]
    tle = tlefile.read(name)
    es = EarthSatellite(tle.line1, tle.line2, name)
    ts = load.timescale()
    gc = es.at(ts.from_datetime(
        cth_dataset.attrs["start_time"].replace(tzinfo=datetime.timezone.utc)))
    (lat, lon) = wgs84.latlon_of(gc)
    height = wgs84.height_of(gc).to("km")
    return (lon.degrees, lat.degrees, height.value)


def _get_first_available_item(data_dict, possible_keys):
    for possible_key in possible_keys:
        try:
            return data_dict[possible_key]
        except KeyError:
            continue
    raise KeyError("None of the possible keys found: {}".format(", ".join(possible_keys)))


def recursive_dict_update(d: MutableMapping, u: Mapping) -> None:
    """Recursive dictionary update.

    Copied from:

        http://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth

    """
    for k, v in u.items():
        if isinstance(v, Mapping):
            r = d.get(k, {})
            recursive_dict_update(r, v)
            d[k] = r
        else:
            d[k] = u[k]


def _check_yaml_configs(configs, key):
    """Get a diagnostic for the yaml *configs*.

    *key* is the section to look for to get a name for the config at hand.
    """
    diagnostic = {}
    for i in configs:
        for fname in i:
            msg = "ok"
            res = None
            with open(fname, "r", encoding="utf-8") as stream:
                try:
                    res = yaml.load(stream, Loader=UnsafeLoader)
                except yaml.YAMLError as err:
                    stream.seek(0)
                    res = yaml.load(stream, Loader=BaseLoader)
                    if err.context == "while constructing a Python object":
                        msg = err.problem
                    else:
                        msg = "error"
                finally:
                    try:
                        diagnostic[res[key]["name"]] = msg
                    except (KeyError, TypeError):
                        # this object doesn't have a 'name'
                        pass
    return diagnostic

def _check_package_version(package_name: str) -> Optional[str]:
    """Check the version of `package_name`.

    Args:
        package_name (str): the distribution package name.

    Returns:
        the version number if available else `None`.
    """
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def show_versions(packages=None):
    """Shows version for system, python and common packages (if installed).

    Args:
        packages (list or None): Limit packages to those specified.

    Returns:
        None.

    """
    packages = (
        (
            "cartopy",
            "geoviews",
            "numpy",
            "dask",
            "xarray",
            "gdal",
            "rasterio",
            "pyproj",
            "netcdf4",
            "h5py",
            "pyhdf",
            "h5netcdf",
            "fsspec",
        )
        if packages is None
        else packages
    )

    print("Versions")  # noqa: T201
    print("======")  # noqa: T201
    print(f"platform: {platform.platform()}")  # noqa: T201
    print(f"python: {platform.python_version()}")  # noqa: T201
    print()  # noqa: T201

    for package_name in sorted(packages):
        package_version = _check_package_version(package_name)
        print( # noqa: T201
            f"{package_name}: {package_version if package_version else 'not installed'}"
        )

    print()  # noqa: T201


def check_satpy(readers=None, writers=None, packages=None):
    """Check the satpy readers and writers for correct installation.

    Args:
        readers (list or None): Limit readers checked to those specified
        writers (list or None): Limit writers checked to those specified
        packages (list or None): Limit packages checked to those specified

    Returns:
        None

    """
    from satpy.readers.core.config import configs_for_reader
    from satpy.writers.core.config import configs_for_writer

    print("Readers")  # noqa: T201
    print("=======")  # noqa: T201
    for reader, res in sorted(_check_yaml_configs(configs_for_reader(reader=readers), "reader").items()):
        print(reader + ": ", res)  # noqa: T201
    print()  # noqa: T201

    print("Writers")  # noqa: T201
    print("=======")  # noqa: T201
    for writer, res in sorted(_check_yaml_configs(configs_for_writer(writer=writers), "writer").items()):
        print(writer + ": ", res)  # noqa: T201
    print()  # noqa: T201

    show_versions(packages=packages)


def unify_chunks(*data_arrays: xr.DataArray) -> tuple[xr.DataArray, ...]:
    """Run :func:`xarray.unify_chunks` if input dimensions are all the same size.

    This is mostly used in :class:`satpy.composites.CompositeBase` to safe
    guard against running :func:`dask.array.map_blocks` with arrays of
    different chunk sizes. Doing so can cause unexpected results or errors.
    However, xarray's ``unify_chunks`` will raise an exception if dimensions
    of the provided DataArrays are different sizes. This is a common case for
    Satpy. For example, the "bands" dimension may be 1 (L), 2 (LA), 3 (RGB), or
    4 (RGBA) for most compositor operations that combine other composites
    together.

    """
    if not hasattr(xr, "unify_chunks"):
        return data_arrays
    if not _all_dims_same_size(data_arrays):
        return data_arrays
    return tuple(xr.unify_chunks(*data_arrays))


def _all_dims_same_size(data_arrays: tuple[xr.DataArray, ...]) -> bool:
    known_sizes: dict[str, int] = {}
    for data_arr in data_arrays:
        for dim, dim_size in data_arr.sizes.items():
            known_size = known_sizes.setdefault(dim, dim_size)
            if dim_size != known_size:
                # this dimension is a different size than previously found
                # xarray.unify_chunks will error out if we tried to use it
                return False
    return True


@contextlib.contextmanager
def ignore_invalid_float_warnings():
    """Ignore warnings generated for working with NaN/inf values.

    Numpy and dask sometimes don't like NaN or inf values in normal function
    calls. This context manager hides/ignores them inside its context.

    Examples:
        Use around numpy operations that you expect to produce warnings::

            with ignore_invalid_float_warnings():
                np.nanmean(np.nan)

    """
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


@contextlib.contextmanager
def ignore_pyproj_proj_warnings():
    """Wrap operations that we know will produce a PROJ.4 precision warning.

    Only to be used internally to Pyresample when we have no other choice but
    to use PROJ.4 strings/dicts. For example, serialization to YAML or other
    human-readable formats or testing the methods that produce the PROJ.4
    versions of the CRS.

    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            "You will likely lose important projection information",
            UserWarning,
        )
        yield


def get_chunk_size_limit(dtype=float):
    """Compute the chunk size limit in bytes given *dtype* (float by default).

    It is derived from PYTROLL_CHUNK_SIZE if defined (although deprecated) first, from dask config's `array.chunk-size`
    then. It defaults to 128MiB.

    Returns:
        The recommended chunk size in bytes.
    """
    pixel_size = _get_chunk_pixel_size()
    if pixel_size is not None:
        return pixel_size * np.dtype(dtype).itemsize
    return get_dask_chunk_size_in_bytes()


def get_dask_chunk_size_in_bytes():
    """Get the dask configured chunk size in bytes."""
    return dask.utils.parse_bytes(dask.config.get("array.chunk-size", "128MiB"))


def _get_chunk_pixel_size():
    """Compute the maximum chunk size from PYTROLL_CHUNK_SIZE."""
    legacy_chunk_size = _get_pytroll_chunk_size()
    if legacy_chunk_size is not None:
        return legacy_chunk_size ** 2


def get_legacy_chunk_size():
    """Get the legacy chunk size.

    This function should only be used while waiting for code to be migrated to use satpy.utils.get_chunk_size_limit
    instead.
    """
    chunk_size = _get_pytroll_chunk_size()

    if chunk_size is not None:
        return chunk_size

    import math

    return int(math.sqrt(get_dask_chunk_size_in_bytes() / 8))


def _get_pytroll_chunk_size():
    try:
        chunk_size = int(os.environ["PYTROLL_CHUNK_SIZE"])
        warnings.warn(
            "The PYTROLL_CHUNK_SIZE environment variable is pending deprecation. "
            "You can use the dask config setting `array.chunk-size` (or the DASK_ARRAY__CHUNK_SIZE environment"
            " variable) and set it to the square of the PYTROLL_CHUNK_SIZE instead.",
            stacklevel=2
        )
        return chunk_size
    except KeyError:
        return None


def normalize_low_res_chunks(
        chunks: tuple[int | Literal["auto"], ...],
        input_shape: tuple[int, ...],
        previous_chunks: tuple[int, ...],
        low_res_multipliers: tuple[int, ...],
        input_dtype: DTypeLike,
) -> tuple[int, ...]:
    """Compute dask chunk sizes based on data resolution.

    First, chunks are computed for the highest resolution version of the data.
    This is done by multiplying the input array shape by the
    ``low_res_multiplier`` and then using Dask's utility functions and
    configuration to produce a chunk size to fit into a specific number of
    bytes. See :doc:`dask:array-chunks` for more information.
    Next, the same multiplier is used to reduce the high resolution chunk sizes
    to the lower resolution of the input data. The end result of reading
    multiple resolutions of data is that each dask chunk covers the same
    geographic region. This also means replicating or aggregating one
    resolution and then combining arrays should not require any rechunking.

    Args:
        chunks: Requested chunk size for each dimension. This is passed
            directly to dask. Use ``"auto"`` for dimensions that should have
            chunks determined for them, ``-1`` for dimensions that should be
            whole (not chunked), and ``1`` or any other positive integer for
            dimensions that have a known chunk size beforehand.
        input_shape: Shape of the array to compute dask chunk size for.
        previous_chunks: Any previous chunking or structure of the data. This
            can also be thought of as the smallest number of high (fine) resolution
            elements that make up a single "unit" or chunk of data. This could
            be a multiple or factor of the scan size for some instruments and/or
            could be based on the on-disk chunk size. This value ensures that
            chunks are aligned to the underlying data structure for best
            performance. On-disk chunk sizes should be multiplied by the
            largest low resolution multiplier if it is the same between all
            files (ex. 500m file has 226 chunk size, 1km file has 226 chunk
            size, etc).. Otherwise, the resulting low resolution chunks may
            not be aligned to the on-disk chunks. For example, if dask decides
            on a chunk size of 226 * 3 for 500m data, that becomes 226 * 3 / 2
            for 1km data which is not aligned to the on-disk chunk size of 226.
        low_res_multipliers: Number of high (fine) resolution pixels that fit
            in a single low (coarse) resolution pixel.
        input_dtype: Dtype for the final unscaled array. This is usually
            32-bit float (``np.float32``) or 64-bit float (``np.float64``)
            for non-category data. If this doesn't represent the final data
            type of the data then the final size of chunks in memory will not
            match the user's request via dask's ``array.chunk-size``
            configuration. Sometimes it is useful to keep this as a single
            dtype for all reading functionality (ex. ``np.float32``) in order
            to keep all read variable chunks the same size regardless of dtype.

    Returns:
        A tuple where each element is the chunk size for that axis/dimension.

    """
    if any(len(input_shape) != len(param) for param in (low_res_multipliers, chunks, previous_chunks)):
        raise ValueError("Input shape, low res multipliers, chunks, and previous chunks must all be the same size")
    high_res_shape = tuple(dim_size * lr_mult for dim_size, lr_mult in zip(input_shape, low_res_multipliers))
    chunks_for_high_res = dask.array.core.normalize_chunks(
        chunks,
        shape=high_res_shape,
        dtype=input_dtype,
        previous_chunks=previous_chunks,
    )
    low_res_chunks: list[int] = []
    for req_chunks, hr_chunks, prev_chunks, lr_mult in zip(
            chunks,
            chunks_for_high_res,
            previous_chunks, low_res_multipliers
    ):
        if req_chunks != "auto":
            low_res_chunks.append(req_chunks)
            continue
        low_res_chunks.append(round(max(hr_chunks[0] / lr_mult, prev_chunks / lr_mult)))
    return tuple(low_res_chunks)


def convert_remote_files_to_fsspec(filenames, storage_options=None):
    """Check filenames for transfer protocols, convert to FSFile objects if possible."""
    if storage_options is None:
        storage_options = {}
    if isinstance(filenames, dict):
        return _check_file_protocols_for_dicts(filenames, storage_options)
    return _check_file_protocols(filenames, storage_options)


def _check_file_protocols_for_dicts(filenames, storage_options):
    res = {}
    for reader, files in filenames.items():
        opts = storage_options.get(reader, {})
        res[reader] = _check_file_protocols(files, opts)
    return res


def _check_file_protocols(filenames, storage_options):
    local_files, remote_files, fs_files = _sort_files_to_local_remote_and_fsfiles(filenames)

    if remote_files:
        return local_files + fs_files + _filenames_to_fsfile(remote_files, storage_options)

    return local_files + fs_files


def _sort_files_to_local_remote_and_fsfiles(filenames):
    from satpy.readers.core.remote import FSFile

    local_files = []
    remote_files = []
    fs_files = []
    for f in filenames:
        if isinstance(f, FSFile):
            fs_files.append(f)
        elif isinstance(f, pathlib.Path):
            local_files.append(f)
        elif urlparse(f).scheme in ("", "file") or "\\" in f:
            local_files.append(f)
        else:
            remote_files.append(f)
    return local_files, remote_files, fs_files


def _filenames_to_fsfile(filenames, storage_options):
    import fsspec

    from satpy.readers.core.remote import FSFile

    if filenames:
        fsspec_files = fsspec.open_files(filenames, **storage_options)
        return [FSFile(f) for f in fsspec_files]
    return []


def get_storage_options_from_reader_kwargs(reader_kwargs):
    """Read and clean storage options from reader_kwargs."""
    if reader_kwargs is None:
        return None, None
    new_reader_kwargs = deepcopy(reader_kwargs)  # don't modify user provided dict
    storage_options = _get_storage_dictionary_options(new_reader_kwargs)
    return storage_options, new_reader_kwargs


def _get_storage_dictionary_options(reader_kwargs):
    storage_opt_dict = {}
    shared_storage_options = reader_kwargs.pop("storage_options", {})
    if not reader_kwargs:
        # no other reader kwargs
        return shared_storage_options
    for reader_name, rkwargs in reader_kwargs.items():
        if not isinstance(rkwargs, dict):
            # reader kwargs are not per-reader, return a single dictionary of storage options
            return shared_storage_options
        if shared_storage_options:
            # set base storage options if there are any
            storage_opt_dict[reader_name] = shared_storage_options.copy()
        if isinstance(rkwargs, dict) and "storage_options" in rkwargs:
            storage_opt_dict.setdefault(reader_name, {}).update(rkwargs.pop("storage_options"))
    return storage_opt_dict


@contextmanager
def import_error_helper(dependency_name):
    """Give more info on an import error."""
    try:
        yield
    except ImportError as err:
        raise ImportError(err.msg + f" It can be installed with the {dependency_name} package.")


def find_in_ancillary(data, dataset):
    """Find a dataset by name in the ancillary vars of another dataset.

    Args:
        data (xarray.DataArray):
            Array for which to search the ancillary variables
        dataset (str):
            Name of ancillary variable to look for.
    """
    matches = [x for x in data.attrs["ancillary_variables"] if x.attrs.get("name") == dataset]
    cnt = len(matches)
    if cnt < 1:
        raise ValueError(
            f"Could not find dataset named {dataset:s} in ancillary "
            f"variables for dataset {data.attrs.get('name')!r}")
    if cnt > 1:
        raise ValueError(
            f"Expected exactly one dataset named {dataset:s} in ancillary "
            f"variables for dataset {data.attrs.get('name')!r}, "
            f"found {cnt:d}")
    return matches[0]


def datetime64_to_pydatetime(dt64: np.datetime64) -> datetime.datetime:
    """Convert numpy.datetime64 timestamp to Python datetime.

    Discards nanosecond precision, because Python datetime only has microsecond
    precision.

    Args:
        dt64: Timestamp to be converted
    Returns:
        Converted timestamp
    """
    return dt64.astype("datetime64[us]").astype(datetime.datetime)


def _import_and_warn_new_location(new_module: str, name: str) -> Any:
    import inspect
    from importlib import import_module

    frame = inspect.stack()[1]
    old_module_name = inspect.getmodule(frame[0]).__name__  # type: ignore

    mod = import_module(new_module)
    obj = getattr(mod, name)

    warnings.warn(
        f"'{old_module_name}.{name}' has been moved to '{new_module}.{name}'. "
        f"Import from the new location instead (ex. 'from {new_module} import {name}'). "
        "The old import paths will be removed in Satpy 1.0",
        stacklevel=3,
    )
    return obj


def flatten_dict(d, parent_key="", sep="_"):
    """Flatten a nested dictionary.

    Based on https://stackoverflow.com/a/6027615/5703449
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, parent_key=new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
