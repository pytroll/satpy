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
"""Base resampling functionality."""
import hashlib
import json
import warnings
from logging import getLogger
from weakref import WeakValueDictionary

import numpy as np
import xarray as xr
from pyresample.geometry import SwathDefinition
from pyresample.resampler import BaseResampler as PRBaseResampler

from satpy._config import config_search_paths, get_config_path
from satpy.utils import get_legacy_chunk_size

LOG = getLogger(__name__)

CHUNK_SIZE = get_legacy_chunk_size()

resamplers_cache: "WeakValueDictionary[tuple, object]" = WeakValueDictionary()


def _hash_dict(the_dict, the_hash=None):
    """Calculate a hash for a dictionary."""
    if the_hash is None:
        the_hash = hashlib.sha1()  # nosec
    the_hash.update(json.dumps(the_dict, sort_keys=True).encode("utf-8"))
    return the_hash


def get_area_file():
    """Find area file(s) to use.

    The files are to be named `areas.yaml` or `areas.def`.
    """
    paths = config_search_paths("areas.yaml")
    if paths:
        return paths
    else:
        return get_config_path("areas.def")


def get_area_def(area_name):
    """Get the definition of *area_name* from file.

    The file is defined to use is to be placed in the $SATPY_CONFIG_PATH
    directory, and its name is defined in satpy's configuration file.
    """
    try:
        from pyresample import parse_area_file
    except ImportError:
        from pyresample.utils import parse_area_file
    return parse_area_file(get_area_file(), area_name)[0]


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
    # add CRS object if pyproj 2.0+
    try:
        from pyproj import CRS
    except ImportError:
        LOG.debug("Could not add 'crs' coordinate with pyproj<2.0")
        crs = None
    else:
        # default lat/lon projection
        latlon_proj = "+proj=latlong +datum=WGS84 +ellps=WGS84"
        # otherwise get it from the area definition
        if hasattr(area, "crs"):
            crs = area.crs
        else:
            proj_str = getattr(area, "proj_str", latlon_proj)
            crs = CRS.from_string(proj_str)
        data_arr = data_arr.assign_coords(crs=crs)
    return crs, data_arr


def _update_resampled_coords(old_data, new_data, new_area):
    """Add coordinate information to newly resampled DataArray.

    Args:
        old_data (xarray.DataArray): Old data before resampling.
        new_data (xarray.DataArray): New data after resampling.
        new_area (pyresample.geometry.BaseDefinition): Area definition
            for the newly resampled data.

    """
    # copy over other non-x/y coordinates
    # this *MUST* happen before we set 'crs' below otherwise any 'crs'
    # coordinate in the coordinate variables we are copying will overwrite the
    # 'crs' coordinate we just assigned to the data
    ignore_coords = ("y", "x", "crs")
    new_coords = {}
    for cname, cval in old_data.coords.items():
        # we don't want coordinates that depended on the old x/y dimensions
        has_ignored_dims = any(dim in cval.dims for dim in ignore_coords)
        if cname in ignore_coords or has_ignored_dims:
            continue
        new_coords[cname] = cval
    new_data = new_data.assign_coords(**new_coords)

    # add crs, x, and y coordinates
    new_data = add_crs_xy_coords(new_data, new_area)
    return new_data


# TODO: move this to pyresample.resampler
#RESAMPLERS = {"kd_tree": KDTreeResampler,
#              "nearest": KDTreeResampler,
#              "bilinear": BilinearResampler,
#              "native": NativeResampler,
#              "gradient_search": create_gradient_search_resampler,
#              "bucket_avg": BucketAvg,
#              "bucket_sum": BucketSum,
#              "bucket_count": BucketCount,
#              "bucket_fraction": BucketFraction,
#              "ewa": DaskEWAResampler,
#              "ewa_legacy": LegacyDaskEWAResampler,
#              }

def get_all_resampler_classes():
    """Get all available resampler classes."""
    resamplers = {}
    try:
        from .native import get_native_resampler_classes
        resamplers.update(get_native_resampler_classes())
    except ImportError:
        pass
    try:
        from .kdtree import get_kdtree_resampler_classes
        resamplers.update(get_kdtree_resampler_classes())
    except ImportError:
        pass
    try:
        from .bucket import get_bucket_resampler_classes
        resamplers.update(get_bucket_resampler_classes())
    except ImportError:
        pass
    try:
        from .ewa import get_ewa_resampler_classes
        resamplers.update(get_ewa_resampler_classes())
    except ImportError:
        pass
    try:
        from pyresample.gradient import create_gradient_search_resampler
        resamplers["gradient_search"] = create_gradient_search_resampler
    except ImportError:
        pass

    return resamplers


# TODO: move this to pyresample
def prepare_resampler(source_area, destination_area, resampler=None, **resample_kwargs):
    """Instantiate and return a resampler."""
    if resampler is None:
        LOG.info("Using default KDTree resampler")
        resampler = "kd_tree"

    if isinstance(resampler, PRBaseResampler):
        raise ValueError("Trying to create a resampler when one already "
                         "exists.")
    if isinstance(resampler, str):
        resampler_class = get_all_resampler_classes().get(resampler, None)
        _check_resampler_class(resampler_class, resampler)
    else:
        resampler_class = resampler

    key = (resampler_class,
           source_area, destination_area,
           _hash_dict(resample_kwargs).hexdigest())
    try:
        resampler_instance = resamplers_cache[key]
    except KeyError:
        resampler_instance = resampler_class(source_area, destination_area)
        resamplers_cache[key] = resampler_instance
    return key, resampler_instance


def _check_resampler_class(resampler_class, resampler):
    if resampler_class is not None:
        return
    if resampler == "gradient_search":
        warnings.warn(
            "Gradient search resampler not available. Maybe missing `shapely`?",
            stacklevel=2
        )
    raise KeyError("Resampler '%s' not available" % resampler)


# TODO: move this to pyresample
def resample(source_area, data, destination_area,
             resampler=None, **kwargs):
    """Do the resampling."""
    if not isinstance(resampler, PRBaseResampler):
        # we don't use the first argument (cache key)
        _, resampler_instance = prepare_resampler(source_area,
                                                  destination_area,
                                                  resampler)
    else:
        resampler_instance = resampler

    if isinstance(data, list):
        res = [resampler_instance.resample(ds, **kwargs) for ds in data]
    else:
        res = resampler_instance.resample(data, **kwargs)

    return res


def _get_fill_value(dataset):
    """Get the fill value of the *dataset*, defaulting to np.nan."""
    if np.issubdtype(dataset.dtype, np.integer):
        return dataset.attrs.get("_FillValue", np.nan)
    return np.nan


def resample_dataset(dataset, destination_area, **kwargs):
    """Resample *dataset* and return the resampled version.

    Args:
        dataset (xarray.DataArray): Data to be resampled.
        destination_area: The destination onto which to project the data,
          either a full blown area definition or a string corresponding to
          the name of the area as defined in the area file.
        **kwargs: The extra parameters to pass to the resampler objects.

    Returns:
        A resampled DataArray with updated ``.attrs["area"]`` field. The dtype
        of the array is preserved.

    """
    # call the projection stuff here
    try:
        source_area = dataset.attrs["area"]
    except KeyError:
        LOG.info("Cannot reproject dataset %s, missing area info",
                 dataset.attrs["name"])

        return dataset

    fill_value = kwargs.pop("fill_value", _get_fill_value(dataset))
    new_data = resample(source_area, dataset, destination_area, fill_value=fill_value, **kwargs)
    new_attrs = new_data.attrs
    new_data.attrs = dataset.attrs.copy()
    new_data.attrs.update(new_attrs)
    new_data.attrs.update(area=destination_area)

    return new_data
