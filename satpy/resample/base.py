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


def _update_resampled_coords(old_data, new_data, new_area):
    """Add coordinate information to newly resampled DataArray.

    Args:
        old_data (xarray.DataArray): Old data before resampling.
        new_data (xarray.DataArray): New data after resampling.
        new_area (pyresample.geometry.BaseDefinition): Area definition
            for the newly resampled data.

    """
    from satpy.coords_utils import add_crs_xy_coords

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
    from pyresample.resampler import BaseResampler as PRBaseResampler

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
    from pyresample.resampler import BaseResampler as PRBaseResampler

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
