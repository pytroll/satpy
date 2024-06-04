#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 Satpy developers
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

"""Utilities for merging metadata from various sources."""

import datetime as dt
import warnings
from collections.abc import Collection
from functools import partial, reduce
from operator import eq, is_

import numpy as np

from satpy.writers.utils import flatten_dict


def combine_metadata(*metadata_objects, average_times=None):
    """Combine the metadata of two or more Datasets.

    If the values corresponding to any keys are not equal or do not
    exist in all provided dictionaries then they are not included in
    the returned dictionary.

    All values of the keys containing the substring 'start_time' will be set
    to the earliest value and similarly for 'end_time' to latest time.  All
    other keys containing the word 'time' are averaged.  Before these adjustments,
    `None` values resulting from data that don't have times associated to them
    are removed. These rules are applied also to values in the 'time_parameters'
    dictionary.

    .. versionchanged:: 0.47

       Before Satpy 0.47, all times, including `start_time` and `end_time`, were averaged.

    In the interest of processing time, lazy arrays are compared by object
    identity rather than by their contents.

    Args:
        *metadata_objects: MetadataObject or dict objects to combine

    Kwargs:
        average_times (bool): Removed option to average all time attributes.

    Returns:
        dict: the combined metadata

    """
    if average_times is not None:
        warnings.warn(
            "'average_time' option has been removed and start/end times are handled with min/max instead.",
            UserWarning
        )

    info_dicts = _get_valid_dicts(metadata_objects)
    if len(info_dicts) == 1:
        return info_dicts[0].copy()

    shared_keys = _shared_keys(info_dicts)

    return _combine_shared_info(shared_keys, info_dicts)


def _get_valid_dicts(metadata_objects):
    """Get the valid dictionaries matching the metadata_objects."""
    info_dicts = []
    for metadata_object in metadata_objects:
        if isinstance(metadata_object, dict):
            metadata_dict = metadata_object
        elif hasattr(metadata_object, "attrs"):
            metadata_dict = metadata_object.attrs
        else:
            continue
        info_dicts.append(metadata_dict)
    return info_dicts


def _shared_keys(info_dicts):
    key_sets = (set(metadata_dict.keys()) for metadata_dict in info_dicts)
    return reduce(set.intersection, key_sets)


def _combine_shared_info(shared_keys, info_dicts):
    shared_info = {}
    for key in shared_keys:
        values = [info[key] for info in info_dicts]
        _combine_values(key, values, shared_info)
    return shared_info


def _combine_values(key, values, shared_info):
    if "time" in key:
        times = _combine_times(key, values)
        if times is not None:
            shared_info[key] = times
    elif _are_values_combinable(values):
        shared_info[key] = values[0]


def _combine_times(key, values):
    if key == "time_parameters":
        return _combine_time_parameters(values)
    filtered_values = _filter_time_values(values)
    if not filtered_values:
        return None
    if "end_time" in key:
        return max(filtered_values)
    elif "start_time" in key:
        return min(filtered_values)
    return average_datetimes(filtered_values)


def _combine_time_parameters(values):
    # Assume the first item has all the keys
    keys = values[0].keys()
    res = {}
    for key in keys:
        sub_values = [itm[key] for itm in values]
        res[key] = _combine_times(key, sub_values)
    return res


def _filter_time_values(values):
     """Remove values that are not datetime objects."""
     return [v for v in values if isinstance(v, dt.datetime)]


def average_datetimes(datetime_list):
    """Average a series of datetime objects.

    .. note::

        This function assumes all datetime objects are naive and in the same
        time zone (UTC).

    Args:
        datetime_list (iterable): Datetime objects to average

    Returns: Average datetime as a datetime object

    """
    total = [dt.datetime.timestamp(d) for d in datetime_list]
    return dt.datetime.fromtimestamp(sum(total) / len(total))


def _are_values_combinable(values):
    """Check if the *values* can be combined."""
    if _contain_dicts(values):
        return _all_dicts_equal(values)
    return _all_non_dicts_equal(values)


def _all_non_dicts_equal(values):
    if _contain_arrays(values):
        return _all_arrays_equal(values)
    if _contain_collections_of_arrays(values):
        # in the real world, the `ancillary_variables` attribute may be
        # List[xarray.DataArray], this means our values are now
        # List[List[xarray.DataArray]].
        # note that this list_of_arrays check is also true for any
        # higher-dimensional ndarray, but we only use this check after we have
        # checked any_arrays so this false positive should have no impact
        return _all_list_of_arrays_equal(values)
    return _all_values_equal(values)


def _contain_arrays(values):
    return any([_is_array(value) for value in values])


def _is_array(val):
    """Check if val is an array."""
    return hasattr(val, "__array__") and not np.isscalar(val)


def _contain_dicts(values):
    return any(isinstance(value, dict) for value in values)


nan_allclose = partial(np.allclose, equal_nan=True)


def _all_arrays_equal(arrays):
    """Check if the arrays are equal.

    If the arrays are lazy, just check if they have the same identity.
    """
    if hasattr(arrays[0], "compute"):
        return _all_identical(arrays)
    return _all_values_equal(arrays)


def _all_values_equal(values):
    try:
        return _all_close(values)
    except (ValueError, TypeError):
        # In case of object type arrays (e.g. datetime) _all_close fails,
        # but _all_equal succeeds.
        return _all_equal(values)


def _all_dicts_equal(dicts):
    try:
        return _pairwise_all(_dict_equal, dicts)
    except AttributeError:
        # There is something else than a dictionary in the list
        return False


def _dict_equal(d1, d2):
    """Check that two dictionaries are equal.

    Nested dictionaries are flattened to facilitate comparison.
    """
    d1_flat = flatten_dict(d1)
    d2_flat = flatten_dict(d2)
    if not _dict_keys_equal(d1_flat, d2_flat):
        return False
    for key in d1_flat.keys():
        value_pair = [d1_flat[key], d2_flat[key]]
        if not _all_non_dicts_equal(value_pair):
            return False
    return True


def _dict_keys_equal(d1, d2):
    return d1.keys() == d2.keys()


def _pairwise_all(func, values):
    for value in values[1:]:
        if not _is_equal(values[0], value, func):
            return False
    return True


def _is_equal(a, b, comp_func):
    res = comp_func(a, b)
    if _is_array(res):
        return res.all()
    return res


def _all_identical(values):
    """Check that the identities of all values are the same."""
    return _pairwise_all(is_, values)


def _all_close(values):
    return _pairwise_all(nan_allclose, values)


def _all_equal(values):
    return _pairwise_all(eq, values)


def _contain_collections_of_arrays(values):
    return any(
        [_is_non_empty_collection(value) and
         _is_all_arrays(value)
         for value in values])


def _is_non_empty_collection(value):
    return isinstance(value, Collection) and len(value) > 0


def _is_all_arrays(value):
    return all([_is_array(sub_value) for sub_value in value])


def _all_list_of_arrays_equal(array_lists):
    """Check that the lists of arrays are equal."""
    for array_list in zip(*array_lists):
        if not _all_arrays_equal(array_list):
            return False
    return True
