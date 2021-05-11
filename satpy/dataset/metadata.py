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

from collections.abc import Collection
from datetime import datetime
from functools import reduce, partial
from operator import is_, eq
import numpy as np


def combine_metadata(*metadata_objects, average_times=True):
    """Combine the metadata of two or more Datasets.

    If the values corresponding to any keys are not equal or do not
    exist in all provided dictionaries then they are not included in
    the returned dictionary.  By default any keys with the word 'time'
    in them and consisting of datetime objects will be averaged. This
    is to handle cases where data were observed at almost the same time
    but not exactly.  In the interest of time, lazy arrays are compared by
    object identity rather than by their contents. Raw dataset metadata
    (`raw_metadata` attribute) are excluded as well.

    Args:
        *metadata_objects: MetadataObject or dict objects to combine
        average_times (bool): Average any keys with 'time' in the name

    Returns:
        dict: the combined metadata

    """
    info_dicts = _get_valid_dicts(metadata_objects)

    if len(info_dicts) == 1:
        return info_dicts[0].copy()

    shared_keys = _shared_keys(info_dicts)
    shared_keys.discard('raw_metadata')

    return _combine_shared_info(shared_keys, info_dicts, average_times)


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


def _combine_shared_info(shared_keys, info_dicts, average_times):
    shared_info = {}
    for key in shared_keys:
        values = [info[key] for info in info_dicts]
        if 'time' in key and isinstance(values[0], datetime) and average_times:
            shared_info[key] = average_datetimes(values)
        elif _are_values_combinable(values):
            shared_info[key] = values[0]
    return shared_info


def average_datetimes(datetime_list):
    """Average a series of datetime objects.

    .. note::

        This function assumes all datetime objects are naive and in the same
        time zone (UTC).

    Args:
        datetime_list (iterable): Datetime objects to average

    Returns: Average datetime as a datetime object

    """
    total = [datetime.timestamp(dt) for dt in datetime_list]
    return datetime.fromtimestamp(sum(total) / len(total))


def _are_values_combinable(values):
    """Check if the *values* can be combined."""
    if _contain_arrays(values):
        return _all_arrays_equal(values)
    elif _contain_collections_of_arrays(values):
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


nan_allclose = partial(np.allclose, equal_nan=True)


def _all_arrays_equal(arrays):
    """Check if the arrays are equal.

    If the arrays are lazy, just check if they have the same identity.
    """
    if hasattr(arrays[0], 'compute'):
        return _all_identical(arrays)
    else:
        return _pairwise_all(nan_allclose, arrays)


def _pairwise_all(func, values):
    for value in values[1:]:
        if not func(values[0], value):
            return False
    return True


def _all_identical(values):
    """Check that the identities of all values are the same."""
    return _pairwise_all(is_, values)


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


def _all_values_equal(values):
    try:
        return _pairwise_all(nan_allclose, values)
    except (ValueError, TypeError):
        return _pairwise_all(eq, values)
