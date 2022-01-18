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
"""Classes and functions related to a dictionary with DataID keys."""

import numpy as np

from .dataid import DataID, create_filtered_query, minimal_default_keys_config


class TooManyResults(KeyError):
    """Special exception when one key maps to multiple items in the container."""


def get_best_dataset_key(key, choices):
    """Choose the "best" `DataID` from `choices` based on `key`.

    To see how the keys are sorted, refer to `:meth:satpy.datasets.DataQuery.sort_dataids`.

    This function assumes `choices` has already been filtered to only
    include datasets that match the provided `key`.

    Args:
        key (DataQuery): Query parameters to sort `choices` by.
        choices (iterable): `DataID` objects to sort through to determine
                            the best dataset.

    Returns: List of best `DataID`s from `choices`. If there is more
             than one element this function could not choose between the
             available datasets.

    """
    sorted_choices, distances = key.sort_dataids(choices)
    if len(sorted_choices) == 0 or distances[0] is np.inf:
        return []
    else:
        return [choice for choice, distance in zip(sorted_choices, distances) if distance == distances[0]]


def get_key(key, key_container, num_results=1, best=True, query=None,
            **kwargs):
    """Get the fully-specified key best matching the provided key.

    Only the best match is returned if `best` is `True` (default). See
    `get_best_dataset_key` for more information on how this is determined.

    `query` is provided as a convenience to filter by multiple parameters
    at once without having to filter by multiple `key` inputs.

    Args:
        key (DataID): DataID of query parameters to use for
                         searching. Any parameter that is `None`
                         is considered a wild card and any match is
                         accepted.
        key_container (dict or set): Container of DataID objects that
                                     uses hashing to quickly access items.
        num_results (int): Number of results to return. Use `0` for all
                           matching results. If `1` then the single matching
                           key is returned instead of a list of length 1.
                           (default: 1)
        best (bool): Sort results to get "best" result first
                     (default: True). See `get_best_dataset_key` for details.
        query (DataQuery): filter for the key which can contain for example:

            resolution (float, int, or list): Resolution of the dataset in
                                            dataset units (typically
                                            meters). This can also be a
                                            list of these numbers.
            calibration (str or list): Dataset calibration
                                    (ex.'reflectance'). This can also be a
                                    list of these strings.
            polarization (str or list): Dataset polarization
                                        (ex.'V'). This can also be a
                                        list of these strings.
            level (number or list): Dataset level (ex. 100). This can also be a
                                    list of these numbers.
            modifiers (list): Modifiers applied to the dataset. Unlike
                            resolution and calibration this is the exact
                            desired list of modifiers for one dataset, not
                            a list of possible modifiers.

    Returns:
        list or DataID: Matching key(s)

    Raises: KeyError if no matching results or if more than one result is
            found when `num_results` is `1`.

    """
    key = create_filtered_query(key, query)

    res = key.filter_dataids(key_container)
    if not res:
        raise KeyError("No dataset matching '{}' found".format(str(key)))

    if best:
        res = get_best_dataset_key(key, res)

    if num_results == 1 and not res:
        raise KeyError("No dataset matching '{}' found".format(str(key)))
    if num_results == 1 and len(res) != 1:
        raise TooManyResults("No unique dataset matching {}".format(str(key)))
    if num_results == 1:
        return res[0]
    if num_results == 0:
        return res

    return res[:num_results]


class DatasetDict(dict):
    """Special dictionary object that can handle dict operations based on dataset name, wavelength, or DataID.

    Note: Internal dictionary keys are `DataID` objects.

    """

    def keys(self, names=False, wavelengths=False):
        """Give currently contained keys."""
        # sort keys so things are a little more deterministic (.keys() is not)
        keys = sorted(super(DatasetDict, self).keys())
        if names:
            return (k.get('name') for k in keys)
        elif wavelengths:
            return (k.get('wavelength') for k in keys)
        else:
            return keys

    def get_key(self, match_key, num_results=1, best=True, **dfilter):
        """Get multiple fully-specified keys that match the provided query.

        Args:
            key (DataID): DataID of query parameters to use for
                          searching. Any parameter that is `None`
                          is considered a wild card and any match is
                          accepted. Can also be a string representing the
                          dataset name or a number representing the dataset
                          wavelength.
            num_results (int): Number of results to return. If `0` return all,
                               if `1` return only that element, otherwise
                               return a list of matching keys.
            **dfilter (dict): See `get_key` function for more information.

        """
        return get_key(match_key, self.keys(), num_results=num_results,
                       best=best, **dfilter)

    def getitem(self, item):
        """Get Node when we know the *exact* DataID."""
        return super(DatasetDict, self).__getitem__(item)

    def __getitem__(self, item):
        """Get item from container."""
        try:
            # short circuit - try to get the object without more work
            return super(DatasetDict, self).__getitem__(item)
        except KeyError:
            key = self.get_key(item)
            return super(DatasetDict, self).__getitem__(key)

    def get(self, key, default=None):
        """Get value with optional default."""
        try:
            key = self.get_key(key)
        except KeyError:
            return default
        return super(DatasetDict, self).get(key, default)

    def __setitem__(self, key, value):
        """Support assigning 'Dataset' objects or dictionaries of metadata."""
        if hasattr(value, 'attrs'):
            # xarray.DataArray objects
            value_info = value.attrs
        else:
            value_info = value
        # use value information to make a more complete DataID
        if not isinstance(key, DataID):
            key = self._create_dataid_key(key, value_info)

        # update the 'value' with the information contained in the key
        try:
            new_info = key.to_dict()
        except AttributeError:
            new_info = key
        if isinstance(value_info, dict):
            value_info.update(new_info)
            if isinstance(key, DataID):
                value_info['_satpy_id'] = key

        return super(DatasetDict, self).__setitem__(key, value)

    def _create_dataid_key(self, key, value_info):
        """Create a DataID key from dictionary."""
        if not isinstance(value_info, dict):
            raise ValueError("Key must be a DataID when value is not an xarray DataArray or dict")
        old_key = key
        try:
            key = self.get_key(key)
        except KeyError:
            if isinstance(old_key, str):
                new_name = old_key
            else:
                new_name = value_info.get("name")
            # this is a new key and it's not a full DataID tuple
            if new_name is None and value_info.get('wavelength') is None:
                raise ValueError("One of 'name' or 'wavelength' attrs "
                                 "values should be set.")
            id_keys = self._create_id_keys_from_dict(value_info)
            value_info['name'] = new_name
            key = DataID(id_keys, **value_info)
        return key

    def _create_id_keys_from_dict(self, value_info_dict):
        """Create id_keys from dict."""
        try:
            id_keys = value_info_dict['_satpy_id'].id_keys
        except KeyError:
            try:
                id_keys = value_info_dict['_satpy_id_keys']
            except KeyError:
                id_keys = minimal_default_keys_config
        return id_keys

    def contains(self, item):
        """Check contains when we know the *exact* DataID."""
        return super(DatasetDict, self).__contains__(item)

    def __contains__(self, item):
        """Check if item exists in container."""
        try:
            key = self.get_key(item)
        except KeyError:
            return False
        return super(DatasetDict, self).__contains__(key)

    def __delitem__(self, key):
        """Delete item from container."""
        try:
            # short circuit - try to get the object without more work
            return super(DatasetDict, self).__delitem__(key)
        except KeyError:
            key = self.get_key(key)
            return super(DatasetDict, self).__delitem__(key)
