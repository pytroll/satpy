#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2023 Satpy developers
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
"""CF processing of attributes."""
import datetime
import json
import logging
from collections import OrderedDict

import numpy as np

from satpy.writers.utils import flatten_dict

logger = logging.getLogger(__name__)


class AttributeEncoder(json.JSONEncoder):
    """JSON encoder for dataset attributes."""

    def default(self, obj):
        """Return a json-serializable object for *obj*.

        In order to facilitate decoding, elements in dictionaries, lists/tuples and multi-dimensional arrays are
        encoded recursively.
        """
        if isinstance(obj, dict):
            serialized = {}
            for key, val in obj.items():
                serialized[key] = self.default(val)
            return serialized
        elif isinstance(obj, (list, tuple, np.ndarray)):
            return [self.default(item) for item in obj]
        return self._encode(obj)

    def _encode(self, obj):
        """Encode the given object as a json-serializable datatype."""
        if isinstance(obj, (bool, np.bool_)):
            # Bool has to be checked first, because it is a subclass of int
            return str(obj).lower()
        elif isinstance(obj, (int, float, str)):
            return obj
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.void):
            return tuple(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()

        return str(obj)


def _encode_object(obj):
    """Try to encode `obj` as a netCDF/Zarr compatible datatype which most closely resembles the object's nature.

    Raises:
        ValueError if no such datatype could be found
    """
    from satpy.writers.cf_writer import NC4_DTYPES

    if isinstance(obj, int) and not isinstance(obj, (bool, np.bool_)):
        return obj
    elif isinstance(obj, (float, str, np.integer, np.floating)):
        return obj
    elif isinstance(obj, np.ndarray):
        # Only plain 1-d arrays are supported. Skip record arrays and multi-dimensional arrays.
        is_plain_1d = not obj.dtype.fields and len(obj.shape) <= 1
        if is_plain_1d:
            if obj.dtype in NC4_DTYPES:
                return obj
            elif obj.dtype == np.bool_:
                # Boolean arrays are not supported, convert to array of strings.
                return [s.lower() for s in obj.astype(str)]
            return obj.tolist()
    raise ValueError('Unable to encode')


def _encode_python_objects(obj):
    """Try to find the datatype which most closely resembles the object's nature.

    If on failure, encode as a string. Plain lists are encoded recursively.
    """
    if isinstance(obj, (list, tuple)) and all([not isinstance(item, (list, tuple)) for item in obj]):
        return [_encode_to_cf(item) for item in obj]
    try:
        dump = _encode_object(obj)
    except ValueError:
        try:
            # Decode byte-strings
            decoded = obj.decode()
        except AttributeError:
            decoded = obj
        dump = json.dumps(decoded, cls=AttributeEncoder).strip('"')
    return dump


def _encode_to_cf(obj):
    """Encode the given object as a netcdf compatible datatype."""
    try:
        return obj.to_cf()
    except AttributeError:
        return _encode_python_objects(obj)


def _encode_nc_attrs(attrs):
    """Encode dataset attributes in a netcdf compatible datatype.

    Args:
        attrs (dict):
            Attributes to be encoded
    Returns:
        dict: Encoded (and sorted) attributes

    """
    encoded_attrs = []
    for key, val in sorted(attrs.items()):
        if val is not None:
            encoded_attrs.append((key, _encode_to_cf(val)))
    return OrderedDict(encoded_attrs)


def _add_ancillary_variables_attrs(dataarray):
    """Replace ancillary_variables DataArray with a list of their name."""
    list_ancillary_variable_names = [da_ancillary.attrs['name']
                                     for da_ancillary in dataarray.attrs.get('ancillary_variables', [])]
    if list_ancillary_variable_names:
        dataarray.attrs['ancillary_variables'] = ' '.join(list_ancillary_variable_names)
    else:
        dataarray.attrs.pop("ancillary_variables", None)
    return dataarray


def _drop_exclude_attrs(dataarray, exclude_attrs):
    """Remove user-specified list of attributes."""
    if exclude_attrs is None:
        exclude_attrs = []
    for key in exclude_attrs:
        dataarray.attrs.pop(key, None)
    return dataarray


def _remove_satpy_attrs(new_data):
    """Remove _satpy attribute."""
    satpy_attrs = [key for key in new_data.attrs if key.startswith('_satpy')]
    for satpy_attr in satpy_attrs:
        new_data.attrs.pop(satpy_attr)
    new_data.attrs.pop('_last_resampler', None)
    return new_data


def _format_prerequisites_attrs(dataarray):
    """Reformat prerequisites attribute value to string."""
    if 'prerequisites' in dataarray.attrs:
        dataarray.attrs['prerequisites'] = [np.bytes_(str(prereq)) for prereq in dataarray.attrs['prerequisites']]
    return dataarray


def _remove_none_attrs(dataarray):
    """Remove attribute keys with None value."""
    for key, val in dataarray.attrs.copy().items():
        if val is None:
            dataarray.attrs.pop(key)
    return dataarray


def preprocess_datarray_attrs(dataarray, flatten_attrs, exclude_attrs):
    """Preprocess DataArray attributes to be written into CF-compliant netCDF/Zarr."""
    dataarray = _remove_satpy_attrs(dataarray)
    dataarray = _add_ancillary_variables_attrs(dataarray)
    dataarray = _drop_exclude_attrs(dataarray, exclude_attrs)
    dataarray = _format_prerequisites_attrs(dataarray)
    dataarray = _remove_none_attrs(dataarray)
    _ = dataarray.attrs.pop("area", None)

    if 'long_name' not in dataarray.attrs and 'standard_name' not in dataarray.attrs:
        dataarray.attrs['long_name'] = dataarray.name

    if flatten_attrs:
        dataarray.attrs = flatten_dict(dataarray.attrs)

    dataarray.attrs = _encode_nc_attrs(dataarray.attrs)

    return dataarray


def _add_history(attrs):
    """Add 'history' attribute to dictionary."""
    _history_create = 'Created by pytroll/satpy on {}'.format(datetime.datetime.utcnow())
    if 'history' in attrs:
        if isinstance(attrs['history'], list):
            attrs['history'] = ''.join(attrs['history'])
        attrs['history'] += '\n' + _history_create
    else:
        attrs['history'] = _history_create
    return attrs


def preprocess_header_attrs(header_attrs, flatten_attrs=False):
    """Prepare file header attributes."""
    if header_attrs is not None:
        if flatten_attrs:
            header_attrs = flatten_dict(header_attrs)
        header_attrs = _encode_nc_attrs(header_attrs)  # OrderedDict
    else:
        header_attrs = {}
    header_attrs = _add_history(header_attrs)
    return header_attrs
