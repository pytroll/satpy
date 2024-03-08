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
from __future__ import annotations

import datetime
import json
import logging
from collections import OrderedDict

import numpy as np
import xarray as xr

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


def _encode_numpy_array(obj):
    """Encode numpy array as a netCDF4 serializable datatype."""
    from satpy.writers.cf_writer import NC4_DTYPES

    # Only plain 1-d arrays are supported. Skip record arrays and multi-dimensional arrays.
    is_plain_1d = not obj.dtype.fields and len(obj.shape) <= 1
    if not is_plain_1d:
        raise ValueError("Only a 1D numpy array can be encoded as netCDF attribute.")
    if obj.dtype in NC4_DTYPES:
        return obj
    if obj.dtype == np.bool_:
        # Boolean arrays are not supported, convert to array of strings.
        return [s.lower() for s in obj.astype(str)]
    return obj.tolist()


def _encode_object(obj):
    """Try to encode `obj` as a netCDF/Zarr compatible datatype which most closely resembles the object's nature.

    Raises:
        ValueError if no such datatype could be found
    """
    is_nonbool_int = isinstance(obj, int) and not isinstance(obj, (bool, np.bool_))
    is_encode_type = isinstance(obj, (float, str, np.integer, np.floating))
    if is_nonbool_int or is_encode_type:
        return obj
    elif isinstance(obj, np.ndarray):
        return _encode_numpy_array(obj)
    raise ValueError("Unable to encode")


def _try_decode_object(obj):
    """Try to decode byte string."""
    try:
        decoded = obj.decode()
    except AttributeError:
        decoded = obj
    return decoded


def _encode_python_objects(obj):
    """Try to find the datatype which most closely resembles the object's nature.

    If on failure, encode as a string. Plain lists are encoded recursively.
    """
    if isinstance(obj, (list, tuple)) and all([not isinstance(item, (list, tuple)) for item in obj]):
        return [_encode_to_cf(item) for item in obj]
    try:
        dump = _encode_object(obj)
    except ValueError:
        decoded = _try_decode_object(obj)
        dump = json.dumps(decoded, cls=AttributeEncoder).strip('"')
    return dump


def _encode_to_cf(obj):
    """Encode the given object as a netcdf compatible datatype."""
    try:
        return obj.to_cf()
    except AttributeError:
        return _encode_python_objects(obj)


def encode_attrs_to_cf(attrs):
    """Encode dataset attributes as a netcdf compatible datatype.

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


def preprocess_attrs(
        data_arr: xr.DataArray,
        flatten_attrs: bool,
        exclude_attrs: list[str] | None
) -> xr.DataArray:
    """Preprocess DataArray attributes to be written into CF-compliant netCDF/Zarr."""
    _drop_attrs(data_arr, exclude_attrs)
    _add_ancillary_variables_attrs(data_arr)
    _format_prerequisites_attrs(data_arr)

    if "long_name" not in data_arr.attrs and "standard_name" not in data_arr.attrs:
        data_arr.attrs["long_name"] = data_arr.name

    if flatten_attrs:
        data_arr.attrs = flatten_dict(data_arr.attrs)

    data_arr.attrs = encode_attrs_to_cf(data_arr.attrs)

    return data_arr


def _drop_attrs(
        data_arr: xr.DataArray,
        user_excluded_attrs: list[str] | None
) -> None:
    """Remove undesirable attributes."""
    attrs_to_drop = (
            (user_excluded_attrs or []) +
            _get_satpy_attrs(data_arr) +
            _get_none_attrs(data_arr) +
            ["area"]
    )
    for key in attrs_to_drop:
        data_arr.attrs.pop(key, None)


def _get_satpy_attrs(data_arr: xr.DataArray) -> list[str]:
    """Remove _satpy attribute."""
    return [key for key in data_arr.attrs if key.startswith("_satpy")] + ["_last_resampler"]


def _get_none_attrs(data_arr: xr.DataArray) -> list[str]:
    """Remove attribute keys with None value."""
    return [attr_name for attr_name, attr_val in data_arr.attrs.items() if attr_val is None]


def _add_ancillary_variables_attrs(data_arr: xr.DataArray) -> None:
    """Replace ancillary_variables DataArray with a list of their name."""
    list_ancillary_variable_names = [da_ancillary.attrs["name"]
                                     for da_ancillary in data_arr.attrs.get("ancillary_variables", [])]
    if list_ancillary_variable_names:
        data_arr.attrs["ancillary_variables"] = " ".join(list_ancillary_variable_names)
    else:
        data_arr.attrs.pop("ancillary_variables", None)


def _format_prerequisites_attrs(data_arr: xr.DataArray) -> None:
    """Reformat prerequisites attribute value to string."""
    if "prerequisites" in data_arr.attrs:
        data_arr.attrs["prerequisites"] = [np.bytes_(str(prereq)) for prereq in data_arr.attrs["prerequisites"]]


def _add_history(attrs):
    """Add 'history' attribute to dictionary."""
    _history_create = "Created by pytroll/satpy on {}".format(datetime.datetime.utcnow())
    if "history" in attrs:
        if isinstance(attrs["history"], list):
            attrs["history"] = "".join(attrs["history"])
        attrs["history"] += "\n" + _history_create
    else:
        attrs["history"] = _history_create
    return attrs


def preprocess_header_attrs(header_attrs, flatten_attrs=False):
    """Prepare file header attributes."""
    if header_attrs is not None:
        if flatten_attrs:
            header_attrs = flatten_dict(header_attrs)
        header_attrs = encode_attrs_to_cf(header_attrs)  # OrderedDict
    else:
        header_attrs = {}
    header_attrs = _add_history(header_attrs)
    return header_attrs
