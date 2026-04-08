#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023 Satpy developers
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

"""CF decoding."""

import copy
import datetime as dt
import json

import numpy as np


def decode_attrs(attrs):
    """Decode CF-encoded attributes to Python object.

    Converts timestamps to datetime and strings starting with "{" to
    dictionary.

    Args:
        attrs (dict): Attributes to be decoded

    Returns (dict): Decoded attributes
    """
    attrs = copy.deepcopy(attrs)
    _decode_dict_type_attrs(attrs)
    _decode_timestamps(attrs)
    return attrs


def _decode_dict_type_attrs(attrs):
    for key, val in attrs.items():
        attrs[key] = _str2dict(val)


def _str2dict(val):
    """Convert string to dictionary."""
    if isinstance(val, str) and val.startswith("{"):
        val = json.loads(val, object_hook=_datetime_parser_json)
    return val


def _decode_timestamps(attrs):
    for key, value in attrs.items():
        timestamp = _str2datetime(value)
        if timestamp:
            attrs[key] = timestamp


def _datetime_parser_json(json_dict):
    """Traverse JSON dictionary and parse timestamps."""
    for key, value in json_dict.items():
        timestamp = _str2datetime(value)
        if timestamp:
            json_dict[key] = timestamp
    return json_dict


def _str2datetime(string):
    """Convert string to datetime object."""
    try:
        return dt.datetime.fromisoformat(string)
    except (TypeError, ValueError):
        return None


def lazy_decode_cf_time(xrda_encoded):
    """Lazily decode CF-encoded time in limited situations.

    This is a restricted alternative to xarray.coding.times.decode_cf_datetime
    that avoids dask computations on the values to be decoded.  It is
    restricted to standard calendars (proleptic gregorian).

    Args:
        xrda_encoded (array-like):
            Xarray data array with units attribute.  The units attribute should
            be a string describing the time units using "x since timestamp",
            UDUNITS-style.
    """
    # An early iteration of this function was written with the
    # assistance of GPT-5.4.

    (unit, ref_str) = xrda_encoded.attrs["units"].split(" since ")
    ref = np.datetime64(ref_str)

    unit_map = {
        "days": "timedelta64[D]",
        "day": "timedelta64[D]",
        "hours": "timedelta64[h]",
        "hour": "timedelta64[h]",
        "minutes": "timedelta64[m]",
        "minute": "timedelta64[m]",
        "seconds": "timedelta64[s]",
        "second": "timedelta64[s]",
        "milliseconds": "timedelta64[ms]",
        "microseconds": "timedelta64[us]",
        "nanoseconds": "timedelta64[ns]",
    }

    return ref + xrda_encoded.astype(unit_map[unit.lower()])
