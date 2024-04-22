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
