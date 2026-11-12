
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
