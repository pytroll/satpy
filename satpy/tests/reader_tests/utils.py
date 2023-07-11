#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2022 Satpy developers
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
"""Utilities for reader tests."""

import inspect
import os


def default_attr_processor(root, attr):
    """Do not change the attribute."""
    return attr


def fill_h5(root, contents, attr_processor=default_attr_processor):
    """Fill hdf5 file with the given contents.

    Args:
        root: hdf5 file rott
        contents: Contents to be written into the file
        attr_processor: A method for modifying attributes before they are
          written to the file.
    """
    for key, val in contents.items():
        if key in ["value", "attrs"]:
            continue
        if "value" in val:
            root[key] = val["value"]
        else:
            grp = root.create_group(key)
            fill_h5(grp, contents[key])
        if "attrs" in val:
            for attr_name, attr_val in val["attrs"].items():
                root[key].attrs[attr_name] = attr_processor(root, attr_val)


def get_jit_methods(module):
    """Get all jit-compiled methods in a module."""
    res = {}
    module_name = module.__name__
    members = inspect.getmembers(module)
    for member_name, obj in members:
        if _is_jit_method(obj):
            full_name = f"{module_name}.{member_name}"
            res[full_name] = obj
    return res


def _is_jit_method(obj):
    return hasattr(obj, "py_func")


def skip_numba_unstable_if_missing():
    """Determine if numba-based tests should be skipped during unstable CI tests.

    If numba fails to import it could be because numba is not compatible with
    a newer version of numpy. This is very likely to happen in the
    unstable/experimental CI environment. This function returns ``True`` if
    numba-based tests should be skipped if ``numba`` could not
    be imported *and* we're in the unstable environment. We determine if we're
    in this CI environment by looking for the ``UNSTABLE="1"``
    environment variable.

    """
    try:
        import numba
    except ImportError:
        numba = None

    return numba is None and os.environ.get("UNSTABLE", "0") in ("1", "true")
