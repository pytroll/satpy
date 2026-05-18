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
"""Utility functions for area definitions."""
from ._config import config_search_paths, get_config_path


def get_area_file():
    """Find area file(s) to use.

    The files are to be named `areas.yaml` or `areas.def`.
    """
    paths = config_search_paths("areas.yaml")
    if paths:
        return paths
    else:
        return get_config_path("areas.def")


def get_area_def(area_name):
    """Get the definition of *area_name* from file.

    The file is defined to use is to be placed in the $SATPY_CONFIG_PATH
    directory, and its name is defined in satpy's configuration file.
    """
    try:
        from pyresample import parse_area_file
    except ImportError:
        from pyresample.utils import parse_area_file
    return parse_area_file(get_area_file(), area_name)[0]
