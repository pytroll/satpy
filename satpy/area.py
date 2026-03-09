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
import warnings

from pyresample.area_config import AreaNotFound, _create_area_def_from_dict, _read_yaml_area_file_content

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

def deprecate_area_name(fun):
    """Decorator to deprecate area names.

    This function is used to retire area names or warn about changed area names.
    """
    def area_with_new_name(area_name):
        try:
            return fun(area_name)
        except AreaNotFound:
            area_file_name = get_area_file()
            areas_dict = _read_yaml_area_file_content(area_file_name)
            area_dict = None
            for new_area_name, v in areas_dict.items():

                if "old_name" in v.keys() and area_name in v["old_name"]:# == area_name:
                    area_dict = areas_dict.get(new_area_name)
                    del area_dict["old_name"]
                    break

            if area_dict is not None:
                warnings.simplefilter("always", DeprecationWarning)
                warnings.warn(f"This area name is being deprecated in Satpy v1.0. Pleas use the new area name:"
                              f" {new_area_name}", DeprecationWarning, stacklevel=2)
                warnings.simplefilter("default", DeprecationWarning)
                return _create_area_def_from_dict(new_area_name, area_dict)
            else:
                raise AreaNotFound('Area "{0}" not found in file "{1}"'.format(area_name, area_file_name))

    return area_with_new_name

@deprecate_area_name
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
