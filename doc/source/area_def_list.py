#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2022 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# satpy is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Module for autogenerating a list and overview of available area definitions ."""

from pyresample.area_config import _read_yaml_area_file_content
from pyresample.formatting_html import area_repr

from satpy.resample import get_area_def, get_area_file


def generate_area_def_list():
    """Create list of available area definitions with overview plot.

    Returns:
        str
    """
    area_list = []

    template = ("{area_name}\n"
                "{n:->{header_title_length}}\n\n"
                ".. raw:: html\n\n"
                "{content}\n\n"
                "     <hr>\n\n")

    area_file = get_area_file()[0]
    for aname in list(_read_yaml_area_file_content(area_file).keys()):
        area = get_area_def(aname)
        if hasattr(area, "_repr_html_"):
            content = "\n".join([x.rjust(len(x) + 5) for x in area_repr(area, include_header=False).split("\n")])
            area_list.append(template.format(area_name=aname, n="", header_title_length=len(aname),
                                             content=content))
        else:
            pass

    return "".join(area_list)
