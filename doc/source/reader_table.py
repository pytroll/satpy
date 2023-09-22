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
"""Module for autogenerating reader table from config files."""

from yaml import BaseLoader

from satpy.readers import available_readers


def rst_table_row(columns=None):
    """Create one row for a rst table.

    Args:
        columns (list[str]): Content of each column.
    Returns:
        str
    """
    row = "    * - {}\n".format(columns[0])
    columns = ["      - {}\n".format(col) for col in columns[1:]]
    row = row + "".join(columns)

    return row


def rst_table_header(name=None, header=None, header_rows=1, widths="auto"):
    """Create header for rst table.

    Args:
        name (str): Name of the table
        header (list[str]): Column names
        header-rows (int): Number of header rows
        width (optional[list[int]]): Width of each column as a list. If not specified
            defaults to auto and will therefore determined by the backend
            (see <https://docutils.sourceforge.io/docs/ref/rst/directives.html#table>)
    Returns:
        str
    """
    if isinstance(widths, list):
        widths = " ".join([str(w) for w in widths])

    header = rst_table_row(header)

    table_header = (f".. list-table:: {name}\n"
                    f"    :header-rows: {header_rows}\n"
                    f"    :widths: {widths}\n"
                    f"    :class: datatable\n\n"
                    f"{header}")

    return table_header


def generate_reader_table():
    """Create reader table from reader yaml config files.

    Returns:
        str
    """
    table = [rst_table_header("Satpy Readers", header=["Description", "Reader name", "Status", "fsspec support"],
                              widths=[45, 25, 30, 30])]

    reader_configs = available_readers(as_dict=True, yaml_loader=BaseLoader)
    for rc in reader_configs:
        table.append(rst_table_row([rc.get("long_name", "").rstrip("\n"), rc.get("name", ""),
                                    rc.get("status", ""), rc.get("supports_fsspec", "false")]))

    return "".join(table)
