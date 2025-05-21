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

"""Shared objects of the various reader classes."""

from __future__ import annotations

import warnings
from importlib import import_module
from typing import Any


def __getattr__(name: str) -> Any:
    warn = True
    if name in (
            "FSFile",
            "open_file_or_filename"
            ):
        from . import remote
        new_submod = "remote"
        obj = getattr(remote, name)
    elif name in (
            "group_files",
            "find_files_and_readers",
            ):
        from . import grouping
        new_submod = "grouping"
        obj = getattr(grouping, name)
    elif name in (
            "read_reader_config",
            "configs_for_reader",
            "available_readers",
            "get_valid_reader_names",
            "OLD_READER_NAMES",
            "PENDING_OLD_READER_NAMES",
    ):
        from . import config
        new_submod = "config"
        obj = getattr(config, name)
    elif name in (
            "load_readers",
            "load_reader",
            "get_valid_reader_names",
    ):
        from . import loading
        new_submod = "loading"
        obj = getattr(loading, name)
    else:
        obj = import_module("."+name, package="satpy.readers")  # type: ignore
        new_submod = name
        warn = False

    if warn:
        warnings.warn(
            f"'satpy.resample.{name}' has been moved to 'satpy.resample.{new_submod}.{name}'. "
            f"Import from the new location instead (ex. 'from satpy.resample.{new_submod} import {name}').",
            stacklevel=2,
        )
    return obj
