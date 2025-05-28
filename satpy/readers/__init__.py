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

IMPORT_PATHS = {
    "FSFile": "satpy.readers.core.remote",
    "open_file_or_filename": "satpy.readers.core.remote",
    "group_files": "satpy.readers.core.grouping",
    "find_files_and_readers": "satpy.readers.core.grouping",
    "read_reader_config": "satpy.readers.core.config",
    "configs_for_reader": "satpy.readers.core.config",
    "available_readers": "satpy.readers.core.config",
    "get_valid_reader_names": "satpy.readers.core.config",
    "OLD_READER_NAMES": "satpy.readers.core.config",
    "PENDING_OLD_READER_NAMES": "satpy.readers.core.config",
    "load_readers": "satpy.readers.core.loading",
    "load_reader": "satpy.readers.core.loading",
}

def __getattr__(name: str) -> Any:
    new_module = IMPORT_PATHS.get(name)

    if new_module is None:
        return import_module("."+name, package="satpy.readers")  # type: ignore

    mod = import_module(new_module)
    warnings.warn(
        f"'satpy.readers.{name}' has been moved to '{new_module}.{name}'. "
        f"Import from the new location instead (ex. 'from {new_module} import {name}'). "
        "The old import paths will be removed in Satpy 1.0",
        stacklevel=2,
    )
    return getattr(mod, name)
