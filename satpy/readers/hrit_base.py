#!/usr/bin/env python
# Copyright (c) 2016-2025 Satpy developers
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
"""HRIT/LRIT format reader.

This module is the base module for all HRIT-based formats. Here, you will find
the common building blocks for hrit reading.

One of the features here is the on-the-fly decompression of hrit files when
compressed hrit files are encountered (files finishing with `.C_`).
"""

from __future__ import annotations

from typing import Any

from satpy.utils import _import_and_warn_new_location


def __getattr__(name: str) -> Any:
    new_module = "satpy.readers.core.hrit"

    return _import_and_warn_new_location(new_module, name)
