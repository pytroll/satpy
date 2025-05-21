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

import warnings
from typing import Any


def __getattr__(name: str) -> Any:
    from .core import hrit

    new_submod = "core.hrit"
    obj = getattr(hrit, name)

    warnings.warn(
        f"'satpy.resample.hrit.{name}' has been moved to 'satpy.resample.{new_submod}.{name}'. "
        f"Import from the new location instead (ex. 'from satpy.resample.{new_submod} import {name}').",
        stacklevel=2,
    )

    return obj
