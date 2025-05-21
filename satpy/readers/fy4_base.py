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

"""Base reader for the L1 HDF data from the AGRI and GHI instruments aboard the FengYun-4A/B satellites.

The files read by this reader are described in the official Real Time Data Service:

    http://fy4.nsmc.org.cn/data/en/data/realtime.html

"""

from __future__ import annotations

import warnings
from typing import Any


def __getattr__(name: str) -> Any:
    from .core import fy4_base

    new_submod = "core.fy4_base"
    obj = getattr(fy4_base, name)

    warnings.warn(
        f"'satpy.readers.fy4_base.{name}' has been moved to 'satpy.readers.{new_submod}.{name}'. "
        f"Import from the new location instead (ex. 'from satpy.readers.{new_submod} import {name}').",
        stacklevel=2,
    )

    return obj
