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
"""Writers subpackage."""
from __future__ import annotations

import warnings
from typing import Any


def __getattr__(name: str) -> Any:
    if name in ("KDTreeResampler", "BilinearResampler"):
        from . import kdtree

        new_submod = "kdtree"
        obj = getattr(kdtree, name)
    elif name == "NativeResampler":
        from .native import NativeResampler
        new_submod = "native"
        obj = NativeResampler
    elif name in (
            "BucketResamplerBase",
            "BucketAvg",
            "BucketSum",
            "BucketCount",
            "BucketFraction"
    ):
        from . import bucket
        new_submod = "bucket"
        obj = getattr(bucket, name)
    elif name in (
            "hash_dict",
            "get_area_file",
            "get_area_def",
            "add_xy_coords",
            "add_crs_xy_coords",
            "update_resampled_coords",
            "resample",
            "prepare_resampler",
            "get_fill_value",
            "resample_dataset"
    ):
            from . import base
            new_submod = "base"
            obj = getattr(base, name)
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    warnings.warn(
        f"'satpy.resample.{name}' has been moved to 'satpy.resample.{new_submod}.{name}'. "
        f"Import from the new location instead (ex. 'from satpy.resample.{new_submod} import {name}').",
        stacklevel=2,
    )
    return obj
