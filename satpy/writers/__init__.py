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
    if name == "show":
        raise AttributeError("The 'show' function has been removed. Use 'get_enhanced_image(data_arr).show()' instead.")

    if name == "Writer":
        from satpy.writers.core.base import Writer

        new_submod = "core.base"
        obj = Writer
    elif name == "ImageWriter":
        from satpy.writers.core.image import ImageWriter

        new_submod = "core.image"
        obj = ImageWriter
    elif name in ("add_overlay", "add_decorate", "add_scale", "add_logo", "add_text"):
        from . import overlay_utils

        new_submod = "overlay_utils"
        obj = getattr(overlay_utils, name)
    elif name in (
        "get_enhanced_image",
        "to_image",
        "split_results",
        "group_results_by_output_file",
        "compute_writer_results",
    ):
        from . import utils

        new_submod = "utils"
        obj = getattr(utils, name)
    elif name in (
            "read_writer_config",
            "load_writer_configs",
            "load_writer",
            "configs_for_writer",
            "available_writers",
        ):
        from .core import config

        new_submod = "core.config"
        obj = getattr(config, name)
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    warnings.warn(
        f"'satpy.writers.{name}' has been moved to 'satpy.writers.{new_submod}.{name}'. "
        f"Import from the new location instead (ex. 'from satpy.writers.{new_submod} import {name}').",
        stacklevel=2,
    )
    return obj
