# Copyright (c) 2017-2023 Satpy developers
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

"""Enhancements."""

from __future__ import annotations

from typing import Any

from satpy.utils import _import_and_warn_new_location

IMPORT_PATHS = {
    "stretch": "satpy.enhancements.contrast",
    "gamma": "satpy.enhancements.contrast",
    "invert": "satpy.enhancements.contrast",
    "piecewise_linear_stretch": "satpy.enhancements.contrast",
    "cira_stretch": "satpy.enhancements.contrast",
    "reinhard_to_srgb": "satpy.enhancements.contrast",
    "btemp_threshold": "satpy.enhancements.contrast",
    "jma_true_color_reproduction": "satpy.enhancements.ahi",
    "three_d_effect": "satpy.enhancements.convolution",
    "exclude_alpha": "satpy.enhancements.wrappers",
    "on_separate_bands": "satpy.enhancements.wrappers",
    "on_dask_array": "satpy.enhancements.wrappers",
    "using_map_blocks": "satpy.enhancements.wrappers",
    "lookup": "satpy.enhancements.colormap",
    "colorize": "satpy.enhancements.colormap",
    "palettize": "satpy.enhancements.colormap",
    "create_colormap": "satpy.enhancements.colormap",
}


def __getattr__(name: str) -> Any:
    new_module = IMPORT_PATHS.get(name)

    if new_module is None:
        raise AttributeError(f"module {__name__} has no attribute '{name}'")

    return _import_and_warn_new_location(new_module, name)
