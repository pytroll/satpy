
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
