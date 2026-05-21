"""Functions and classes related to MultiScene functionality."""
from __future__ import annotations

from typing import Any

from satpy.utils import _import_and_warn_new_location

from ._multiscene import MultiScene  # noqa

IMPORT_PATHS = {
    "stack": "satpy.multiscene.blend_funcs",
    "temporal_rgb": "satpy.multiscene.blend_funcs",
    "timeseries": "satpy.multiscene.blend_funcs",
}


def __getattr__(name: str) -> Any:
    new_module = IMPORT_PATHS.get(name)

    if new_module is None:
        raise AttributeError(f"module {__name__} has no attribute '{name}'")

    return _import_and_warn_new_location(new_module, name)
