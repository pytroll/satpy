"""Helpers for reading hdf5-based files."""

from __future__ import annotations

from typing import Any

from satpy.utils import _import_and_warn_new_location


def __getattr__(name: str) -> Any:
    new_module = "satpy.readers.core.hdf5"

    return _import_and_warn_new_location(new_module, name)
