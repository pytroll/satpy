
"""Advance Baseline Imager reader base class for the Level 1b and l2+ reader."""

from __future__ import annotations

from typing import Any

from satpy.utils import _import_and_warn_new_location


def __getattr__(name: str) -> Any:
    new_module = "satpy.readers.core.abi"

    return _import_and_warn_new_location(new_module, name)
