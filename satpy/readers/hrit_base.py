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
