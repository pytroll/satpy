
"""Base reader for the L1 HDF data from the AGRI and GHI instruments aboard the FengYun-4A/B satellites.

The files read by this reader are described in the official Real Time Data Service:

    http://fy4.nsmc.org.cn/data/en/data/realtime.html

"""

from __future__ import annotations

from typing import Any

from satpy.utils import _import_and_warn_new_location


def __getattr__(name: str) -> Any:
    new_module = "satpy.readers.core.fy4"

    return _import_and_warn_new_location(new_module, name)
