"""Writers subpackage."""
from __future__ import annotations

from typing import Any

from satpy.utils import _import_and_warn_new_location

IMPORT_PATHS = {
    "Writer": "satpy.writers.core.base",
    "ImageWriter": "satpy.writers.core.image",
    "add_overlay": "satpy.enhancements.overlays",
    "add_decorate": "satpy.enhancements.overlays",
    "add_scale": "satpy.enhancements.overlays",
    "add_logo": "satpy.enhancements.overlays",
    "add_text": "satpy.enhancements.overlays",
    "split_results": "satpy.writers.core.compute",
    "group_results_by_output_file": "satpy.writers.core.compute",
    "compute_writer_results": "satpy.writers.core.compute",
    "get_enhanced_image": "satpy.enhancements.enhancer",
    "read_writer_config": "satpy.writers.core.config",
    "load_writer_configs": "satpy.writers.core.config",
    "load_writer": "satpy.writers.core.config",
    "configs_for_writer": "satpy.writers.core.config",
    "available_writers": "satpy.writers.core.config",
}

def __getattr__(name: str) -> Any:
    if name == "show":
        raise AttributeError("The 'show' function has been removed. Use 'get_enhanced_image(data_arr).show()' instead.")
    if name == "to_image":
        raise AttributeError("The 'to_image' function has been removed. Use 'trollimage.xrimage.XRImage' instead.")

    new_module = IMPORT_PATHS.get(name)

    if new_module is None:
        raise AttributeError(f"module {__name__} has no attribute '{name}'")

    return _import_and_warn_new_location(new_module, name)
