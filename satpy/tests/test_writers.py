"""Test objects in writers init module."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "name",
    [
        "read_writer_config",
        "load_writer_configs",
        "load_writer",
        "configs_for_writer",
        "available_writers",
        "add_overlay",
        "add_text",
        "add_logo",
        "add_scale",
        "add_decorate",
        "get_enhanced_image",
        "split_results",
        "group_results_by_output_file",
        "compute_writer_results",
        "Writer",
        "ImageWriter",
    ],
)
def test_deprecated_imports(name: str):
    """Test that moved objects can be imported but warn."""
    import importlib

    writers_mod = importlib.import_module("satpy.writers")
    with pytest.warns(UserWarning, match=".*has been moved.*") as warn_catcher:
        old_obj = getattr(writers_mod, name)

    assert len(warn_catcher) == 1
    w = warn_catcher[0]
    new_imp = str(w.message).split("'")[3]
    new_mod = importlib.import_module(new_imp.rsplit(".", 1)[0])
    new_obj = getattr(new_mod, name)
    assert new_obj is old_obj
