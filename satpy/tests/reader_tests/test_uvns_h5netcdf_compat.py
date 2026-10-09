"""Tests for UVNS h5netcdf compatibility helpers."""
from __future__ import annotations

from types import SimpleNamespace

from xarray.backends.h5netcdf_ import H5NetCDFStore

from satpy.readers._h5netcdf_compat import (
    apply_h5netcdf_compatibility_fixes,
)


def test_apply_patch_is_idempotent():
    """Installing the patch multiple times should be harmless."""
    if hasattr(H5NetCDFStore, "_uvns_vlen_patch"):
        delattr(H5NetCDFStore, "_uvns_vlen_patch")

    apply_h5netcdf_compatibility_fixes()

    first = H5NetCDFStore.open_store_variable

    apply_h5netcdf_compatibility_fixes()

    second = H5NetCDFStore.open_store_variable

    assert first is second


def test_non_vlen_uses_original_path(monkeypatch):
    """Non-VLEN variables should continue through xarray's original path."""
    if hasattr(H5NetCDFStore, "_uvns_vlen_patch"):
        delattr(H5NetCDFStore, "_uvns_vlen_patch")

    original_called = False

    def fake_original(self, name, var):
        nonlocal original_called
        original_called = True
        return "original_result"

    monkeypatch.setattr(
        H5NetCDFStore,
        "open_store_variable",
        fake_original,
    )

    apply_h5netcdf_compatibility_fixes()

    fake_var = SimpleNamespace(
        _root=SimpleNamespace(
            _h5py=SimpleNamespace(
                check_string_dtype=lambda dtype: None
            )
        ),
        _h5ds=SimpleNamespace(
            dtype=object(),
        ),
    )

    fake_store = SimpleNamespace(
        _filename="dummy.nc",
    )

    result = H5NetCDFStore.open_store_variable(
        fake_store,
        "test",
        fake_var,
    )

    assert original_called
    assert result == "original_result"


def test_vlen_string_bypasses_original_implementation(
    monkeypatch,
):
    """VLEN strings should not use the original xarray implementation."""
    if hasattr(H5NetCDFStore, "_uvns_vlen_patch"):
        delattr(H5NetCDFStore, "_uvns_vlen_patch")

    original_called = False

    def fake_original(
        self,
        name,
        var,
    ):
        nonlocal original_called

        original_called = True

        raise AssertionError(
            "Original implementation should not be called "
            "for VLEN strings."
        )

    monkeypatch.setattr(
        H5NetCDFStore,
        "open_store_variable",
        fake_original,
    )

    apply_h5netcdf_compatibility_fixes()

    fake_var = SimpleNamespace(
        dimensions=("x",),
        _root=SimpleNamespace(
            _h5py=SimpleNamespace(
                check_string_dtype=lambda dtype:
                SimpleNamespace(length=None)
            )
        ),
        _h5ds=SimpleNamespace(
            dtype=object(),
        ),
    )

    fake_store = SimpleNamespace(
        _filename="dummy.nc",
    )

    try:
        H5NetCDFStore.open_store_variable(
            fake_store,
            "test",
            fake_var,
        )
    except Exception:
        #
        # We don't care if later xarray internals fail.
        #
        # The behaviour under test is that the patched
        # implementation takes the VLEN branch and does
        # not invoke the original implementation.
        #
        pass

    assert not original_called
