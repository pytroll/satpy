"""Tests for the writer image module."""

from __future__ import annotations

import dask.array as da
import xarray as xr

from satpy.writers.core.image import ImageWriter


class _RecordingImageWriter(ImageWriter):
    def __init__(self, **kwargs):
        super().__init__(name="test", config_files=[], **kwargs)
        self.img = None

    def save_image(self, img, **kwargs):
        self.img = img
        return [None]


def test_save_dataset_preserves_p_mode():
    """Test that saving a paletted dataset preserves its P mode and bands dim."""
    arr = xr.DataArray(
        da.from_array([[[1, 2, 3], [4, 5, 6]]]),
        dims=("bands", "y", "x"),
        coords={"bands": ["P"]},
        attrs={"mode": "P", "name": "test"},
    )
    writer = _RecordingImageWriter(enhance=False)
    writer.save_dataset(arr)
    assert writer.img is not None
    assert writer.img.mode == "P"
    assert list(writer.img.data.coords["bands"].values) == ["P"]


def test_save_dataset_1x1_preserves_spatial_dims():
    """Test that saving a 1x1 dataset does not squeeze out y or x dimensions."""
    arr = xr.DataArray(
        da.from_array([[1.0]]),
        dims=("y", "x"),
        attrs={"name": "test"},
    )
    writer = _RecordingImageWriter(enhance=False)
    writer.save_dataset(arr)
    assert writer.img is not None
    assert writer.img.data.dims == ("bands", "y", "x")
    assert writer.img.data.shape == (1, 1, 1)
