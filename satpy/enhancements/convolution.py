"""Enhancements based on convolution."""

from __future__ import annotations

import logging

import dask.array as da
import numpy as np

from .wrappers import exclude_alpha, on_dask_array, on_separate_bands

LOG = logging.getLogger(__name__)


def three_d_effect(img, **kwargs):
    """Create 3D effect using convolution."""
    w = kwargs.get("weight", 1)
    LOG.debug("Applying 3D effect with weight %.2f", w)
    kernel = np.array([[-w, 0, w],
                       [-w, 1, w],
                       [-w, 0, w]])
    mode = kwargs.get("convolve_mode", "same")
    return _three_d_effect(img.data, kernel=kernel, mode=mode)


@exclude_alpha
@on_separate_bands
@on_dask_array
def _three_d_effect(band_data, kernel=None, mode=None, index=None):
    del index

    new_data = da.map_blocks(
        _three_d_effect_numpy,
        band_data.rechunk(band_data.shape),
        kernel,
        mode,
        dtype=band_data.dtype,
        meta=np.ndarray((), dtype=band_data.dtype),
    )
    return new_data.rechunk(band_data.chunks)


def _three_d_effect_numpy(band_data, kernel, mode):
    """Kernel for running delayed 3D effect creation."""
    from scipy.signal import convolve2d
    band_data = band_data.reshape(band_data.shape[1:])
    new_data = convolve2d(band_data, kernel, mode=mode)
    return new_data.reshape((1, band_data.shape[0], band_data.shape[1]))
