# Copyright (c) 2017-2025 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Enhancements based on convolution."""

from __future__ import annotations

import logging

import dask
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

    delay = dask.delayed(_three_d_effect_delayed)(band_data, kernel, mode)
    new_data = da.from_delayed(delay, shape=band_data.shape, dtype=band_data.dtype)
    return new_data


def _three_d_effect_delayed(band_data, kernel, mode):
    """Kernel for running delayed 3D effect creation."""
    from scipy.signal import convolve2d
    band_data = band_data.reshape(band_data.shape[1:])
    new_data = convolve2d(band_data, kernel, mode=mode)
    return new_data.reshape((1, band_data.shape[0], band_data.shape[1]))
