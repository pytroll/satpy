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

"""Utilities for enhancement testing."""

import dask.array as da
import numpy as np
import xarray as xr

from satpy.tests.utils import assert_maximum_dask_computes


def run_and_check_enhancement(func, data, expected, **kwargs):
    """Perform basic checks that apply to multiple tests."""
    pre_attrs = data.attrs
    with assert_maximum_dask_computes(max_computes=0):
        img = _get_enhanced_image(func, data, **kwargs)

    _assert_image(img, pre_attrs, func.__name__, "palettes" in kwargs)
    _assert_image_data(img, expected)


def _get_enhanced_image(func, data, **kwargs):
    from trollimage.xrimage import XRImage

    img = XRImage(data)
    func(img, **kwargs)

    return img


def _assert_image(img, pre_attrs, func_name, has_palette):
    assert isinstance(img.data, xr.DataArray)
    assert isinstance(img.data.data, da.Array)

    old_keys = set(pre_attrs.keys())
    # It is OK to have "enhancement_history" added
    new_keys = set(img.data.attrs.keys()) - {"enhancement_history"}
    # In case of palettes are used, _FillValue is added.
    # Colorize doesn't add the fill value, so ignore that
    if has_palette and func_name != "colorize":
        assert "_FillValue" in new_keys
        # Remove it from further comparisons
        new_keys = new_keys - {"_FillValue"}
    assert old_keys == new_keys


def _assert_image_data(img, expected, dtype=None):
    # Compute the data to mimic what xrimage geotiff writing does
    res_data = img.data.data.compute()
    assert not isinstance(res_data, da.Array)
    np.testing.assert_allclose(res_data, expected, atol=1.e-6, rtol=0)
    if dtype:
        assert img.data.dtype == dtype
        assert res_data.dtype == dtype


def run_and_check_enhancement_with_dtype(func, data, expected, **kwargs):
    """Perform basic checks that apply to multiple tests."""
    pre_attrs = data.attrs
    img = _get_enhanced_image(func, data, **kwargs)

    _assert_image(img, pre_attrs, func.__name__, "palettes" in kwargs)
    _assert_image_data(img, expected, dtype=data.dtype)


def _create_data():
    data = np.arange(-210, 790, 100).reshape((2, 5)) * 0.95
    data[0, 0] = np.nan  # one bad value for testing

    return data


def create_ch1():
    """Create single channel data."""
    data = _create_data()
    return xr.DataArray(da.from_array(data, chunks=2), dims=("y", "x"), attrs={"test": "test"})


def _create_crefl_data():
    crefl_data = np.arange(-210, 790, 100).reshape((2, 5)) * 0.95
    crefl_data /= 5.605
    crefl_data[0, 0] = np.nan  # one bad value for testing
    crefl_data[0, 1] = 0.

    return crefl_data


def create_ch2():
    """Create test data for crefl."""
    crefl_data = _create_crefl_data()
    return xr.DataArray(da.from_array(crefl_data, chunks=2), dims=("y", "x"), attrs={"test": "test"})


def create_rgb():
    """Create RGB test data."""
    data = _create_data()
    rgb_data = np.stack([data, data, data])
    return xr.DataArray(da.from_array(rgb_data, chunks=(3, 2, 2)),
                        dims=("bands", "y", "x"),
                                coords={"bands": ["R", "G", "B"]})
