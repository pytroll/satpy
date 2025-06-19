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

"""Helpers for enhancement testing."""

import dask.array as da
import numpy as np
import xarray as xr


def run_and_check_enhancement(func, data, expected, **kwargs):
    """Perform basic checks that apply to multiple tests."""
    pre_attrs = data.attrs
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


def identical_decorator(func):
    """Decorate but do nothing."""
    return func
