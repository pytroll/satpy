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

"""Unit testing the enhancement decoration functions."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from satpy.enhancements.wrappers import on_dask_array, on_separate_bands, using_map_blocks

from .utils import create_ch1, create_ch2, create_rgb, run_and_check_enhancement


def identical_decorator(func):
    """Decorate but do nothing."""
    return func


class TestEnhancementsConvolution:
    """Class for testing decorators in satpy.enhancements.wrappers module."""

    def setup_method(self):
        """Create test data used by every test."""
        self.ch1 = create_ch1()
        self.ch2 = create_ch2()
        self.rgb = create_rgb()

    @pytest.mark.parametrize(
        ("decorator", "exp_call_cls"),
        [
            (identical_decorator, xr.DataArray),
            (on_dask_array, da.Array),
            (using_map_blocks, np.ndarray),
        ],
    )
    @pytest.mark.parametrize("input_data_name", ["ch1", "ch2", "rgb"])
    def test_enhancement_decorators(self, input_data_name, decorator, exp_call_cls):
        """Test the utility decorators."""

        def _enh_func(img):
            def _calc_func(data):
                assert isinstance(data, exp_call_cls)
                return data

            decorated_func = decorator(_calc_func)
            return decorated_func(img.data)

        in_data = getattr(self, input_data_name)
        exp_data = in_data.values
        if "bands" not in in_data.coords:
            exp_data = exp_data[np.newaxis, :, :]
        run_and_check_enhancement(_enh_func, in_data, exp_data)

    def tearDown(self):
        """Clean up."""


def test_on_separate_bands():
    """Test the `on_separate_bands` decorator."""

    def func(array, index, gain=2):
        return xr.DataArray(np.ones(array.shape, dtype=array.dtype) * index * gain,
                            coords=array.coords, dims=array.dims, attrs=array.attrs)

    separate_func = on_separate_bands(func)
    arr = xr.DataArray(np.zeros((3, 10, 10)), dims=["bands", "y", "x"], coords={"bands": ["R", "G", "B"]})
    assert separate_func(arr).shape == arr.shape
    assert all(separate_func(arr, gain=1).values[:, 0, 0] == [0, 1, 2])


def test_using_map_blocks():
    """Test the `using_map_blocks` decorator."""

    def func(np_array, block_info=None):
        value = block_info[0]["chunk-location"][-1]
        return np.ones(np_array.shape) * value

    map_blocked_func = using_map_blocks(func)
    arr = xr.DataArray(da.zeros((3, 10, 10), dtype=int, chunks=5), dims=["bands", "y", "x"])
    res = map_blocked_func(arr)
    assert res.shape == arr.shape
    assert res[0, 0, 0].compute() != res[0, 9, 9].compute()


def test_on_dask_array():
    """Test the `on_dask_array` decorator."""

    def func(dask_array):
        if not isinstance(dask_array, da.core.Array):
            pytest.fail("Array is not a dask array")
        return dask_array

    dask_func = on_dask_array(func)
    arr = xr.DataArray(da.zeros((3, 10, 10), dtype=int, chunks=5), dims=["bands", "y", "x"])
    res = dask_func(arr)
    assert res.shape == arr.shape
