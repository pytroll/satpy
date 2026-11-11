"""Implementation of some image filters."""

import logging

import dask.array as da
import numpy as np
import xarray as xr

from satpy.modifiers.filters import Median


def test_median(caplog):
    """Test the median filter modifier."""
    caplog.set_level(logging.DEBUG)
    dims = "y", "x"
    coordinates = dict(x=np.arange(6), y=np.arange(6))
    attrs = dict(units="K")
    median_filter_params = dict(size=3)
    name = "median_filter"
    median_filter = Median(median_filter_params, name=name)
    array = xr.DataArray(da.arange(36).reshape((6, 6)), coords=coordinates, dims=dims, attrs=attrs)
    res = median_filter([array])
    filtered_array = np.array([[1, 2, 3, 4, 5, 5],
                               [6, 7, 8, 9, 10, 11],
                               [12, 13, 14, 15, 16, 17],
                               [18, 19, 20, 21, 22, 23],
                               [24, 25, 26, 27, 28, 29],
                               [30, 30, 31, 32, 33, 34]])
    np.testing.assert_allclose(res, filtered_array)
    assert res.dims == dims
    assert attrs.items() <= res.attrs.items()
    assert res.attrs["name"] == name
    np.testing.assert_equal(res.coords["x"], coordinates["x"])
    np.testing.assert_equal(res.coords["y"], coordinates["y"])
    assert "Apply median filtering with parameters {'size': 3}" in caplog.text
