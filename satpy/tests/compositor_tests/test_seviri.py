
"""Tests for compositors."""

import dask
import dask.array as da
import numpy as np
import xarray as xr

from satpy.tests.utils import CustomScheduler


class TestRealisticColors:
    """Test the SEVIRI Realistic Colors compositor."""

    def test_realistic_colors(self):
        """Test the compositor."""
        from satpy.composites.seviri import RealisticColors

        vis06 = xr.DataArray(da.arange(0, 15, dtype=np.float32).reshape(3, 5), dims=("y", "x"),
                             attrs={"foo": "foo"})
        vis08 = xr.DataArray(da.arange(15, 0, -1, dtype=np.float32).reshape(3, 5), dims=("y", "x"),
                             attrs={"bar": "bar"})
        hrv = xr.DataArray(6 * da.ones((3, 5), dtype=np.float32), dims=("y", "x"),
                           attrs={"baz": "baz"})

        expected_red = np.array([[0.0, 2.733333, 4.9333334, 6.6, 7.733333],
                                 [8.333333, 8.400001, 7.9333334, 7.0, 6.0],
                                 [5.0, 4.0, 3.0, 2.0, 1.0]], dtype=np.float32)
        expected_green = np.array([
            [15.0, 12.266666, 10.066668, 8.400001, 7.2666664],
            [6.6666665, 6.6000004, 7.0666666, 8.0, 9.0],
            [10.0, 11.0, 12.0, 13.0, 14.0]], dtype=np.float32)

        with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
            comp = RealisticColors("Ni!")
            res = comp((vis06, vis08, hrv))

        arr = res.values

        assert res.dtype == np.float32
        np.testing.assert_allclose(arr[0, :, :], expected_red)
        np.testing.assert_allclose(arr[1, :, :], expected_green)
        np.testing.assert_allclose(arr[2, :, :], 3.0)
