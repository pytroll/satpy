"""Unit testing for the ABI enhancement functions."""

import unittest

import dask.array as da
import numpy as np
import xarray as xr


class TestABIEnhancement(unittest.TestCase):
    """Test the ABI enhancement functions."""

    def setUp(self):
        """Create fake data for the tests."""
        data = da.linspace(0, 1, 16).reshape((4, 4))
        self.da = xr.DataArray(data, dims=("y", "x"), attrs={"test": "test"})

    def test_cimss_true_color_contrast(self):
        """Test the cimss_true_color_contrast enhancement."""
        from trollimage.xrimage import XRImage

        from satpy.enhancements.abi import cimss_true_color_contrast

        expected = np.array([[
            [0., 0., 0.05261956, 0.13396146],
            [0.21530335, 0.29664525, 0.37798715, 0.45932905],
            [0.54067095, 0.62201285, 0.70335475, 0.78469665],
            [0.86603854, 0.94738044, 1., 1.],
            ]])
        img = XRImage(self.da)
        cimss_true_color_contrast(img)
        np.testing.assert_almost_equal(img.data.compute(), expected)
