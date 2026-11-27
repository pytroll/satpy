
"""Unit testing for AHI-specific enhancements functions."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr


class TestTCREnhancement:
    """Test the AHI enhancement functions."""

    def setup_method(self):
        """Create test data."""
        data = da.arange(-100, 1000, 110).reshape(2, 5)
        rgb_data = np.stack([data, data, data])
        self.rgb = xr.DataArray(rgb_data, dims=("bands", "y", "x"),
                                coords={"bands": ["R", "G", "B"]},
                                attrs={"platform_name": "Himawari-8"})

    def test_jma_true_color_reproduction(self):
        """Test the jma_true_color_reproduction enhancement."""
        from trollimage.xrimage import XRImage

        from satpy.enhancements.ahi import jma_true_color_reproduction

        expected = [[[-108.260, 10.826, 129.912, 248.998, 368.084],
                     [487.170, 606.256, 725.342, 844.428, 963.514]],

                    [[ -98.170, 9.817, 117.804, 225.791, 333.778],
                     [441.765, 549.752, 657.739, 765.726, 873.713]],

                    [[-93.800, 9.380, 112.560, 215.740, 318.920],
                     [422.100, 525.280, 628.460, 731.640, 834.820]]]

        img = XRImage(self.rgb)
        jma_true_color_reproduction(img)

        np.testing.assert_almost_equal(img.data.compute(), expected)

        self.rgb.attrs["platform_name"] = None
        img = XRImage(self.rgb)
        with pytest.raises(ValueError, match="Missing platform name."):
            jma_true_color_reproduction(img)

        self.rgb.attrs["platform_name"] = "Fakesat"
        img = XRImage(self.rgb)
        with pytest.raises(KeyError, match="No conversion matrix found for platform Fakesat"):
            jma_true_color_reproduction(img)
