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

        expected = [[[-109.93, 10.993, 131.916, 252.839, 373.762],
                     [494.685, 615.608, 736.531, 857.454, 978.377]],

                    [[-97.73, 9.773, 117.276, 224.779, 332.282],
                     [439.785, 547.288, 654.791, 762.294, 869.797]],

                    [[-93.29, 9.329, 111.948, 214.567, 317.186],
                     [419.805, 522.424, 625.043, 727.662, 830.281]]]

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
