# Copyright (c) 2022- Satpy developers
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
"""Tests for enhancements in enhancements/atmosphere.py."""

import dask.array as da
import numpy as np
import xarray as xr
from trollimage.xrimage import XRImage


def test_essl_moisture():
    """Test ESSL moisture compositor."""
    from satpy.enhancements.atmosphere import essl_moisture

    ratio = xr.DataArray(
        da.from_array(np.full((5, 5), 0.5), chunks=5),
        dims=("y", "x"),
        attrs={"name": "ratio",
               "calibration": "reflectance",
               "units": "%",
               "mode": "L"})
    im = XRImage(ratio)

    essl_moisture(im)
    assert im.data.attrs["mode"] == "RGB"
    np.testing.assert_array_equal(im.data["bands"], ["R", "G", "B"])
