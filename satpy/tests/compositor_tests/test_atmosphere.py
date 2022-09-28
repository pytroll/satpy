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
"""Tests for compositors in composites/atmosphere.py."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pyresample import create_area_def


@pytest.fixture
def area():
    """Return fake area."""
    return create_area_def(
        "fribullus_xax",
        4087,
        units="m",
        resolution=1000,
        center=(0, 0),
        shape=(5, 5))


def test_essl_moisture(area):
    """Test ESSL moisture compositor."""
    from satpy.composites.atmosphere import ESSLMoisture
    compositor = ESSLMoisture(name="essl_moisture")

    nir_08 = xr.DataArray(
        da.from_array(np.full(area.shape, 0.5), chunks=5),
        dims=("y", "x"),
        attrs={"name": "nir_08",
               "calibration": "reflectance",
               "units": "%",
               "area": area})

    nir_09 = xr.DataArray(
        da.from_array(np.full(area.shape, 0.3), chunks=5),
        dims=("y", "x"),
        attrs={"name": "nir_09",
               "calibration": "reflectance",
               "units": "%",
               "area": area})

    res = compositor([nir_08, nir_09])
    assert "area" in res.attrs
    res.compute()
