# Copyright (c) 2025 Satpy developers
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

"""Testing of writer helper functions."""

import datetime

import numpy as np
import pytest
import xarray as xr


def test_get_valid_time():
    """Test extracting valid time from a dataset."""
    from satpy.writers.core.utils import get_valid_time

    da = xr.DataArray(
            np.zeros((3, 3)),
            name="Čuonajóhka",
            dims=("y", "x"),
            coords={"time":
                    xr.DataArray(
                        np.full((3, 3), 2),
                        dims=("y", "x"),
                        attrs={"units": "seconds since 2222-02-02T22:22:20"})})

    assert get_valid_time(da) == datetime.datetime(2222, 2, 2, 22, 22, 22)

    da = xr.DataArray(np.zeros((3, 3)), dims=("y", "x"), name="Liedik")
    with pytest.raises(ValueError,
                       match="Dataset Liedik has no time coordinate."):
        get_valid_time(da)

    da = xr.DataArray(np.zeros((3, 3)), dims=("y", "x"), name="Liedik", attrs={"name": "Gámasčearru"})
    with pytest.raises(ValueError,
                       match="Dataset Gámasčearru has no time coordinate."):
        get_valid_time(da)
