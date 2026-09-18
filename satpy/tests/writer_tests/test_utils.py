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


def test_get_mean_time():
    """Test extracting valid time from a dataset."""
    from satpy.writers.core.utils import get_mean_time

    xrda = xr.DataArray(
            np.zeros((3, 3)),
            name="Čuonajóhka",
            dims=("y", "x"),
            coords={"time":
                    xr.DataArray(
                        np.full((3, 3), 2),
                        dims=("y", "x"),
                        attrs={"units": "seconds since 2222-02-02T22:22:20"})})

    assert get_mean_time(xrda) == datetime.datetime(2222, 2, 2, 22, 22, 22)


def test_get_mean_time_no_time_coordinate_dataset_name():
    """Test raising of ValueError in the absence of a time coordinate.

    Takes name from dataset.
    """
    from satpy.writers.core.utils import get_mean_time

    xrda = xr.DataArray(np.zeros((3, 3)), dims=("y", "x"), name="Liedik")
    with pytest.raises(ValueError,
                       match="Dataset Liedik has no time coordinate."):
        get_mean_time(xrda)


def test_get_mean_time_no_time_coordinate_attribute_name():
    """Test raising of ValueError in the absence of a time coordinate.

    Takes name from dataset attribute (more common in satpy).
    """
    from satpy.writers.core.utils import get_mean_time
    xrda = xr.DataArray(np.zeros((3, 3)), dims=("y", "x"), name="Liedik", attrs={"name": "Gámasčearru"})
    with pytest.raises(ValueError,
                       match="Dataset Gámasčearru has no time coordinate."):
        get_mean_time(xrda)
