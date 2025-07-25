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
"""Utilities shared between writers."""

from xarray.coding.times import decode_cf_datetime


def get_valid_time(dataset):
    """Get the valid time for a dataset.

    If a dataset has time coordinates, get the arithmetic mean time as a
    representative time for the dataset.

    Args:
        dataset: xarray dataarray with time coordinates

    Returns:
        datetime.datetime object with arithmetic mean time for dataset
    """
    if "time" not in dataset.coords:
        raise ValueError(
            "Dataset {dataset.attrs['name']:s} has no time coordinate. "
            "No valid time can be calculated.  To track valid time, "
            "pass `reader_kwargs = {'track_time': True}` to `Scene.__init__` "
            "for a supported reader and, if applicable, `resample_coords=True` "
            "to `Scene.resample`.")

    tm = dataset.coords["time"].mean()
    return decode_cf_datetime(tm, dataset.coords["time"].attrs["units"])
