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
"""Enhancements related to visualising atmospheric phenomena."""

import datetime

import dask.array as da
import xarray as xr


def essl_moisture(img, low=1.1, high=1.6) -> None:
    """Low level moisture by European Severe Storms Laboratory (ESSL).

    Must be passed exactly two projectables.  The first one should correspond
    to a channel at around 0.86 µm, the second one at 0.91 µm.

    This composite and its colorisation was developed by ESSL.
    """
    ratio = img.data
    if _is_fci_test_data(img.data):
        # Due to a bug in the FCI pre-launch simulated test data,
        # the 0.86 µm channel is too bright.  To correct for this, its
        # reflectances should be multiplied by 0.8.
        ratio *= 0.8

    with xr.set_options(keep_attrs=True):
        ratio = _scale_and_clip(ratio, low, high)
        red = _calc_essl_red(ratio)
        green = _calc_essl_green(ratio)
        blue = _calc_essl_blue(ratio)
        data = xr.concat([red, green, blue], dim="bands")
        data.attrs["mode"] = "RGB"
        data["bands"] = ["R", "G", "B"]
    img.data = data


def _scale_and_clip(ratio, low, high):
    """Scale ratio values to [0, 1] and clip values outside this range."""
    scaled = (ratio - low) / (high - low)
    scaled.data = da.clip(scaled.data, 0, 1)
    return scaled


def _calc_essl_red(ratio):
    """Calculate values for red based on scaled and clipped ratio."""
    red_a = 1.375 - 2.67 * ratio
    red_b = -0.75 + ratio
    red = xr.where(red_a > red_b, red_a, red_b)
    red.data = da.clip(red.data, 0, 1)
    return red


def _calc_essl_green(ratio):
    """Calculate values for green based on scaled and clipped ratio."""
    green = 1 - (8/7) * ratio
    green.data = da.clip(green.data, 0, 1)
    return green


def _calc_essl_blue(ratio):
    """Calculate values for blue based on scaled and clipped ratio."""
    blue_a = 0.75 - 1.5 * ratio
    blue_b = 0.25 - (ratio - 0.75)**2
    blue = xr.where(blue_a > blue_b, blue_a, blue_b)
    blue.data = da.clip(blue.data, 0, 1)
    return blue


def _is_fci_test_data(data):
    """Check if we are working with FCI test data."""
    return (data.attrs["sensor"] == "fci" and
            data.attrs["start_time"] < datetime.datetime(2022, 11, 30))
