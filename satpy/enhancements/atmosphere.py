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
    r"""Low level moisture by European Severe Storms Laboratory (ESSL).

    Expects a mode L image with data corresponding to the ratio of the
    calibrated reflectances for the 0.86 µm and 0.906 µm channel.

    This composite and its colorisation were developed by ESSL.

    Ratio values are scaled from the range ``[low, high]``, which is by default
    between 1.1 and 1.6, but might be tuned based on region or sensor,
    to ``[0, 1]``.  Values outside this range are clipped.  Color values
    for red, green, and blue are calculated as follows, where ``x`` is the
    ratio between the 0.86 µm and 0.905 µm channels:

    .. math::

        R = \max(1.375 - 2.67 x, -0.75 + x) \\
        G = 1 - \frac{8x}{7} \\
        B = \max(0.75 - 1.5 x, 0.25 - (x - 0.75)^2) \\

    The value of ``img.data`` is modified in-place.

    A color interpretation guide is pending further adjustments to the
    parameters for current and future sensors.

    Args:
        img: XRImage containing the relevant composite
        low: optional, low end for scaling, defaults to 1.1
        high: optional, high end for scaling, defaults to 1.6
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
    return ("sensor" in data.attrs and
            "start_time" in data.attrs and
            data.attrs["sensor"] == "fci" and
            isinstance(data.attrs["start_time"], datetime.datetime) and
            data.attrs["start_time"] < datetime.datetime(2022, 11, 30))
