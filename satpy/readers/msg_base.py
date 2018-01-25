#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017 Adam.Dybbroe

# Author(s):

#   Adam.Dybbroe <a000680@c20671.ad.smhi.se>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Utilities and eventually also base classes for MSG HRIT/Native data reading
"""

from datetime import datetime, timedelta
import numpy as np
import xarray.ufuncs as xu

C1 = 1.19104273e-5
C2 = 1.43877523

def get_cds_time(days, msecs):
    """Get the datetime object of the time since epoch given in days and
    milliseconds of day
    """
    return datetime(1958, 1, 1) + timedelta(days=float(days),
                                            milliseconds=float(msecs))


def dec10216(data):
    """Unpacking the 10 bit data to 16 bit"""

    arr10 = data.astype(np.uint16).flat
    new_shape = list(data.shape[:-1]) + [(data.shape[-1] * 8) / 10]
    new_shape = [int(s) for s in new_shape]
    arr16 = np.zeros(new_shape, dtype=np.uint16)
    arr16.flat[::4] = np.left_shift(arr10[::5], 2) + \
        np.right_shift((arr10[1::5]), 6)
    arr16.flat[1::4] = np.left_shift((arr10[1::5] & 63), 4) + \
        np.right_shift((arr10[2::5]), 4)
    arr16.flat[2::4] = np.left_shift(arr10[2::5] & 15, 6) + \
        np.right_shift((arr10[3::5]), 2)
    arr16.flat[3::4] = np.left_shift(arr10[3::5] & 3, 8) + \
        arr10[4::5]
    return arr16


def convert_to_radiance(data, gain, offset):
    """Calibrate to radiance."""
    return (data * gain + offset).clip(0.0, None)


def vis_calibrate(data, solar_irradiance):
    return data * 100.0 / solar_irradiance


def tl15(data, wavenumber):
    """Compute the L15 temperature."""
    return ((C2 * wavenumber) /
            xu.log((1.0 / data) * C1 * wavenumber ** 3 + 1.0))


def erads2bt(data, wavenumber, alpha, beta):
    return (tl15(data, wavenumber) - beta) / alpha


def srads2bt(data, wavenumber, a__, b__, c__):
    res = tl15(data, wavenumber)
    return a__ * res * res + b__ * res + c__
