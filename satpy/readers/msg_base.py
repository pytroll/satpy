#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017, 2018 Adam.Dybbroe

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

import dask.array as da
import xarray.ufuncs as xu

C1 = 1.19104273e-5
C2 = 1.43877523

# CHANNEL_LIST = ['VIS006', 'VIS008', 'IR_016', 'IR_039',
#                'WV_062', 'WV_073', 'IR_087', 'IR_097',
#                'IR_108', 'IR_120', 'IR_134', 'HRV']
CHANNEL_NAMES = {1: "VIS006",
                 2: "VIS008",
                 3: "IR_016",
                 4: "IR_039",
                 5: "WV_062",
                 6: "WV_073",
                 7: "IR_087",
                 8: "IR_097",
                 9: "IR_108",
                 10: "IR_120",
                 11: "IR_134",
                 12: "HRV"}


def get_cds_time(days, msecs):
    """Get the datetime object of the time since epoch given in days and
    milliseconds of day
    """
    return datetime(1958, 1, 1) + timedelta(days=float(days),
                                            milliseconds=float(msecs))

def dec10216(inbuf):
    """Decode 10 bits data into 16 bits words.

    ::

        /*
         * pack 4 10-bit words in 5 bytes into 4 16-bit words
         *
         * 0       1       2       3       4       5
         * 01234567890123456789012345678901234567890
         * 0         1         2         3         4
         */
        ip = &in_buffer[i];
        op = &out_buffer[j];
        op[0] = ip[0]*4 + ip[1]/64;
        op[1] = (ip[1] & 0x3F)*16 + ip[2]/16;
        op[2] = (ip[2] & 0x0F)*64 + ip[3]/4;
        op[3] = (ip[3] & 0x03)*256 +ip[4];

    """
    arr10 = inbuf.astype(np.uint16)
    arr16_len = int(len(arr10) * 4 / 5)
    arr10_len = int((arr16_len * 5) / 4)
    arr10 = arr10[:arr10_len]  # adjust size

    # dask is slow with indexing
    arr10_0 = arr10[::5]
    arr10_1 = arr10[1::5]
    arr10_2 = arr10[2::5]
    arr10_3 = arr10[3::5]
    arr10_4 = arr10[4::5]

    arr16_0 = (arr10_0 << 2) + (arr10_1 >> 6)
    arr16_1 = ((arr10_1 & 63) << 4) + (arr10_2 >> 4)
    arr16_2 = ((arr10_2 & 15) << 6) + (arr10_3 >> 2)
    arr16_3 = ((arr10_3 & 3) << 8) + arr10_4
    arr16 = da.stack([arr16_0, arr16_1, arr16_2, arr16_3], axis=-1).ravel()
    arr16 = da.rechunk(arr16, arr16.shape[0])

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
