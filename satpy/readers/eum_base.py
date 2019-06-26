#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 Satpy developers
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

# Copyright (c) 2017-2018 PyTroll Community

# Author(s):

#   Sauli Joro <sauli.joro@eumetsat.int>

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
"""Utilities for EUMETSAT satellite data (HRIT/NATIVE)
"""

from datetime import datetime, timedelta
import numpy as np

# 6 bytes, 8 bytes, 10 bytes
time_cds_short = [('Days', '>u2'), ('Milliseconds', '>u4')]
time_cds = time_cds_short + [('Microseconds', '>u2')]
time_cds_expanded = time_cds + [('Nanoseconds', '>u2')]


def timecds2datetime(tcds):
    """Method for converting time_cds-variables to datetime-objectsself.
    Works both with a dictionary and a numpy record_array.
    """

    days = int(tcds['Days'])
    milliseconds = int(tcds['Milliseconds'])
    try:
        microseconds = int(tcds['Microseconds'])
    except (KeyError, ValueError):
        microseconds = 0
    try:
        microseconds += int(tcds['Nanoseconds']) / 1000.
    except (KeyError, ValueError):
        pass

    reference = datetime(1958, 1, 1)
    delta = timedelta(days=days, milliseconds=milliseconds,
                      microseconds=microseconds)

    return reference + delta


def recarray2dict(arr):

    res = {}
    tcds_types = [time_cds_short, time_cds, time_cds_expanded]

    for dtuple in arr.dtype.descr:
        key = dtuple[0]
        ntype = dtuple[1]
        data = arr[key]
        if ntype in tcds_types:
            if data.size > 1:
                res[key] = np.array([timecds2datetime(item)
                                     for item in data.ravel()]).reshape(data.shape)
            else:
                res[key] = timecds2datetime(data)
        elif isinstance(ntype, list):
            res[key] = recarray2dict(data)
        else:
            if data.size == 1:
                data = data[0]
                if ntype[:2] == '|S':
                    # Python2 and Python3 handle strings differently
                    try:
                        data = data.decode()
                    except ValueError:
                        pass
                    data = data.split(':')[0].strip()
                res[key] = data
            else:
                res[key] = data.squeeze()

    return res
