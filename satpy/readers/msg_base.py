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


def get_cds_time(days, msecs):
    """Get the datetime object of the time since epoch given in days and
    milliseconds of day
    """
    return datetime(1958, 1, 1) + timedelta(days=float(days),
                                            milliseconds=float(msecs))
