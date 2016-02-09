#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2012 SMHI

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

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

"""Odim format reader.
"""

filename = "/data/temp/Martin.Raspaud/metop02_20120329_0043_28234_satproj_00000_12959_avhrr.h5"


import h5py

def get_lonlats(filename):


    f__ = h5py.File(filename, "r")

    lat_g = f__["where"]["lat"]["what"].attrs["gain"]
    lat_o = f__["where"]["lat"]["what"].attrs["offset"]
    lat = f__["where"]["lat"]["data"][:] * latg + lato

    lon_g = f__["where"]["lon"]["what"].attrs["gain"]
    lon_o = f__["where"]["lon"]["what"].attrs["offset"]
    lon = f__["where"]["lon"]["data"][:] * lon_g + lon_o

    return lon, lat
