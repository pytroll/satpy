#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 Satpy developers
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
"""Common functionality for GOES Imager readers."""

SPACECRAFTS = {
    # these are GP_SC_ID
    18007: "GOES-7",
    18008: "GOES-8",
    18009: "GOES-9",
    18010: "GOES-10",
    18011: "GOES-11",
    18012: "GOES-12",
    18013: "GOES-13",
    18014: "GOES-14",
    18015: "GOES-15",
    # these are in block-0
    7: "GOES-7",
    8: "GOES-8",
    9: "GOES-9",
    10: "GOES-10",
    11: "GOES-11",
    12: "GOES-12",
    13: "GOES-13",
    14: "GOES-14",
    15: "GOES-15"}
VISSR = "VISSR"
IMAGER_8_11 = "IMAGER (GOES 8-11)"
IMAGER_12_15 = "IMAGER (GOES 12-15)"
INSTRUMENTS = {
    "GOES-7": VISSR,
    "GOES-8": IMAGER_8_11,
    "GOES-9": IMAGER_8_11,
    "GOES-10": IMAGER_8_11,
    "GOES-11": IMAGER_8_11,
    "GOES-12": IMAGER_12_15,
    "GOES-13": IMAGER_12_15,
    "GOES-14": IMAGER_12_15,
    "GOES-15": IMAGER_12_15,
}

# Geometric constants [meters]
EQUATOR_RADIUS = 6378169.00
POLE_RADIUS = 6356583.80
ALTITUDE = 35785831.00
