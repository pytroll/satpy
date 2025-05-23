#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Satpy developers
#
# satpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# satpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with satpy.  If not, see <http://www.gnu.org/licenses/>.

"""Utilities for the management of VII products."""


# PLANCK COEFFICIENTS FOR CALIBRATION AS DEFINED BY EUMETSAT
C1 = 1.191062e+8   # [W/m2·sr-1·µm4]
C2 = 1.4387863e+4  # [K·µm]

# CONSTANTS DEFINING THE TIE POINTS
TIE_POINTS_FACTOR = 8    # Sub-sampling factor of tie points wrt pixel points
SCAN_ALT_TIE_POINTS = 4  # Number of tie points along the satellite track for each scan

# MEAN EARTH RADIUS AS DEFINED BY IUGG
MEAN_EARTH_RADIUS = 6371008.7714  # [m]
