#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Satpy developers
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

import logging
import scipy.interpolate as si
import numpy as np

# PLANCK COEFFICIENTS FOR CALIBRATION AS DEFINED BY EUMETSAT
C1 = 1.191062e+8   # [W/m2·sr-1·µm4]
C2 = 1.4387863e+4  # [K·µm]

# CONSTANTS DEFINING THE TIE POINTS
TIE_POINTS_FACTOR = 8    # Sub-sampling factor of tie points wrt pixel points
SCAN_ALT_TIE_POINTS = 4  # Number of tie points along the satellite track for each scan

# MEAN EARTH RADIUS AS DEFINED BY IUGG
MEAN_EARTH_RADIUS = 6371008.7714  # [m]

logger = logging.getLogger(__name__)


def tie_points_interpolation(data_on_tie_points):
    """Interpolate the data from the tie points to the pixel points.

    Args:
        data_on_tie_points: numpy ndarray containing the values defined on the tie points.

    Returns:
        ndarray: array containing the interpolated values on the pixel points.

    """
    # Extract the dimensions of the tie points array across and along track
    n_tie_act, n_tie_alt = data_on_tie_points.shape

    # Check that the number of tie points along track is multiple of the number of tie points per scan
    if n_tie_alt % SCAN_ALT_TIE_POINTS != 0:
        logger.warning("The number of tie points in the along-route dimension must be a multiple of %d",
                       SCAN_ALT_TIE_POINTS)
        raise ValueError("The number of tie points in the along-route dimension must be a multiple of %d",
                         SCAN_ALT_TIE_POINTS)

    # Compute the number of scans
    n_scans = n_tie_alt // SCAN_ALT_TIE_POINTS

    # Compute the dimensions of the pixel points array across and along track (total and per scan)
    n_pixel_act = (n_tie_act - 1) * TIE_POINTS_FACTOR
    n_pixel_alt = n_scans * (SCAN_ALT_TIE_POINTS - 1) * TIE_POINTS_FACTOR
    n_pixel_alt_per_scan = (SCAN_ALT_TIE_POINTS - 1) * TIE_POINTS_FACTOR

    # Create the output array with the correct dimensions
    data_on_pixel_points = np.empty((n_pixel_act, n_pixel_alt))

    # Create the array for interpolation
    tie_grid_act = np.arange(0, n_pixel_act + 1, TIE_POINTS_FACTOR)
    tie_grid_alt = np.arange(0, n_pixel_alt_per_scan + 1, TIE_POINTS_FACTOR)
    pixels_grid_act = np.arange(0, n_pixel_act + 1)
    pixels_grid_alt = np.arange(0, n_pixel_alt_per_scan + 1)

    # Interpolate separately for each scan
    for i_scan in range(n_scans):

        # Select the tie points for the current scan
        tie_points_for_scan = range(i_scan * SCAN_ALT_TIE_POINTS, (i_scan + 1) * SCAN_ALT_TIE_POINTS)
        data_on_tie_points_for_scan = data_on_tie_points[:, tie_points_for_scan]

        # Unwrap the data to remove possible discontinuities between degrees (360->0 or 180->-180)
        # Note that the values are not wrapped again as they were originally,
        # therefore the user must explicitly bring the returned values to the desired range if needed
        data_unwrapped = np.degrees(np.unwrap(np.unwrap(np.radians(data_on_tie_points_for_scan), axis=1), axis=0))

        # Interpolate
        fun = si.interp2d(tie_grid_alt, tie_grid_act, data_unwrapped, kind='linear')
        data_on_pixel_points_for_scan = fun(pixels_grid_alt, pixels_grid_act)

        # Append the results to the output array
        pixel_points_for_scan = range(i_scan * n_pixel_alt_per_scan, (i_scan + 1) * n_pixel_alt_per_scan)
        data_on_pixel_points[:, pixel_points_for_scan] = data_on_pixel_points_for_scan[:-1, :-1]

    return data_on_pixel_points
