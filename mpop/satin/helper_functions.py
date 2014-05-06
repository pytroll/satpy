
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2014.
# 
# Author(s):
#  
#   Panu Lahtinen <panu.lahtinen@fmi.fi>
#   Esben S. Nielsen
#
# This file is part of mpop.
# 
# mpop is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# 
# mpop is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along with
# mpop.  If not, see <http://www.gnu.org/licenses/>.

'''Helper functions for area extent and boundary calculations.
'''

import numpy as np
from mpop.projector import get_area_def
from pyresample.geometry import Boundary
import logging
from pyproj import Proj

LOGGER = logging.getLogger(__name__)

def get_area_boundaries(area_def):
    '''Get area boundaries (lon+lat coordinates) from area definition.
    '''
    
    # upper boundary
    lonlat = np.array([area_def.get_lonlat(0, i) \
                           for i in range(area_def.x_size)])
    up_lons = lonlat[:, 0]
    up_lats = lonlat[:, 1]

    # lower boundary
    lonlat = np.array([area_def.get_lonlat(area_def.y_size-1, i) \
                           for i in range(area_def.x_size)])
    down_lons = lonlat[:, 0]
    down_lats = lonlat[:, 1]
    
    # left boundary
    lonlat = np.array([area_def.get_lonlat(i, 0) \
                           for i in range(area_def.y_size)])
    left_lons = lonlat[:, 0]
    left_lats = lonlat[:, 1]
    
    # right boundary
    lonlat = np.array([area_def.get_lonlat(i, area_def.x_size-1) \
                           for i in range(area_def.y_size)])
    right_lons = lonlat[:, 0]
    right_lats = lonlat[:, 1]
    
    return (Boundary(up_lons, right_lons, down_lons, left_lons),
            Boundary(up_lats, right_lats, down_lats, left_lats))


def get_indices_from_boundaries(boundary_lons, boundary_lats, 
                                           lons, lats, radius_of_influence):
    """Find relevant indices from grid boundaries using the 
    winding number theorem"""
    
    valid_index = _get_valid_index(boundary_lons.side1, boundary_lons.side2, 
                                   boundary_lons.side3, boundary_lons.side4,
                                   boundary_lats.side1, boundary_lats.side2, 
                                   boundary_lats.side3, boundary_lats.side4,
                                   lons, lats, radius_of_influence)
    
    return valid_index


def get_angle_sum(lons_side1, lons_side2, lons_side3, lons_side4):
    '''Calculate angle sum for winding number theorem.  Note that all
    the sides need to be connected, that is:

    lons_side[-1] == lons_side2[0], 
    ...
    lons_side4[-1] == lons_side1[0]
    '''
    angle_sum = 0
    for side in (lons_side1, lons_side2, lons_side3, lons_side4):
        side_diff = np.sum(np.diff(side))
        idxs, = np.where(np.abs(side_diff) > 180)
        if idxs:
            side_diff[idxs] = (np.abs(side_diff[idxs])-360) * \
                np.sign(side_diff[idxs])
        angle_sum += np.sum(side_diff)

    return angle_sum
    

def _get_valid_index(lons_side1, lons_side2, lons_side3, lons_side4,
                     lats_side1, lats_side2, lats_side3, lats_side4,
                     lons, lats, radius_of_influence):
    """Find relevant indices from grid boundaries using the 
    winding number theorem"""
    
    earth_radius = 6370997.0

    # Coarse reduction of data based on extrema analysis of the boundary 
    # lon lat values of the target grid
    illegal_lons = (((lons_side1 < -180) | (lons_side1 > 180)).any() or
                    ((lons_side2 < -180) | (lons_side2 > 180)).any() or
                    ((lons_side3 < -180) | (lons_side3 > 180)).any() or
                    ((lons_side4 < -180) | (lons_side4 > 180)).any())
    
    illegal_lats = (((lats_side1 < -90) | (lats_side1 > 90)).any() or
                    ((lats_side2 < -90) | (lats_side2 > 90)).any() or
                    ((lats_side3 < -90) | (lats_side3 > 90)).any() or
                    ((lats_side4 < -90) | (lats_side4 > 90)).any())
    
    if illegal_lons or illegal_lats:
        # Grid boundaries are not safe to operate on
        return np.ones(lons.size, dtype=np.bool)   
    
    # Find sum angle sum of grid boundary
    angle_sum = get_angle_sum(lons_side1, lons_side2, 
                              lons_side3[::-1], lons_side4[::-1])

    # Buffer min and max lon and lat of interest with radius of interest
    lat_min = min(lats_side1.min(), lats_side2.min(), 
                  lats_side3.min(), lats_side4.min())
    lat_min_buffered = lat_min - np.degrees(float(radius_of_influence) / \
                                                earth_radius)
    lat_max = max(lats_side1.max(), lats_side2.max(), lats_side3.max(),
                  lats_side4.max())
    lat_max_buffered = lat_max + np.degrees(float(radius_of_influence) / \
                                                earth_radius)

    max_angle_s2 = max(abs(lats_side2.max()), abs(lats_side2.min()))
    max_angle_s4 = max(abs(lats_side4.max()), abs(lats_side4.min()))
    lon_min_buffered = lons_side4.min() - \
        np.degrees(float(radius_of_influence) / 
                   (np.sin(np.radians(max_angle_s4)) * earth_radius))
                    
    lon_max_buffered = lons_side2.max() + \
        np.degrees(float(radius_of_influence) / 
                   (np.sin(np.radians(max_angle_s2)) * earth_radius))
    
    # From the winding number theorem follows:
    # angle_sum possiblilities:
    # -360: area covers north pole
    #  360: area covers south pole
    #    0: area covers no poles
    # else: area covers both poles    
    if round(angle_sum) == -360:
        LOGGER.debug("Area covers north pole")
        # Covers NP
        valid_index = (lats >= lat_min_buffered)        
    elif round(angle_sum) == 360:
        LOGGER.debug("Area covers south pole")
        # Covers SP
        valid_index = (lats <= lat_max_buffered)        
    elif round(angle_sum) == 0:
        LOGGER.debug("Area covers no poles")
        # Covers no poles
        valid_lats = (lats >= lat_min_buffered) * (lats <= lat_max_buffered)

        if lons_side2.min() > lons_side4.max():
            # No date line crossing                      
            valid_lons = (lons >= lon_min_buffered) * (lons <= lon_max_buffered)
        else:
            # Date line crossing
            seg1 = (lons >= lon_min_buffered) * (lons <= 180)
            seg2 = (lons <= lon_max_buffered) * (lons >= -180)
            valid_lons = seg1 + seg2                        
        
        valid_index = valid_lats * valid_lons        
    else:
        LOGGER.debug("Area covers both poles")
        # Covers both poles, don't reduce
        return True
        # valid_index = np.ones(lons.size, dtype=np.bool)

    return valid_index


def area_def_names_to_extent(area_def_names, proj4_str, 
                             default_extent=(-5567248.07, -5570248.48, 
                                              5570248.48, 5567248.07)):
    '''Convert a list of *area_def_names* to maximal area extent in
    destination projection defined by *proj4_str*. *default_extent*
    gives the extreme values.  Default value is MSG3 extents at
    lat0=0.0.
    '''

    if type(area_def_names) is not list:
        area_def_names = [area_def_names]

    # proj4-ify the projection string
    global_proj4_str = proj4_str.split(' ')
    global_proj4_str = '+' + ' +'.join(global_proj4_str)

    pro = Proj(global_proj4_str)

    maximum_area_extent = None

    for name in area_def_names:

        boundaries = get_area_boundaries(get_area_def(name))

        # extents for edges
        _, up_y = pro(boundaries[0].side1, boundaries[1].side1)
        right_x, _ = pro(boundaries[0].side2, boundaries[1].side2)
        _, down_y = pro(boundaries[0].side3, boundaries[1].side3)
        left_x, _ = pro(boundaries[0].side4, boundaries[1].side4)

        # replace invalid values with NaN
        up_y[np.abs(up_y) > 1e20] = np.nan
        right_x[np.abs(right_x) > 1e20] = np.nan
        down_y[np.abs(down_y) > 1e20] = np.nan
        left_x[np.abs(left_x) > 1e20] = np.nan

        # Get the maximum needed extent from different corners.
        extent = [np.nanmin(left_x), 
                  np.nanmin(down_y),
                  np.nanmax(right_x), 
                  np.nanmax(up_y)]

        # Replace "infinity" values with default extent
        for i in range(4):
            if extent[i] is np.nan:
                extent[i] = default_extent[i]

        # update maximum extent
        if maximum_area_extent is None:
            maximum_area_extent = extent
        else:
            if maximum_area_extent[0] > extent[0]:
                maximum_area_extent[0] = extent[0]
            if maximum_area_extent[1] > extent[1]:
                maximum_area_extent[1] = extent[1]
            if maximum_area_extent[2] < extent[2]:
                maximum_area_extent[2] = extent[2]
            if maximum_area_extent[3] < extent[3]:
                maximum_area_extent[3] = extent[3]

        # Replace "infinity" values with default extent
        for i in range(4):
            if not np.isfinite(maximum_area_extent[i]):
                maximum_area_extent[i] = default_extent[i]


    return maximum_area_extent


def reduce_swath(global_data, area_def_names):
    '''Remove unnecessary areas from the swath data edges.
    '''

    if type(area_def_names) is not list:
        area_def_names = [area_def_names]

    for name in area_def_names:
        area_def = get_area_def(name)
        boundary_lons, boundary_lats = get_area_boundaries(area_def)
        lons, lats, resolution = None, None, None
        global_coordinates = False
        coordinate_data_reduced = False
        idxs = None
        
        for chan in global_data.loaded_channels():
            # if resolution is not known, or it changes,
            # reload the coordinates
            if chan.resolution != resolution:
                resolution = chan.resolution
                idxs = None # re-calculate indices
                try:
                    lons, lats = chan.area.get_lonlats()
                except AttributeError:
                    lons, lats = global_data.area.get_lonlats()
                    global_coordinates = True

            if idxs is None:
                idxs = get_indices_from_boundaries(boundary_lons,
                                                   boundary_lats, 
                                                   lons, lats, 
                                                   resolution)

            if not np.all(idxs):
                if len(idxs) == 0:
                    idxs = np.array([0])
                LOGGER.debug('Reducing data size for channel %s' % \
                                 chan.name)
                # slice the channel data to smaller size
                global_data[chan.name].data = \
                    global_data[chan.name].data[idxs]
                
                # coordinates are per channel
                if not global_coordinates:
                    global_data[chan.name].area.lons = \
                        global_data[chan.name].area.lons[idxs]
                    global_data[chan.name].area.lats = \
                        global_data[chan.name].area.lats[idxs]
                    global_data[chan.name].area.shape = \
                        global_data[chan.name].area.lons.shape
                    size_before = global_data.area.size
                    val = 1
                    for i in global_data[chan.name].area.shape:
                        val *= i
                    global_data[chan.name].area.size = val
                    size_after = global_data.area.size

                # one set of coordinates for all channel data
                if global_coordinates and not coordinate_data_reduced:
                    global_data.area.lons = global_data.area.lons[idxs]
                    global_data.area.lats = global_data.area.lats[idxs]
                    global_data.area.shape = global_data.area.lons.shape
                    val = 1
                    size_before = global_data.area.size
                    for i in global_data.area.shape:
                        val *= i
                    global_data.area.size = val
                    size_after = global_data.area.size
                    coordinate_data_reduced = True
                    
                LOGGER.debug("Data reduced by %.1f %%" % \
                                 (100*(1-float(size_after)/ \
                                           size_before)))
