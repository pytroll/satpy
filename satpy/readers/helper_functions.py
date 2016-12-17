
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2014, 2015.
#
# Author(s):
#
#   Panu Lahtinen <panu.lahtinen@fmi.fi>
#   Martin Raspaud <martin.raspaud@smhi.se>
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

'''Helper functions for area extent calculations.
'''

import glob
import logging

import numpy as np
from pyproj import Proj

import pyresample
from pyresample.geometry import AreaDefinition
from satpy.resample import get_area_def
from trollsift import Parser

# from pyresample.utils import AreaNotFound


LOGGER = logging.getLogger(__name__)


def area_def_names_to_extent(area_def_names, proj4_str,
                             default_extent=(-5567248.07, -5570248.48,
                                             5570248.48, 5567248.07)):
    '''Convert a list of *area_def_names* to maximal area extent in destination
    projection defined by *proj4_str*. *default_extent* gives the extreme
    values.  Default value is MSG3 extents at lat0=0.0. If a boundary of one of
    the area_defs is entirely invalid, the *default_extent* is taken.
    '''

    if not isinstance(area_def_names, (list, tuple, set)):
        area_def_names = [area_def_names]

    areas = []
    for name in area_def_names:

        try:
            areas.append(get_area_def(name))
        except pyresample.utils.AreaNotFound:
            LOGGER.warning('Area definition not found ' + name)
            continue

    return area_defs_to_extent(areas, proj4_str, default_extent)


def area_defs_to_extent(area_defs, proj4_str,
                        default_extent=(-5567248.07, -5570248.48,
                                        5570248.48, 5567248.07)):
    '''Convert a list of *area_def_names* to maximal area extent in
    destination projection defined by *proj4_str*. *default_extent*
    gives the extreme values.  Default value is MSG3 extents at
    lat0=0.0.
    '''

    if not isinstance(area_defs, (list, tuple, set)):
        area_defs = [area_defs]

    maximum_extent = None

    for area in area_defs:

        boundaries = area.get_boundary_lonlats()

        if (all(boundaries[0].side1 > 1e20) or
                all(boundaries[0].side2 > 1e20) or
                all(boundaries[0].side3 > 1e20) or
                all(boundaries[0].side4 > 1e20)):
            maximum_extent = list(default_extent)
            continue

        lon_sides = (boundaries[0].side1, boundaries[0].side2,
                     boundaries[0].side3, boundaries[0].side4)
        lat_sides = (boundaries[1].side1, boundaries[1].side2,
                     boundaries[1].side3, boundaries[1].side4)

        maximum_extent = boundaries_to_extent(proj4_str, maximum_extent,
                                              default_extent,
                                              lon_sides, lat_sides)

    maximum_extent[0] -= 10000
    maximum_extent[1] -= 10000
    maximum_extent[2] += 10000
    maximum_extent[3] += 10000

    return maximum_extent


def boundaries_to_extent(proj4_str, maximum_extent, default_extent,
                         lon_sides, lat_sides):
    '''Get area extent from given boundaries.
    '''

    # proj4-ify the projection string
    if '+' not in proj4_str:
        proj4_str = proj4_str.split(' ')
        proj4_str = '+' + ' +'.join(proj4_str)

    pro = Proj(proj4_str)

    # extents for edges
    x_dir, y_dir = pro(np.concatenate(lon_sides),
                       np.concatenate(lat_sides))

    # replace invalid values with NaN
    x_dir[np.abs(x_dir) > 1e20] = np.nan
    y_dir[np.abs(y_dir) > 1e20] = np.nan

    # Get the maximum needed extent from different corners.
    extent = [np.nanmin(x_dir),
              np.nanmin(y_dir),
              np.nanmax(x_dir),
              np.nanmax(y_dir)]

    # Replace "infinity" values with default extent
    for i in range(4):
        if extent[i] is np.nan:
            extent[i] = default_extent[i]

    # update maximum extent
    if maximum_extent is None:
        maximum_extent = extent
    else:
        if maximum_extent[0] > extent[0]:
            maximum_extent[0] = extent[0]
        if maximum_extent[1] > extent[1]:
            maximum_extent[1] = extent[1]
        if maximum_extent[2] < extent[2]:
            maximum_extent[2] = extent[2]
        if maximum_extent[3] < extent[3]:
            maximum_extent[3] = extent[3]

    # Replace "infinity" values with default extent
    for i in range(4):
        if not np.isfinite(maximum_extent[i]):
            maximum_extent[i] = default_extent[i]

    return maximum_extent


def get_geostationary_bounding_box(geos_area):

    # TODO: take into account sweep_axis_angle parameter

    # get some projection parameters
    req = geos_area.proj_dict['a'] / 1000
    rp = geos_area.proj_dict['b'] / 1000
    h = geos_area.proj_dict['h'] / 1000 + req

    # compute some constants
    a = 1 - req**2 / (h / 1.2) ** 2
    c = (h**2 - req**2)
    b = req**2 / rp**2

    # generate points around the north hemisphere in satellite projection
    # make it a bit smaller so that we stay inside the valide area
    xmax = np.arccos(np.sqrt(a)) - 0.000000001
    x = np.cos(np.linspace(-np.pi, np.pi, 50)) * xmax
    y = np.arctan(np.sqrt(1 / b * (np.cos(x) ** 2 / a - 1)))

    # clip the projection coordinates to fit the area extent of geos_area
    ll_x, ll_y, ur_x, ur_y = geos_area.area_extent

    x = np.clip(np.concatenate([x, x[::-1]]), min(ll_x, ur_x), max(ll_x, ur_x))
    y = np.clip(np.concatenate([y, -y]), min(ll_y, ur_y), max(ll_y, ur_y))

    # compute the latitudes and longitudes

    # sd = np.sqrt((h * np.cos(x) * np.cos(y))**2 -
    #             (np.cos(y)**2 + b * (np.sin(y)**2)) * c)
    sd = 0
    sn = (h * np.cos(x) * np.cos(y) - sd) / (np.cos(y)**2 + b * np.sin(y)**2)

    s1 = h - sn * np.cos(x) * np.cos(y)
    s2 = sn * np.sin(x) * np.cos(y)
    s3 = -sn * np.sin(y)
    sxy = np.sqrt(s1**2 + s2**2)

    lons = np.rad2deg(np.arctan2(s2, s1)) + geos_area.proj_dict.get('lon_0', 0)
    lats = np.rad2deg(np.arctan2(b * s3, sxy))

    return lons, lats


def get_area_slices(data_area, area_to_cover):
    """This function computes the slice to read from an *area* based on an
     *area_to_cover*.
     """
    from trollsched.boundary import AreaDefBoundary, Boundary

    if data_area.proj_dict['proj'] != 'geos':
        raise NotImplementedError('Only geos supported')

    data_boundary = Boundary(*get_geostationary_bounding_box(data_area))

    area_boundary = AreaDefBoundary(area_to_cover, 100)

    # from trollsched.satpass import Mapper
    # import matplotlib.pyplot as plt
    # with Mapper() as mapper:
    #     data_boundary.draw(mapper, 'g+-')
    #     area_boundary.draw(mapper, 'b+-')
    #     res = area_boundary.contour_poly.intersection(
    #         data_boundary.contour_poly)
    #     res.draw(mapper, 'r+-')
    # plt.show()

    intersection = data_boundary.contour_poly.intersection(
        area_boundary.contour_poly)

    x, y = data_area.get_xy_from_lonlat(np.rad2deg(intersection.lon),
                                        np.rad2deg(intersection.lat))

    return slice(min(x), max(x) + 1), slice(min(y), max(y) + 1)


def get_sub_area(area, xslice, yslice):
    new_area_extent = ((area.pixel_upper_left[0] +
                        (xslice.start - 0.5) * area.pixel_size_x),
                       (area.pixel_upper_left[1] -
                        (yslice.stop - 0.5) * area.pixel_size_y),
                       (area.pixel_upper_left[0] +
                        (xslice.stop - 0.5) * area.pixel_size_x),
                       (area.pixel_upper_left[1] -
                        (yslice.start - 0.5) * area.pixel_size_y))

    return AreaDefinition(area.area_id, area.name,
                          area.proj_id, area.proj_dict,
                          xslice.stop - xslice.start,
                          yslice.stop - yslice.start,
                          new_area_extent)


def get_filenames(filepattern):
    parser = Parser(filepattern)
    for filename in glob.iglob(parser.globify()):
        yield filename, parser.parse(filename)
