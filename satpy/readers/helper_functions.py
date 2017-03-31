
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

"""Helper functions for area extent calculations."""

import logging

import numpy as np
from pyresample.geometry import AreaDefinition

LOGGER = logging.getLogger(__name__)


def get_geostationary_angle_extent(geos_area):
    """Get the max earth (vs space) viewing angles in x and y."""
    # TODO: take into account sweep_axis_angle parameter

    # get some projection parameters
    req = geos_area.proj_dict['a'] / 1000
    rp = geos_area.proj_dict['b'] / 1000
    h = geos_area.proj_dict['h'] / 1000 + req

    # compute some constants
    aeq = 1 - req**2 / (h ** 2)
    ap_ = 1 - rp**2 / (h ** 2)

    # generate points around the north hemisphere in satellite projection
    # make it a bit smaller so that we stay inside the valid area
    xmax = np.arccos(np.sqrt(aeq))
    ymax = np.arccos(np.sqrt(ap_))
    return xmax, ymax


def _lonlat_from_geos_angle(x, y, geos_area):
    """Get lons and lats from x, y in projection coordinates."""
    h = (geos_area.proj_dict['h'] + geos_area.proj_dict['a']) / 1000
    b__ = (geos_area.proj_dict['a'] / geos_area.proj_dict['b']) ** 2

    sd = np.sqrt((h * np.cos(x) * np.cos(y)) ** 2 -
                 (np.cos(y)**2 + b__ * np.sin(y)**2) *
                 (h**2 - (geos_area.proj_dict['a'] / 1000)**2))
    #sd = 0

    sn = (h * np.cos(x) * np.cos(y) - sd) / (np.cos(y)**2 + b__ * np.sin(y)**2)
    s1 = h - sn * np.cos(x) * np.cos(y)
    s2 = sn * np.sin(x) * np.cos(y)
    s3 = -sn * np.sin(y)
    sxy = np.sqrt(s1**2 + s2**2)

    lons = np.rad2deg(np.arctan2(s2, s1)) + geos_area.proj_dict.get('lon_0', 0)
    lats = np.rad2deg(-np.arctan2(b__ * s3, sxy))

    return lons, lats


def get_geostationary_bounding_box(geos_area, nb_points=50):
    """Get the bbox in lon/lats of the valid pixels inside *geos_area*.

    Args:
      nb_points: Number of points on the polygon
    """
    xmax, ymax = get_geostationary_angle_extent(geos_area)

    # generate points around the north hemisphere in satellite projection
    # make it a bit smaller so that we stay inside the valid area
    x = np.cos(np.linspace(-np.pi, 0, nb_points / 2)) * (xmax - 0.001)
    y = -np.sin(np.linspace(-np.pi, 0, nb_points / 2)) * (ymax - 0.001)

    # clip the projection coordinates to fit the area extent of geos_area
    ll_x, ll_y, ur_x, ur_y = (np.array(geos_area.area_extent) /
                              geos_area.proj_dict['h'])

    x = np.clip(np.concatenate([x, x[::-1]]), min(ll_x, ur_x), max(ll_x, ur_x))
    y = np.clip(np.concatenate([y, -y]), min(ll_y, ur_y), max(ll_y, ur_y))

    return _lonlat_from_geos_angle(x, y, geos_area)


def get_area_slices(data_area, area_to_cover):
    """Compute the slice to read from an *area* based on an *area_to_cover*."""

    if data_area.proj_dict['proj'] != 'geos':
        raise NotImplementedError('Only geos supported')

    # Intersection only required for two different projections
    if area_to_cover.proj_dict['proj'] == data_area.proj_dict['proj']:
        LOGGER.debug('Projections for data and slice areas are'
                     ' identical: {}'.format(area_to_cover.proj_dict['proj']))
        # Get xy coordinates
        llx, lly, urx, ury = area_to_cover.area_extent
        x, y = data_area.get_xy_from_proj_coords([llx, urx], [lly, ury])

        return slice(x[0], x[1] + 1), slice(y[1], y[0] + 1)

    from trollsched.boundary import AreaDefBoundary, Boundary

    data_boundary = Boundary(*get_geostationary_bounding_box(data_area))

    area_boundary = AreaDefBoundary(area_to_cover, 100)
    intersection = data_boundary.contour_poly.intersection(
        area_boundary.contour_poly)

    x, y = data_area.get_xy_from_lonlat(np.rad2deg(intersection.lon),
                                        np.rad2deg(intersection.lat))

    return slice(min(x), max(x) + 1), slice(min(y), max(y) + 1)


def get_sub_area(area, xslice, yslice):
    """Apply slices to the area_extent and size of the area."""
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
