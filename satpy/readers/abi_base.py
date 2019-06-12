#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016-2018 Satpy developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Advance Baseline Imager reader base class for the Level 1b and l2+ reader
"""

import logging
from datetime import datetime

import numpy as np
import xarray as xr

from pyresample import geometry
from satpy.readers.file_handlers import BaseFileHandler
from satpy import CHUNK_SIZE

logger = logging.getLogger(__name__)

PLATFORM_NAMES = {
    'G16': 'GOES-16',
    'G17': 'GOES-17',
}


class NC_ABI_BASE(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(NC_ABI_BASE, self).__init__(filename, filename_info, filetype_info)
        # xarray's default netcdf4 engine
        try:
            self.nc = xr.open_dataset(self.filename,
                                      decode_cf=True,
                                      mask_and_scale=False,
                                      chunks={'x': CHUNK_SIZE, 'y': CHUNK_SIZE},)
        except ValueError:
            self.nc = xr.open_dataset(self.filename,
                                      decode_cf=True,
                                      mask_and_scale=False,
                                      chunks={'lon': CHUNK_SIZE, 'lat': CHUNK_SIZE},)

        self.nc = self.nc.rename({'t': 'time'})
        platform_shortname = filename_info['platform_shortname']
        self.platform_name = PLATFORM_NAMES.get(platform_shortname)
        self.sensor = 'abi'

        if 'goes_imager_projection' in self.nc.keys():
            self.nlines = self.nc['y'].size
            self.ncols = self.nc['x'].size
        elif 'goes_lat_lon_projection' in self.nc.keys():
            self.nlines = self.nc['lat'].size
            self.ncols = self.nc['lon'].size
            self.nc = self.nc.rename({'lon': 'x', 'lat': 'y'})

        self.coords = {}

    def __getitem__(self, item):
        """Wrapper around `self.nc[item]`.

        Some datasets use a 32-bit float scaling factor like the 'x' and 'y'
        variables which causes inaccurate unscaled data values. This method
        forces the scale factor to a 64-bit float first.

        """
        data = self.nc[item]
        attrs = data.attrs
        factor = data.attrs.get('scale_factor')
        offset = data.attrs.get('add_offset')
        fill = data.attrs.get('_FillValue')
        if fill is not None:
            data = data.where(data != fill)
        if factor is not None:
            # make sure the factor is a 64-bit float
            # can't do this in place since data is most likely uint16
            # and we are making it a 64-bit float
            data = data * float(factor) + offset
        data.attrs = attrs

        # handle coordinates (and recursive fun)
        new_coords = {}
        # 'time' dimension causes issues in other processing
        if 'time' in data.coords:
            del data.coords['time']
        if item in data.coords:
            self.coords[item] = data
        for coord_name in data.coords.keys():
            if coord_name not in self.coords:
                self.coords[coord_name] = self[coord_name]
            new_coords[coord_name] = self.coords[coord_name]
        data.coords.update(new_coords)
        return data

    def get_dataset(self, key, info):
        """Load a dataset."""
        raise NotImplementedError("Writer {} has not implemented get_dataset".format(self.name))

    def get_area_def(self, key):
        """Get the area definition of the data at hand.
        """
        if 'goes_imager_projection' in self.nc.keys():
            return self._get_areadef_fixedgrid(key)
        elif 'goes_lat_lon_projection' in self.nc.keys():
            return self._get_areadef_latlon(key)
        else:
            raise ValueError('Unsupported projection found in the dataset')

    def _get_areadef_latlon(self, key):
        """Get the area definition of the data at hand.
        """
        projection = self.nc["goes_lat_lon_projection"]

        a = projection.attrs['semi_major_axis']
        b = projection.attrs['semi_minor_axis']
        fi = projection.attrs['inverse_flattening']
        pm = projection.attrs['longitude_of_prime_meridian']

        proj_ext = self.nc["geospatial_lat_lon_extent"]

        w_lon = proj_ext.attrs['geospatial_westbound_longitude']
        e_lon = proj_ext.attrs['geospatial_eastbound_longitude']
        n_lat = proj_ext.attrs['geospatial_northbound_latitude']
        s_lat = proj_ext.attrs['geospatial_southbound_latitude']

        lat_0 = proj_ext.attrs['geospatial_lat_center']
        lon_0 = proj_ext.attrs['geospatial_lon_center']

        area_extent = (w_lon, s_lat, e_lon, n_lat)
        proj_dict = {'proj': 'latlong',
                     'lon_0': float(lon_0),
                     'lat_0': float(lat_0),
                     'a': float(a),
                     'b': float(b),
                     'fi': float(fi),
                     'pm': float(pm)}

        ll_area_def = geometry.AreaDefinition(
            self.nc.attrs.get('orbital_slot', 'GOES-R'),
            self.nc.attrs.get('spatial_resolution', 'ABI L2+ file area'),
            'abi_l2+_latlon',
            proj_dict,
            self.ncols,
            self.nlines,
            np.asarray(area_extent))

        return ll_area_def

    def _get_areadef_fixedgrid(self, key):
        """Get the area definition of the data at hand.
        """
        projection = self.nc["goes_imager_projection"]

        a = projection.attrs['semi_major_axis']
        b = projection.attrs['semi_minor_axis']
        h = projection.attrs['perspective_point_height']

        lon_0 = projection.attrs['longitude_of_projection_origin']
        sweep_axis = projection.attrs['sweep_angle_axis'][0]

        # compute x and y extents in m
        h = float(h)

        x = self.nc['x']
        y = self.nc['y']
        x_l = h * x[0]
        x_r = h * x[-1]
        y_l = h * y[-1]
        y_u = h * y[0]

        x_half = (x_r - x_l) / (self.ncols - 1) / 2.
        y_half = (y_u - y_l) / (self.nlines - 1) / 2.
        area_extent = (x_l - x_half, y_l - y_half, x_r + x_half, y_u + y_half)

        proj_dict = {'proj': 'geos',
                     'lon_0': float(lon_0),
                     'a': float(a),
                     'b': float(b),
                     'h': h,
                     'units': 'm',
                     'sweep': sweep_axis}

        fg_area_def = geometry.AreaDefinition(
            self.nc.attrs.get('orbital_slot', 'GOES-R'),  # "GOES-East", "GOES-West"
            self.nc.attrs.get('spatial_resolution', 'ABI L2+ file area'),
            'abi_fixed_grid',
            proj_dict,
            self.ncols,
            self.nlines,
            np.asarray(area_extent))

        return fg_area_def

    @property
    def start_time(self):
        return datetime.strptime(self.nc.attrs['time_coverage_start'], '%Y-%m-%dT%H:%M:%S.%fZ')

    @property
    def end_time(self):
        return datetime.strptime(self.nc.attrs['time_coverage_end'], '%Y-%m-%dT%H:%M:%S.%fZ')

    def __del__(self):
        try:
            self.nc.close()
        except (IOError, OSError, AttributeError):
            pass
