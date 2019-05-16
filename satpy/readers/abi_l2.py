#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016-2019 Satpy developers
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
"""
Advance Baseline Imager NOAA Level 2+ products reader 
The files read by this reader are described in the official PUG document:
    https://www.goes-r.gov/products/docs/PUG-L2+-vol5.pdf
"""

import logging
from datetime import datetime

import numpy as np
import xarray as xr

from pyresample import geometry
from satpy.readers.file_handlers import BaseFileHandler
from satpy import CHUNK_SIZE

LOG = logging.getLogger(__name__)

PLATFORM_NAMES = {'G16': 'GOES-16', 
                  'G17': 'GOES-17'}


class NC_ABI_L2(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(NC_ABI_L2, self).__init__(filename, filename_info, filetype_info)
        try:
            self.nc = xr.open_dataset(filename, decode_cf=True,
                                      chunks={'x': CHUNK_SIZE, 'y': CHUNK_SIZE},)
        except ValueError:
            self.nc = xr.open_dataset(filename, decode_cf=True,
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


    def get_shape(self, key, info):
        """Get the shape of the data.
        """
        return self.nlines, self.ncols


    def get_dataset(self, key, info):
        """Load a dataset.
        """
        LOG.debug('Reading in get_dataset %s.', key.name)
        variable = self.nc[key.name]

        # handle coordinates (and recursive fun)
        new_coords = {}
        # 'time' dimension causes issues in other processing
        if 'time' in variable.coords:
            del variable.coords['time']

        if key.name in variable.coords:
            self.coords[key.name] = variable

        for coord_name in variable.coords.keys():
            if coord_name not in self.coords:
                self.coords[coord_name] = self.nc[coord_name]
            new_coords[coord_name] = self.coords[coord_name]

        variable.coords.update(new_coords)

        _units = variable.attrs['units'] if 'units' in variable.attrs else None

        variable.attrs.update({'platform_name': self.platform_name,
                               'sensor': self.sensor,
                               'units': _units,
                               'satellite_latitude': float(self.nc['nominal_satellite_subpoint_lat']),
                               'satellite_longitude': float(self.nc['nominal_satellite_subpoint_lon']),
                               'satellite_altitude': float(self.nc['nominal_satellite_height'])})

        variable.attrs.update(key.to_dict())

        # remove attributes that could be confusing later
        variable.attrs.pop('_FillValue', None)
        variable.attrs.pop('scale_factor', None)
        variable.attrs.pop('add_offset', None)
        variable.attrs.pop('valid_range', None)
        
        # add in information from the filename that may be useful to the user
        for k in ('scan_mode', 'platform_shortname'):
            variable.attrs[k] = self.filename_info[k]

        # copy global attributes to metadata
        for k in ('scene_id', 'orbital_slot', 'instrument_ID', 'production_site', 'timeline_ID'):
            variable.attrs[k] = self.nc.attrs.get(k)

        return variable


    def get_area_def(self, key):
        """Get the area definition of the data at hand.
        """
        if 'goes_imager_projection' in self.nc.keys():
            return self._get_areadef_fixedgrid(key)
        elif 'goes_lat_lon_projection' in self.nc.keys():
            return self._get_areadef_latlon(key)


    def _get_areadef_latlon(self, key):
        """Get the area definition of the data at hand.
        """
        from pyproj import Proj

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
                     'pm': float(pm),
                    }

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
        except (IOError, OSError):
            pass
