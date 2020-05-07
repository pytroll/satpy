#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016-2018 Satpy developers
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
"""Advance Baseline Imager reader base class for the Level 1b and l2+ reader."""

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
    """Base reader for ABI L1B  L2+ NetCDF4 files."""

    def __init__(self, filename, filename_info, filetype_info):
        """Open the NetCDF file with xarray and prepare the Dataset for reading."""
        super(NC_ABI_BASE, self).__init__(filename, filename_info, filetype_info)
        # xarray's default netcdf4 engine
        try:
            self.nc = xr.open_dataset(self.filename,
                                      decode_cf=True,
                                      mask_and_scale=False,
                                      chunks={'x': CHUNK_SIZE, 'y': CHUNK_SIZE}, )
        except ValueError:
            self.nc = xr.open_dataset(self.filename,
                                      decode_cf=True,
                                      mask_and_scale=False,
                                      chunks={'lon': CHUNK_SIZE, 'lat': CHUNK_SIZE}, )

        if 't' in self.nc.dims or 't' in self.nc.coords:
            self.nc = self.nc.rename({'t': 'time'})
        platform_shortname = filename_info['platform_shortname']
        self.platform_name = PLATFORM_NAMES.get(platform_shortname)

        if 'goes_imager_projection' in self.nc:
            self.nlines = self.nc['y'].size
            self.ncols = self.nc['x'].size
        elif 'goes_lat_lon_projection' in self.nc:
            self.nlines = self.nc['lat'].size
            self.ncols = self.nc['lon'].size
            self.nc = self.nc.rename({'lon': 'x', 'lat': 'y'})

        self.coords = {}

    @property
    def sensor(self):
        """Get sensor name for current file handler."""
        return 'abi'

    def __getitem__(self, item):
        """Wrap `self.nc[item]` for better floating point precision.

        Some datasets use a 32-bit float scaling factor like the 'x' and 'y'
        variables which causes inaccurate unscaled data values. This method
        forces the scale factor to a 64-bit float first.
        """
        def is_int(val):
            return np.issubdtype(val.dtype, np.integer) if hasattr(val, 'dtype') else isinstance(val, int)

        data = self.nc[item]
        attrs = data.attrs

        factor = data.attrs.get('scale_factor', 1)
        offset = data.attrs.get('add_offset', 0)
        fill = data.attrs.get('_FillValue')
        unsigned = data.attrs.get('_Unsigned', None)

        # Ref. GOESR PUG-L1B-vol3, section 5.0.2 Unsigned Integer Processing
        if unsigned is not None and unsigned.lower() == 'true':
            # cast the data from int to uint
            data = data.astype('u%s' % data.dtype.itemsize)

            if fill is not None:
                fill = fill.astype('u%s' % fill.dtype.itemsize)

        if fill is not None:
            if is_int(data) and is_int(factor) and is_int(offset):
                new_fill = fill
            else:
                new_fill = np.nan
            data = data.where(data != fill, new_fill)

        if factor != 1 and item in ('x', 'y'):
            # be more precise with x/y coordinates
            # see get_area_def for more information
            data = data * np.round(float(factor), 6) + np.round(float(offset), 6)
        elif factor != 1:
            # make sure the factor is a 64-bit float
            # can't do this in place since data is most likely uint16
            # and we are making it a 64-bit float
            if not is_int(factor):
                factor = float(factor)
            data = data * factor + offset

        data.attrs = attrs

        # handle coordinates (and recursive fun)
        new_coords = {}
        # 'time' dimension causes issues in other processing
        # 'x_image' and 'y_image' are confusing to some users and unnecessary
        # 'x' and 'y' will be overwritten by base class AreaDefinition
        for coord_name in ('x_image', 'y_image', 'time', 'x', 'y'):
            if coord_name in data.coords:
                data = data.drop_vars(coord_name)
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
        raise NotImplementedError("Reader {} has not implemented get_dataset".format(self.name))

    def get_area_def(self, key):
        """Get the area definition of the data at hand."""
        if 'goes_imager_projection' in self.nc:
            return self._get_areadef_fixedgrid(key)
        elif 'goes_lat_lon_projection' in self.nc:
            return self._get_areadef_latlon(key)
        else:
            raise ValueError('Unsupported projection found in the dataset')

    def _get_areadef_latlon(self, key):
        """Get the area definition of the data at hand."""
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
            self.nc.attrs.get('orbital_slot', 'abi_geos'),
            self.nc.attrs.get('spatial_resolution', 'ABI file area'),
            'abi_latlon',
            proj_dict,
            self.ncols,
            self.nlines,
            np.asarray(area_extent))

        return ll_area_def

    def _get_areadef_fixedgrid(self, key):
        """Get the area definition of the data at hand.

        Note this method takes special care to round and cast numbers to new
        data types so that the area definitions for different resolutions
        (different bands) should be equal. Without the special rounding in
        `__getitem__` and this method the area extents can be 0 to 1.0 meters
        off depending on how the calculations are done.

        """
        projection = self.nc["goes_imager_projection"]
        a = projection.attrs['semi_major_axis']
        b = projection.attrs['semi_minor_axis']
        h = projection.attrs['perspective_point_height']

        lon_0 = projection.attrs['longitude_of_projection_origin']
        sweep_axis = projection.attrs['sweep_angle_axis'][0]

        # compute x and y extents in m
        h = np.float64(h)
        x = self['x']
        y = self['y']
        x_l = x[0].values
        x_r = x[-1].values
        y_l = y[-1].values
        y_u = y[0].values
        x_half = (x_r - x_l) / (self.ncols - 1) / 2.
        y_half = (y_u - y_l) / (self.nlines - 1) / 2.
        area_extent = (x_l - x_half, y_l - y_half, x_r + x_half, y_u + y_half)
        area_extent = tuple(np.round(h * val, 6) for val in area_extent)

        proj_dict = {'proj': 'geos',
                     'lon_0': float(lon_0),
                     'a': float(a),
                     'b': float(b),
                     'h': h,
                     'units': 'm',
                     'sweep': sweep_axis}

        fg_area_def = geometry.AreaDefinition(
            self.nc.attrs.get('orbital_slot', 'abi_geos'),
            self.nc.attrs.get('spatial_resolution', 'ABI file area'),
            'abi_fixed_grid',
            proj_dict,
            self.ncols,
            self.nlines,
            np.asarray(area_extent))

        return fg_area_def

    @property
    def start_time(self):
        """Start time of the current file's observations."""
        return datetime.strptime(self.nc.attrs['time_coverage_start'], '%Y-%m-%dT%H:%M:%S.%fZ')

    @property
    def end_time(self):
        """End time of the current file's observations."""
        return datetime.strptime(self.nc.attrs['time_coverage_end'], '%Y-%m-%dT%H:%M:%S.%fZ')

    def spatial_resolution_to_number(self):
        """Convert the 'spatial_resolution' global attribute to meters."""
        res = self.nc.attrs['spatial_resolution'].split(' ')[0]
        if res.endswith('km'):
            res = int(float(res[:-2]) * 1000)
        elif res.endswith('m'):
            res = int(res[:-1])
        else:
            raise ValueError("Unexpected 'spatial_resolution' attribute '{}'".format(res))
        return res

    def __del__(self):
        """Close the NetCDF file that may still be open."""
        try:
            self.nc.close()
        except (IOError, OSError, AttributeError):
            pass
