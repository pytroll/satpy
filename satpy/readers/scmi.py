#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 PyTroll developers
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
"""SCMI NetCDF4 Reader

SCMI files are typically used for data for the ABI instrument onboard the
GOES-16/17 satellites. It is the primary format used for providing ABI data
to the AWIPS visualization clients used by the US National Weather Service
forecasters. The python code for this reader may be reused by other readers
as NetCDF schemes/metadata change for different products. The initial reader
using this code is the "scmi_abi" reader (see `abi_l1b_scmi.yaml` for more
information).

There are two forms of these files that this reader supports:

1. Official SCMI format: NetCDF4 files where the main data variable is stored
    in a variable called "Sectorized_CMI". This variable name can be
    configured in the YAML configuration file.
2. Satpy/Polar2Grid SCMI format: NetCDF4 files based on the official SCMI
    format created for the Polar2Grid project. This format was migrated to
    Satpy as part of Polar2Grid's adoption of Satpy for the majority of its
    features. This format is what is produced by Satpy's `scmi` writer.
    This format can be identified by a single variable named "data" and a
    global attribute named ``"awips_id"`` that is set to a string starting with
    ``"AWIPS_"``.

"""

import logging
from datetime import datetime

import os
import numpy as np
import xarray as xr

from pyresample import geometry
from satpy.readers.file_handlers import BaseFileHandler
from satpy import CHUNK_SIZE

# NetCDF doesn't support multi-threaded reading, trick it by opening
# as one whole chunk then split it up before we do any calculations
LOAD_CHUNK_SIZE = int(os.getenv('PYTROLL_LOAD_CHUNK_SIZE', -1))
logger = logging.getLogger(__name__)


class SCMIFileHandler(BaseFileHandler):
    """Handle a single SCMI NetCDF4 file."""

    def __init__(self, filename, filename_info, filetype_info):
        super(SCMIFileHandler, self).__init__(filename, filename_info,
                                              filetype_info)
        # xarray's default netcdf4 engine
        self.nc = xr.open_dataset(self.filename,
                                  decode_cf=True,
                                  mask_and_scale=False,
                                  chunks={'x': LOAD_CHUNK_SIZE, 'y': LOAD_CHUNK_SIZE})
        self.platform_name = self.nc.attrs['satellite_id']
        self.sensor = self._get_sensor()
        self.nlines = self.nc.dims['y']
        self.ncols = self.nc.dims['x']
        self.coords = {}

    def _get_sensor(self):
        """Determine the sensor for this file."""
        # sometimes Himawari-8 (or 9) data is stored in SCMI format
        is_h8 = 'H8' in self.platform_name
        is_h9 = 'H9' in self.platform_name
        is_ahi = is_h8 or is_h9
        return 'ahi' if is_ahi else 'abi'

    @property
    def sensor_names(self):
        return [self.sensor]

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

    def get_shape(self, key, info):
        """Get the shape of the data."""
        return self.nlines, self.ncols

    def get_dataset(self, key, info):
        """Load a dataset."""
        logger.debug('Reading in get_dataset %s.', key.name)
        var_name = info.get('file_key', self.filetype_info.get('file_key'))
        if var_name:
            data = self[var_name]
        elif 'Sectorized_CMI' in self.nc:
            data = self['Sectorized_CMI']
        elif 'data' in self.nc:
            data = self['data']
        # NetCDF doesn't support multi-threaded reading, trick it by opening
        # as one whole chunk then split it up before we do any calculations
        data = data.chunk({'x': CHUNK_SIZE, 'y': CHUNK_SIZE})

        # convert to satpy standard units
        factor = data.attrs.pop('scale_factor', 1)
        offset = data.attrs.pop('add_offset', 0)
        units = data.attrs.get('units', 1)
        # the '*1' unit is some weird convention added/needed by AWIPS
        if units in ['1', '*1'] and key.calibration == 'reflectance':
            data *= 100
            factor *= 100  # used for valid_min/max
            data.attrs['units'] = '%'

        # set up all the attributes that might be useful to the user/satpy
        data.attrs.update({'platform_name': self.platform_name,
                           'sensor': data.attrs.get('sensor', self.sensor),
                           })
        if 'satellite_longitude' in self.nc.attrs:
            data.attrs['satellite_longitude'] = self.nc.attrs['satellite_longitude']
            data.attrs['satellite_latitude'] = self.nc.attrs['satellite_latitude']
            data.attrs['satellite_altitude'] = self.nc.attrs['satellite_altitude']

        scene_id = self.nc.attrs.get('scene_id')
        if scene_id is not None:
            data.attrs['scene_id'] = scene_id
        data.attrs.update(key.to_dict())
        data.attrs.pop('_FillValue', None)
        if 'valid_min' in data.attrs:
            vmin = data.attrs.pop('valid_min')
            vmax = data.attrs.pop('valid_max')
            vmin = vmin * factor + offset
            vmax = vmax * factor + offset
            data.attrs['valid_min'] = vmin
            data.attrs['valid_max'] = vmax
        return data

    def _get_cf_grid_mapping_var(self):
        """Figure out which grid mapping should be used"""
        gmaps = ['fixedgrid_projection', 'goes_imager_projection',
                 'lambert_projection', 'polar_projection',
                 'mercator_projection']
        if 'grid_mapping' in self.filename_info:
            gmaps = [self.filename_info.get('grid_mapping')] + gmaps
        for grid_mapping in gmaps:
            if grid_mapping in self.nc:
                return self.nc[grid_mapping]
        raise KeyError("Can't find grid mapping variable in SCMI file")

    def _get_proj4_name(self, projection):
        """Map CF projection name to PROJ.4 name."""
        gmap_name = projection.attrs['grid_mapping_name']
        proj = {
            'geostationary': 'geos',
            'lambert_conformal_conic': 'lcc',
            'polar_stereographic': 'stere',
            'mercator': 'merc',
        }.get(gmap_name, gmap_name)
        return proj

    def _get_proj_specific_params(self, projection):
        """Convert CF projection parameters to PROJ.4 dict."""
        proj = self._get_proj4_name(projection)
        proj_dict = {
            'proj': proj,
            'a': float(projection.attrs['semi_major_axis']),
            'b': float(projection.attrs['semi_minor_axis']),
            'units': 'm',
        }
        if proj == 'geos':
            proj_dict['h'] = float(projection.attrs['perspective_point_height'])
            proj_dict['sweep'] = projection.attrs.get('sweep_angle_axis', 'y')
            proj_dict['lon_0'] = float(projection.attrs['longitude_of_projection_origin'])
            proj_dict['lat_0'] = float(projection.attrs.get('latitude_of_projection_origin', 0.0))
        elif proj == 'lcc':
            proj_dict['lat_0'] = float(projection.attrs['standard_parallel'])
            proj_dict['lon_0'] = float(projection.attrs['longitude_of_central_meridian'])
            proj_dict['lat_1'] = float(projection.attrs['latitude_of_projection_origin'])
        elif proj == 'stere':
            proj_dict['lat_ts'] = float(projection.attrs['standard_parallel'])
            proj_dict['lon_0'] = float(projection.attrs['straight_vertical_longitude_from_pole'])
            proj_dict['lat_0'] = float(projection.attrs['latitude_of_projection_origin'])
        elif proj == 'merc':
            proj_dict['lat_ts'] = float(projection.attrs['standard_parallel'])
            proj_dict['lat_0'] = proj_dict['lat_ts']
            proj_dict['lon_0'] = float(projection.attrs['longitude_of_projection_origin'])
        else:
            raise ValueError("Can't handle projection '{}'".format(proj))
        return proj_dict

    def _calc_extents(self, proj_dict):
        """Calculate area extents from x/y variables."""
        h = float(proj_dict.get('h', 1.))  # force to 64-bit float
        x = self['x']
        y = self['y']
        x_units = x.attrs.get('units', 'rad')
        if x_units == 'meters':
            h_factor = 1.
            factor = 1.
        elif x_units == 'microradian':
            h_factor = h
            factor = 1e6
        else:  # radians
            h_factor = h
            factor = 1.
        x_l = h_factor * x[0] / factor
        x_r = h_factor * x[-1] / factor
        y_l = h_factor * y[-1] / factor
        y_u = h_factor * y[0] / factor
        x_half = (x_r - x_l) / (self.ncols - 1) / 2.
        y_half = (y_u - y_l) / (self.nlines - 1) / 2.
        return x_l - x_half, y_l - y_half, x_r + x_half, y_u + y_half

    def get_area_def(self, key):
        """Get the area definition of the data at hand."""
        # FIXME: Can't we pass dataset info to the get_area_def?
        projection = self._get_cf_grid_mapping_var()
        proj_dict = self._get_proj_specific_params(projection)
        area_extent = self._calc_extents(proj_dict)
        area_name = '{}_{}'.format(self.sensor, proj_dict['proj'])
        return geometry.AreaDefinition(
            area_name,
            "SCMI file area",
            area_name,
            proj_dict,
            self.ncols,
            self.nlines,
            np.asarray(area_extent))

    @property
    def start_time(self):
        return datetime.strptime(self.nc.attrs['start_date_time'], '%Y%j%H%M%S')

    @property
    def end_time(self):
        return self.start_time

    def __del__(self):
        try:
            self.nc.close()
        except (IOError, OSError):
            pass
