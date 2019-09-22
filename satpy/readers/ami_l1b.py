#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Satpy developers
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
"""Advanced Meteorological Imager reader for the Level 1b NetCDF4 format."""

import logging
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

from pyresample import geometry
from satpy.readers.file_handlers import BaseFileHandler
from satpy import CHUNK_SIZE

logger = logging.getLogger(__name__)

PLATFORM_NAMES = {
    'GK-2A': 'GEO-KOMPSAT-2A',
    'GK-2B': 'GEO-KOMPSAT-2B',
}


class AMIL1bNetCDF(BaseFileHandler):
    """Base reader for AMI L1B NetCDF4 files."""

    def __init__(self, filename, filename_info, filetype_info):
        """Open the NetCDF file with xarray and prepare the Dataset for reading."""
        super(AMIL1bNetCDF, self).__init__(filename, filename_info, filetype_info)
        self.nc = xr.open_dataset(self.filename,
                                  decode_cf=True,
                                  mask_and_scale=False,
                                  chunks={'dim_image_x': CHUNK_SIZE, 'dim_image_y': CHUNK_SIZE})
        self.nc = self.nc.rename_dims({'dim_image_x': 'x', 'dim_image_y': 'y'})

        platform_shortname = self.nc.attrs['satellite_name']
        self.platform_name = PLATFORM_NAMES.get(platform_shortname)
        self.sensor = 'ami'

    @property
    def start_time(self):
        """Get observation start time."""
        base = datetime(2000, 1, 1, 12, 0, 0)
        return base + timedelta(seconds=self.nc.attrs['observation_start_time'])

    @property
    def end_time(self):
        """Get observation end time."""
        base = datetime(2000, 1, 1, 12, 0, 0)
        return base + timedelta(seconds=self.nc.attrs['observation_end_time'])

    def get_area_def(self, dsid):
        """Get area definition for this file."""
        a = self.nc.attrs['earth_equatorial_radius']
        b = self.nc.attrs['earth_polar_radius']
        h = self.nc.attrs['nominal_satellite_height']
        lon_0 = self.nc.attrs['sub_longitude'] * 180 / np.pi  # it's in radians?
        cols = self.nc.attrs['number_of_columns']
        rows = self.nc.attrs['number_of_lines']
        obs_mode = self.nc.attrs['observation_mode']
        resolution = self.nc.attrs['channel_spatial_resolution']

        cfac = self.nc.attrs['cfac']
        coff = self.nc.attrs['coff']
        lfac = self.nc.attrs['lfac']
        loff = self.nc.attrs['loff']
        area_extent = (
            (0 - coff - 0.5) * cfac,
            (0 - loff - 0.5) * lfac,
            (cols - coff + 0.5) * cfac,
            (rows - loff + 0.5) * lfac,
        )

        proj_dict = {'proj': 'geos',
                     'lon_0': float(lon_0),
                     'a': float(a),
                     'b': float(b),
                     'h': h,
                     'units': 'm'}

        fg_area_def = geometry.AreaDefinition(
            'ami_geos_{}'.format(obs_mode.lower()),
            'AMI {} Area at {} resolution'.format(obs_mode, resolution),
            'ami_fixed_grid',
            proj_dict,
            cols,
            rows,
            np.asarray(area_extent))

        return fg_area_def

    def get_dataset(self, dataset_id, ds_info):
        """Load a dataset as a xarray DataArray."""
        file_key = ds_info.get('file_key', dataset_id.name)
        data = self.nc[file_key]
        return data
