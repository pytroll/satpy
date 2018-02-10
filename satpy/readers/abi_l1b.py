#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 PyTroll developers

# Author(s):

#   Guido della Bruna <Guido.DellaBruna@meteoswiss.ch>
#   Marco Sassi       <Marco.Sassi@meteoswiss.ch>
#   Martin Raspaud    <Martin.Raspaud@smhi.se>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Advance Baseline Imager reader
"""

import logging
import os
from datetime import datetime

import numpy as np
import xarray as xr
import xarray.ufuncs as xu

from pyresample import geometry
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)

PLATFORM_NAMES = {'G16': 'GOES-16'}


class NC_ABI_L1B(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(NC_ABI_L1B, self).__init__(filename, filename_info,
                                         filetype_info)
        self.nc = xr.open_dataset(filename,
                                  decode_cf=True,
                                  mask_and_scale=True,
                                  engine='h5netcdf',
                                  chunks={'x': 1000, 'y': 1000})
        self.nc = self.nc.rename({'t': 'time'})
        platform_shortname = filename_info['platform_shortname']
        self.platform_name = PLATFORM_NAMES.get(platform_shortname)
        self.sensor = 'abi'
        self.nlines, self.ncols = self.nc["Rad"].shape

    def get_shape(self, key, info):
        """Get the shape of the data."""
        return self.nlines, self.ncols

    def get_dataset(self, key, info,
                    xslice=slice(None), yslice=slice(None)):
        """Load a dataset."""
        logger.debug('Reading in get_dataset %s.', key.name)

        radiances = self.nc["Rad"][yslice, xslice].expand_dims('time')

        res = self.calibrate(radiances)

        # convert to satpy standard units
        if res.attrs['units'] == '1':
            res = res * 100
            res.attrs['units'] = '%'

        res.attrs.update({'platform_name': self.platform_name,
                          'sensor': self.sensor,
                          'satellite_latitude': float(self.nc['nominal_satellite_subpoint_lat']),
                          'satellite_longitude': float(self.nc['nominal_satellite_subpoint_lon']),
                          'satellite_altitude': float(self.nc['nominal_satellite_height'])})
        res.attrs.update(key.to_dict())

        return res

    def get_area_def(self, key):
        """Get the area definition of the data at hand."""
        projection = self.nc["goes_imager_projection"]
        a = projection.attrs['semi_major_axis']
        h = projection.attrs['perspective_point_height']
        b = projection.attrs['semi_minor_axis']

        lon_0 = projection.attrs['longitude_of_projection_origin']
        sweep_axis = projection.attrs['sweep_angle_axis'][0]

        # x and y extents in m
        h = float(h)
        x_l = h * self.nc['x'][0]
        x_r = h * self.nc['x'][-1]
        y_l = h * self.nc['y'][-1]
        y_u = h * self.nc['y'][0]
        x_half = (x_r - x_l) / (self.ncols - 1) / 2.
        y_half = (y_u - y_l) / (self.nlines - 1) / 2.
        area_extent = (x_l - x_half, y_l - y_half, x_r + x_half, y_u + y_half)

        proj_dict = {'a': float(a),
                     'b': float(b),
                     'lon_0': float(lon_0),
                     'h': h,
                     'proj': 'geos',
                     'units': 'm',
                     'sweep': sweep_axis}

        area = geometry.AreaDefinition(
            'some_area_name',
            "On-the-fly area",
            'geosabii',
            proj_dict,
            self.ncols,
            self.nlines,
            np.asarray(area_extent))

        return area

    def _vis_calibrate(self, data):
        """Calibrate visible channels to reflectance."""
        solar_irradiance = self.nc['esun']
        esd = self.nc["earth_sun_distance_anomaly_in_AU"].astype(float)

        factor = np.pi * esd * esd / solar_irradiance

        res = data * factor
        res.attrs = data.attrs
        res.attrs['units'] = '1'

        return res

    def _ir_calibrate(self, data):
        """Calibrate IR channels to BT."""
        fk1 = float(self.nc["planck_fk1"])
        fk2 = float(self.nc["planck_fk2"])
        bc1 = float(self.nc["planck_bc1"])
        bc2 = float(self.nc["planck_bc2"])

        res = (fk2 / xu.log(fk1 / data + 1) - bc1) / bc2
        res.attrs = data.attrs
        res.attrs['units'] = 'K'

        return res

    def calibrate(self, data):
        """Calibrate the data."""
        logger.debug("Calibrate")

        ch = int(self.nc["band_id"])

        if ch < 7:
            return self._vis_calibrate(data)
        else:
            return self._ir_calibrate(data)

    @property
    def start_time(self):
        return datetime.strptime(self.nc.attrs['time_coverage_start'], '%Y-%m-%dT%H:%M:%S.%fZ')

    @property
    def end_time(self):
        return datetime.strptime(self.nc.attrs['time_coverage_end'], '%Y-%m-%dT%H:%M:%S.%fZ')
