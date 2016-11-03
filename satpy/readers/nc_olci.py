#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Martin Raspaud

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

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
"""Compact viirs format.
"""

import logging
import os
from datetime import datetime

import numpy as np

import h5netcdf
from satpy.projectable import Projectable
from satpy.readers import DatasetID
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)


class NCOLCIGeo(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(NCOLCIGeo, self).__init__(filename, filename_info,
                                        filetype_info)
        self.nc = h5netcdf.File(filename, 'r')

        self.cache = {}

    def get_dataset(self, key, info=None):
        """Load a dataset
        """

        if key in self.cache:
            return self.cache[key]

        logger.debug('Reading %s.', key.name)
        variable = self.nc[key.name]

        ds = (np.ma.masked_equal(variable[:],
                                 variable.attrs['_FillValue']) *
              (variable.attrs['scale_factor'] * 1.0) +
              variable.attrs.get('add_offset', 0))
        self.cache[key] = ds
        return ds

    def get_area(self, navid, nav_info, lon_out, lat_out):
        """Load an area.
        """
        lon_out[:] = self.get_dataset(DatasetID('longitude'))
        lat_out[:] = self.get_dataset(DatasetID('latitude'))

    @property
    def start_time(self):
        return datetime.strptime(self.nc.attrs['start_time'], '%Y-%m-%dT%H:%M:%S.%fZ')

    @property
    def end_time(self):
        return datetime.strptime(self.nc.attrs['stop_time'], '%Y-%m-%dT%H:%M:%S.%fZ')


class NCOLCI1B(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(NCOLCI1B, self).__init__(filename, filename_info,
                                       filetype_info)
        self.nc = h5netcdf.File(filename, 'r')
        self.channel = filename_info['dataset_name']
        cal_file = os.path.join(os.path.dirname(
            filename), 'instrument_data.nc')
        self.cal = h5netcdf.File(cal_file, 'r')

    def get_dataset(self, key, info):
        """Load a dataset
        """
        if self.channel != key.name:
            return
        logger.debug('Reading %s.', key.name)
        variable = self.nc[self.channel + '_radiance']

        radiances = (np.ma.masked_equal(variable[:],
                                        variable.attrs['_FillValue'], copy=False) *
                     variable.attrs['scale_factor'] +
                     variable.attrs['add_offset'])
        units = variable.attrs['units']
        if key.calibration == 'reflectance':
            solar_flux = self.cal['solar_flux'][:]
            d_index = np.ma.masked_equal(self.cal['detector_index'][:],
                                         self.cal['detector_index'].attrs[
                                             '_FillValue'],
                                         copy=False)
            idx = int(key.name[2:]) - 1
            radiances /= solar_flux[idx, d_index]
            radiances *= np.pi * 100
            units = '%'

        proj = Projectable(radiances,
                           copy=False,
                           units=units)
        return proj

    @property
    def start_time(self):
        return datetime.strptime(self.nc.attrs['start_time'], '%Y-%m-%dT%H:%M:%S.%fZ')

    @property
    def end_time(self):
        return datetime.strptime(self.nc.attrs['stop_time'], '%Y-%m-%dT%H:%M:%S.%fZ')
