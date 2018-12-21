# -*- coding: utf-8 -*-
# Copyright (c) 2018.
#
# Author(s):

#   Eysteinn Már Sigurðsson <eysteinn@vedur.is>

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
""" OSI SAF ASCAT reader for Metop wind products as provided by KNMI
"""

from datetime import datetime

from satpy.readers.file_handlers import BaseFileHandler

from netCDF4 import Dataset as CDF4_Dataset
import xarray as xr
import dask.array as da
from satpy import CHUNK_SIZE


class ASCATMETOPFileHandler(BaseFileHandler):

    def _read_data(self, filename):
        ds = CDF4_Dataset(filename, 'r')
        self.filename_info['start_time'] = datetime.strptime(ds.getncattr('start_date') +
                ' ' + ds.getncattr('start_time'), '%Y-%m-%d %H:%M:%S')
        self.filename_info['end_time'] = datetime.strptime(ds.getncattr('stop_date') +
                ' ' + ds.getncattr('stop_time'), '%Y-%m-%d %H:%M:%S')
        self.filename_info['equator_crossing_time'] = datetime.strptime(ds.getncattr('equator_crossing_date') +
                ' ' + ds.getncattr('equator_crossing_time'), '%Y-%m-%d %H:%M:%S')
        self.filename_info['orbit_number'] = str(ds.getncattr('orbit_number'))
        self.wind_speed = ds['wind_speed'][:]
        self.wind_speed[self.wind_speed.data == ds['wind_speed'].getncattr('missing_value')] = None
        self.wind_direction = ds['wind_dir'][:]
        self.wind_direction[self.wind_direction.data == ds['wind_dir'].getncattr('missing_value')] = None
        self.ice_prob = ds['ice_prob'][:]
        self.ice_prob[self.ice_prob.data == ds['ice_prob'].getncattr('missing_value')] = None
        self.ice_age = ds['ice_age'][:]
        self.ice_age[self.ice_age.data == ds['ice_age'].getncattr('missing_value')] = None
        self.lons = ds['lon'][:]
        self.lons[self.lons > 180.0] -= 360.0
        self.lats = ds['lat'][:]
        ds.close()

    def __init__(self, filename, filename_info, filetype_info):
        super(ASCATMETOPFileHandler, self).__init__(filename, filename_info,
                                                      filetype_info)

        self.lons = None
        self.lats = None
        if filename.endswith('gz'):
            from satpy.readers.utils import unzip_file
            import os
            unzip_filename = None
            try:
                unzip_filename = unzip_file(filename)
                self._read_data(unzip_filename)
            finally:
                if unzip_filename:
                    os.remove(unzip_filename)
        else:
            self._read_data(filename)

    def get_dataset(self, key, info):
        stdname = info['standard_name']
        if stdname in ['longitude']:
            return xr.DataArray(self.lons, name=key,
                                attrs=info, dims=('y', 'x'))
        elif stdname in ['latitude']:
            return xr.DataArray(self.lats, name=key,
                                attrs=info, dims=('y', 'x'))
        elif stdname in ['wind_speed']:
            return xr.DataArray(da.from_array(self.wind_speed, chunks=CHUNK_SIZE), name=key,
                                attrs=info, dims=('y', 'x'))

        elif stdname in ['wind_direction']:
            return xr.DataArray(da.from_array(self.wind_direction, chunks=CHUNK_SIZE), name=key,
                                attrs=info, dims=('y', 'x'))

        elif stdname in ['ice_prob']:
            return xr.DataArray(da.from_array(self.ice_prob, chunks=CHUNK_SIZE), name=key,
                                attrs=info, dims=('y', 'x'))

        elif stdname in ['ice_age']:
            return xr.DataArray(da.from_array(self.ice_age, chunks=CHUNK_SIZE), name=key,
                                attrs=info, dims=('y', 'x'))





