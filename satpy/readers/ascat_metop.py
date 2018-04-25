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
import xarray as xr
import dask.array as da

from satpy.readers.file_handlers import BaseFileHandler
from satpy import CHUNK_SIZE

from netCDF4 import Dataset as CDF4_Dataset

SHORT_NAMES = {'metopa': 'Metop-A',
               'metopb': 'Metop-B'}


class ASCATMETOPFileHandler(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(ASCATMETOPFileHandler, self).__init__(filename, filename_info,
                                                    filetype_info)
        ds = CDF4_Dataset(self.filename, 'r')
        self.ds = ds

        self.finfo = filename_info
        self.finfo['start_time'] = \
            datetime.strptime(ds.getncattr('start_date')
                              + ' ' +
                              ds.getncattr('start_time'), '%Y-%m-%d %H:%M:%S')
        self.finfo['end_time'] = \
            datetime.strptime(ds.getncattr('stop_date')
                              + ' ' +
                              ds.getncattr('stop_time'), '%Y-%m-%d %H:%M:%S')
        self.finfo['platform_name'] = SHORT_NAMES[filename_info['platform_id']]
        self.lats = None
        self.lons = None

    def get_dataset(self, key, info):
        info['platform_name'] = self.finfo['platform_name']
        stdname = info.get('standard_name')
        chunk_size = CHUNK_SIZE
        if stdname in ['latitude', 'longitude']:
            if self.lons is None or self.lats is None:
                self.lons = self.ds['lon'][:]
                self.lats = self.ds['lat'][:]

            if info['standard_name'] == 'longitude':
                return xr.DataArray(da.from_array(self.lons,
                                                  chunks=chunk_size),
                                    name=key.name, attrs=info, dims=('y', 'x'))
            else:
                return xr.DataArray(da.from_array(self.lats,
                                                  chunks=chunk_size),
                                    name=key.name, attrs=info, dims=('y', 'x'))

        if stdname in ['wind_speed']:
            return xr.DataArray(da.from_array(self.ds['wind_speed'][:],
                                              chunks=chunk_size),
                                name=key.name, attrs=info, dims=('y', 'x'))

        if stdname in ['wind_direction']:
            return xr.DataArray(da.from_array(self.ds['wind_dir'][:],
                                              chunks=chunk_size),
                                name=key.name, attrs=info, dims=('y', 'x'))

        if stdname in ['ice_prob']:
            return xr.DataArray(da.from_array(self.ds['ice_prob'][:],
                                              chunks=chunk_size),
                                name=key.name, attrs=info, dims=('y', 'x'))

        if stdname in ['ice_age']:
            return xr.DataArray(da.from_array(self.ds['ice_age'][:],
                                              chunks=chunk_size),
                                name=key.name, attrs=info, dims=('y', 'x'))

    @property
    def start_time(self):
        return self.finfo['start_time']

    @property
    def end_time(self):
        return self.finfo['end_time']
