# -*- coding: utf-8 -*-
# Copyright (c) 2017.
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
""" ScatSat-1 L2B Reader, distributed by Eumetsat in HDF5 format
"""

from datetime import datetime, timedelta
import h5py
import xarray as xr
import xarray.ufuncs as xu
import dask.array as da

from satpy.readers.file_handlers import BaseFileHandler
from satpy import CHUNK_SIZE


class SCATSAT1L2BFileHandler(BaseFileHandler):

    def _read_data(self, filename):
        h5f = h5py.File(filename, "r")
        h5data=h5f['science_data']
        self.filename_info['start_time'] = datetime.strptime(h5data.attrs['Range Beginning Date'],'%Y-%jT%H:%M:%S.%f')
        self.filename_info['end_time'] = datetime.strptime(h5data.attrs['Range Ending Date'],'%Y-%jT%H:%M:%S.%f')
        self.wind_speed_scale = float(h5data.attrs['Wind Speed Selection Scale'])
        self.wind_direction_scale = float(h5data.attrs['Wind Direction Selection Scale'])
        self.latitude_scale = float(h5data.attrs['Latitude Scale'])
        self.longitude_scale = float(h5data.attrs['Longitude Scale'])
        self.lons = h5data['Longitude'][:]*self.longitude_scale
        self.lats = h5data['Latitude'][:]*self.latitude_scale
        self.windspeed=h5data['Wind_speed_selection'][:,:]*self.wind_speed_scale
        self.wind_direction = h5data['Wind_direction_selection'][:,:]*self.wind_direction_scale


    def __init__(self, filename, filename_info, filetype_info):
        super(SCATSAT1L2BFileHandler, self).__init__(filename, filename_info,
                                                      filetype_info)

        self.lons = None
        self.lats = None
        if filename.endswith('bz2'):
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
        stdname = info.get('standard_name')
        if stdname in ['latitude', 'longitude']:

            if info['standard_name'] == 'longitude':
                return xr.DataArray(self.lons, name=key,
                                        attrs=info, dims=('y', 'x'))
            else:
                return xr.DataArray(self.lats, name=key,
                                        attrs=info, dims=('y', 'x'))

        if stdname in ['wind_speed']:
            return xr.DataArray(da.from_array(self.windspeed, chunks=CHUNK_SIZE), name=key,
                                        attrs=info, dims=('y', 'x'))

        if stdname in ['wind_direction']:
            return xr.DataArray(da.from_array(self.wind_direction, chunks=CHUNK_SIZE), name=key,
                                        attrs=info, dims=('y', 'x'))





