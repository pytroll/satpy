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
"""ScatSat-1 L2B Reader, distributed by Eumetsat in HDF5 format
"""

from datetime import datetime
import h5py

from satpy.dataset import Dataset
from satpy.readers.file_handlers import BaseFileHandler


class SCATSAT1L2BFileHandler(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(SCATSAT1L2BFileHandler, self).__init__(filename, filename_info, filetype_info)
        self.h5f = h5py.File(self.filename, "r")
        h5data = self.h5f['science_data']

        self.filename_info['start_time'] = datetime.strptime(h5data.attrs['Range Beginning Date'], '%Y-%jT%H:%M:%S.%f')
        self.filename_info['end_time'] = datetime.strptime(h5data.attrs['Range Ending Date'], '%Y-%jT%H:%M:%S.%f')

        self.lons = None
        self.lats = None

        self.wind_speed_scale = float(h5data.attrs['Wind Speed Selection Scale'])
        self.wind_direction_scale = float(h5data.attrs['Wind Direction Selection Scale'])
        self.latitude_scale = float(h5data.attrs['Latitude Scale'])
        self.longitude_scale = float(h5data.attrs['Longitude Scale'])

    def get_dataset(self, key, info):
        h5data = self.h5f['science_data']
        stdname = info.get('standard_name')

        if stdname in ['latitude', 'longitude']:

            if self.lons is None or self.lats is None:
                self.lons = h5data['Longitude'][:]*self.longitude_scale
                self.lats = h5data['Latitude'][:]*self.latitude_scale

            if info['standard_name'] == 'longitude':
                return Dataset(self.lons, id=key, **info)
            else:
                return Dataset(self.lats, id=key, **info)

        if stdname in ['wind_speed']:
            windspeed = h5data['Wind_speed_selection'][:, :] * self.wind_speed_scale
            return Dataset(windspeed, id=key, **info)

        if stdname in ['wind_direction']:
            wind_direction = h5data['Wind_direction_selection'][:, :] * self.wind_direction_scale
            return Dataset(wind_direction, id=key, **info)
