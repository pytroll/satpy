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

from datetime import datetime, timedelta

from satpy.dataset import Dataset
from satpy.readers.file_handlers import BaseFileHandler

from netCDF4 import Dataset as CDF4_Dataset



class ASCATMETOPFileHandler(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(ASCATMETOPFileHandler, self).__init__(filename, filename_info,
                                                      filetype_info)
        ds = CDF4_Dataset(self.filename, 'r')
        self.ds = ds

        self.filename_info['start_time'] = datetime.strptime(ds.getncattr('start_date')
                                                        +' '+ds.getncattr('start_time'),'%Y-%m-%d %H:%M:%S')
        self.filename_info['end_time'] = datetime.strptime(ds.getncattr('stop_date')
                                                        +' '+ds.getncattr('stop_time'),'%Y-%m-%d %H:%M:%S')
        self.filename_info['equator_crossing_time'] = datetime.strptime(ds.getncattr('equator_crossing_date')
                                                        +' '+ds.getncattr('equator_crossing_time'),'%Y-%m-%d %H:%M:%S')
        self.filename_info['orbit_number'] = str(ds.getncattr('orbit_number'))
        print(filename_info)
        self.lons = None
        self.lats = None


    def get_dataset(self, key, info):

        stdname = info.get('standard_name')
        if stdname in ['latitude', 'longitude']:

            if self.lons is None or self.lats is None:
                self.lons = self.ds['lon'][:]
                self.lats = self.ds['lat'][:]

            if info['standard_name'] == 'longitude':
                return Dataset(self.lons, id=key, **info)
            else:
                return Dataset(self.lats, id=key, **info)

        if stdname in ['wind_speed']:
            return Dataset(self.ds['wind_speed'][:], id=key, **info)

        if stdname in ['wind_direction']:
            return Dataset(self.ds['wind_dir'][:], id=key, **info)

        if stdname in ['ice_prob']:
            return Dataset(self.ds['ice_prob'][:], id=key, **info)

        if stdname in ['ice_age']:
            return Dataset(self.ds['ice_age'][:], id=key, **info)





