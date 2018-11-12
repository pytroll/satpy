#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017.
#
# Author(s):
#
#   inspired in maia.py
#   edited from original autors 
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
"""Reader for NWPSAF AAPP MAIA Cloud product.


"""
import logging
from datetime import datetime

import h5py
import numpy as np
from xarray import DataArray
import dask.array as da

from satpy.readers.file_handlers import BaseFileHandler
from satpy import CHUNK_SIZE

logger = logging.getLogger(__name__)


class EPIC_L1B(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(EPIC_L1B, self).__init__(filename, filename_info, filetype_info)

        self.finfo = filename_info
        self.selected = None
        self.read(filename)

    def read(self, filename):
        self.h5 = h5py.File(filename, 'r')
        missing = -9999.
        lskesy = self.h5.keys()
        self.lat = da.from_array(self.h5[lskesy[0]]['Geolocation']['Earth']['Longitude'][:], chunks=CHUNK_SIZE) 
        self.lon = da.from_array(self.h5[lskesy[0]]['Geolocation']['Earth']['Longitude'][:], chunks=CHUNK_SIZE)
        self.selected = (self.lon > missing)
        self.file_content = {}
        for key in lskesy:
            self.file_content[key] = da.from_array(self.h5[key]['Image'], chunks=CHUNK_SIZE)
#        for key in self.h5[u'HEADER'].keys():
#            self.file_content[key] = self.h5[u'HEADER/' + key][:]
        
    @property
    def start_time(self):
        return datetime.strptime(self.h5.attrs['begin_time'],'%Y-%m-%d %H:%M:%S')

    @property
    def end_time(self):
        return datetime.strptime(self.h5.attrs['end_time'],'%Y-%m-%d %H:%M:%S')

    def get_dataset(self, key, info, out=None):
        """Get a dataset from the file."""

        logger.debug("Reading %s.", key.name)
        if key.name == "latitude":
            values = self.lat
        elif key.name == "longitude":
            values = self.lon
        else:
            values = self.file_content[key.name] / 1000
            values = np.fliplr(np.rot90(values, -1)) 
        selected = np.array(self.selected)
#        if key.name in ("latitude", "longitude"):
#            values = values / 10000.
#        if key.name in ('Tsurf', 'CloudTopPres', 'CloudTopTemp'):
#            goods = values > -9998.
#            selected = np.array(selected & goods)
#            if key.name in ('Tsurf', "Alt_surface", "CloudTopTemp"):
#                values = values / 100.
#            if key.name in ("CloudTopPres"):
#                values = values / 10.
#        else:
#            selected = self.selected
        selected = self.selected
        info.update(self.finfo)

        fill_value = np.nan

        if key.name == 'ct':
            fill_value = 0
            info['_FillValue'] = 0
        ds = DataArray(values, dims=['y', 'x'], attrs=info).where(selected, fill_value)
        
        ds.attrs.update({'platform_name': self.h5.attrs['keywords'].split(',')[0],
                  'sensor': self.h5.attrs['keywords'].split(',')[1].strip(),
                  'satellite_latitude': float(self.h5.attrs['centroid_mean_latitude']),
                  'satellite_longitude': float(self.h5.attrs['centroid_mean_longitude']),
                  'satellite_altitude': float(1500000)})

        # update dataset info with file_info
        return ds
