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
""" Sentinel-5p Tropomi NetCDF reader
"""

from netCDF4 import Dataset as CDF4_Dataset
import xarray as xr
import dask.array as da
import numpy as np
from satpy import CHUNK_SIZE
from satpy.readers.file_handlers import BaseFileHandler

class TROPOMIL2FileHandler(BaseFileHandler):

    """Tropomi NetCDF file readed."""

    def __init__(self, filename, filename_info, filetype_info):
        super(TROPOMIL2FileHandler, self).__init__(filename, filename_info,
                                                   filetype_info)
        self.filename = filename
        cdf4ds = CDF4_Dataset(filename, 'r')
        self.lats = cdf4ds['PRODUCT'].variables['latitude'][0, :]
        self.lons = cdf4ds['PRODUCT'].variables['longitude'][0, :]
        self.qa_value = cdf4ds['PRODUCT']['qa_value'][0, :]
        self.cdf4ds = cdf4ds

    def get_dataset(self, key, info):
        stdname = info['standard_name']

        if stdname in ['longitude']:
            return xr.DataArray(self.lons, name=stdname,
                                attrs=info, dims=('y', 'x'))
        elif stdname in ['latitude']:
            return xr.DataArray(self.lats, name=stdname,
                                attrs=info, dims=('y', 'x'))
        elif stdname in self.cdf4ds['PRODUCT'].variables:
            var = self.cdf4ds['PRODUCT'].variables[stdname][0, :]
            var = np.ma.masked_where(self.qa_value < 0.5, var).filled(np.nan)
            info['nc_attrs'] = {}
            for attrsname in self.cdf4ds['PRODUCT'].variables[stdname].ncattrs():
                info['nc_attrs'][str(attrsname)] = \
                    str(self.cdf4ds['PRODUCT'].variables[stdname].getncattr(attrsname))
            return xr.DataArray(da.from_array(var, chunks=CHUNK_SIZE), \
                name=stdname, attrs=info, dims=('y', 'x'))

        return None

    @property
    def sensor_names(self):
        return ['tropomi']
