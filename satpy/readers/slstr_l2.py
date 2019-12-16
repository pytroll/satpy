# -*- coding: utf-8 -*-
# Copyright (c) 2017 Satpy developers
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
""" Reader for Sentinel-3 SLSTR SST data
"""

from datetime import datetime
from satpy.dataset import Dataset
from satpy.readers.file_handlers import BaseFileHandler
import h5py
from xarray.core.dataarray import DataArray
import numpy as np


class SLSTRL2FileHandler(BaseFileHandler):

    def _loadh5f(self, filename):
        def scale_and_offset(var):
            scale_factor = var.attrs['scale_factor'][0]
            add_offset = var.attrs['add_offset'][0]
            fill_value = var.attrs['_FillValue'][0]
            valid_max = var.attrs['valid_max'][0]
            valid_min = var.attrs['valid_min'][0]
            ret = var[0, :].astype(np.float)
            ret[(ret == fill_value) | (ret > valid_max) |
                (ret < valid_min)] = np.nan
            ret = ret * scale_factor + add_offset
            return ret

        with h5py.File(filename, "r") as h5f:
            self.sea_surface_temperature = scale_and_offset(
                h5f['sea_surface_temperature'])
            # Qualit estimation 0-5: no data, cloud, worst, low, acceptable, best
            quality = h5f['quality_level'][0, :, :]
            self.sea_surface_temperature[quality <= 1] = np.nan

            self.sea_ice_fraction = scale_and_offset(h5f['sea_ice_fraction'])
            self.lat = h5f['lat'][:]
            self.lon = h5f['lon'][:]
            self.filename_info['start_time'] = datetime.strptime(
                h5f.attrs['start_time'].decode('UTF-8'), '%Y%m%dT%H%M%SZ')
            self.filename_info['end_time'] = datetime.strptime(
                h5f.attrs['stop_time'].decode('UTF-8'), '%Y%m%dT%H%M%SZ')

    def __init__(self, filename, filename_info, filetype_info):
        super(SLSTRL2FileHandler, self).__init__(filename, filename_info, filetype_info)

        if filename.endswith('tar'):
            import tarfile
            with tarfile.open(name=filename, mode='r') as tf:
                sst_filename = next((name for name in tf.getnames()
                                    if name.endswith('nc') and 'GHRSST-SSTskin' in name))
                self._loadh5f(tf.extractfile(sst_filename))
        else:
            self._loadh5f(filename)

    def get_dataset(self, key, info):
        stdname = info.get('standard_name')
        if stdname in ['latitude']:
            return DataArray(self.lat, name=key.name, attrs=info, dims=('y', 'x'))
        if stdname in ['longitude']:
            return DataArray(self.lon, name=key.name, attrs=info, dims=('y', 'x'))
        if stdname in ['sea_surface_temperature']:
            return DataArray(self.sea_surface_temperature, coords=None, dims=('y', 'x'))
        if stdname in ['sea_ice_fraction']:
            return DataArray(self.sea_ice_fraction, coords=None, dims=('y', 'x'))

    @property
    def start_time(self):
        return self.filename_info['start_time']

    @property
    def end_time(self):
        return self.filename_info['end_time']
