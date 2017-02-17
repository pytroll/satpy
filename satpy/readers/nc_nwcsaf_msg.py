#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017 Martin Raspaud

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

"""Nowcasting SAF MSG NetCDF4 format reader
"""

import logging
from datetime import datetime

import h5netcdf
import numpy as np

from pyresample.utils import get_area_def
from satpy.dataset import Dataset
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)

PLATFORM_NAMES = {'MSG1': 'Meteosat-8',
                  'MSG2': 'Meteosat-9',
                  'MSG3': 'Meteosat-10',
                  'MSG4': 'Meteosat-11', }


class NcNWCSAFMSG(BaseFileHandler):
    """NWCSAF MSG NetCDF reader."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init method."""
        super(NcNWCSAFMSG, self).__init__(filename, filename_info,
                                          filetype_info)
        self.nc = h5netcdf.File(filename, 'r')
        self.sensor = 'seviri'
        sat_id = self.nc.attrs['satellite_identifier']
        self.platform_name = PLATFORM_NAMES[sat_id]

    def get_dataset(self, key, info):
        """Load a dataset."""
        logger.debug('Reading %s.', key.name)
        variable = self.nc[key.name]

        try:
            values = np.ma.masked_equal(variable[:],
                                        variable.attrs['_FillValue'], copy=False)
        except KeyError:
            values = np.ma.array(variable[:], copy=False)
        if 'scale_factor' in variable.attrs:
            values = values * variable.attrs['scale_factor']
        if 'add_offset' in variable.attrs:
            values = values + variable.attrs['add_offset']

        info = {'platform_name': self.platform_name,
                'sensor': self.sensor}

        if 'valid_range' in variable.attrs:
            info['valid_range'] = variable.attrs['valid_range']
        if 'units' in variable.attrs:
            info['units'] = variable.attrs['units']

        proj = Dataset(values,
                       copy=False,
                       **info)
        return proj

    def get_area_def(self, dsid):
        """Get the area definition of the datasets in the file."""
        if dsid.name.endswith('_pal'):
            raise NotImplementedError

        proj_str = self.nc.attrs['gdal_projection'] + ' +units=km'

        nlines, ncols = self.nc[dsid.name].shape

        area_extent = (float(self.nc.attrs['gdal_xgeo_up_left']) / 1000,
                       float(self.nc.attrs['gdal_ygeo_low_right']) / 1000,
                       float(self.nc.attrs['gdal_xgeo_low_right']) / 1000,
                       float(self.nc.attrs['gdal_ygeo_up_left']) / 1000)

        area = get_area_def('some_area_name',
                            "On-the-fly area",
                            'geosmsg',
                            proj_str,
                            ncols,
                            nlines,
                            area_extent)

        return area

    @property
    def start_time(self):
        return datetime.strptime(self.nc.attrs['time_coverage_start'], '%Y-%m-%dT%H:%M:%SZ')

    @property
    def end_time(self):
        return datetime.strptime(self.nc.attrs['time_coverage_end'], '%Y-%m-%dT%H:%M:%SZ')
