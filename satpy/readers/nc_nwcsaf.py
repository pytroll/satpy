#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017 Pytroll

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   Adam.Dybbroe <adam.dybbroe@smhi.se>

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

"""Nowcasting SAF common PPS&MSG NetCDF/CF format reader
"""

import logging
from datetime import datetime

import h5netcdf
import numpy as np

from satpy.dataset import Dataset
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)

SENSOR = {'NOAA-19': 'avhrr/3',
          'NOAA-18': 'avhrr/3',
          'NOAA-15': 'avhrr/3',
          'Metop-A': 'avhrr/3',
          'Metop-B': 'avhrr/3',
          'Metop-C': 'avhrr/3',
          'EOS-Aqua': 'modis',
          'EOS-Terra': 'modis',
          'Suomi-NPP': 'viirs',
          'JPSS-1': 'viirs', }

PLATFORM_NAMES = {'MSG1': 'Meteosat-8',
                  'MSG2': 'Meteosat-9',
                  'MSG3': 'Meteosat-10',
                  'MSG4': 'Meteosat-11', }


class NcNWCSAF(BaseFileHandler):

    """NWCSAF PPS&MSG NetCDF reader."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init method"""
        super(NcNWCSAF, self).__init__(filename, filename_info,
                                       filetype_info)
        self.nc = h5netcdf.File(filename, 'r')
        self.pps = False

        try:
            # MSG:
            sat_id = self.nc.attrs['satellite_identifier']
            self.platform_name = PLATFORM_NAMES[sat_id]
        except KeyError:
            # PPS:
            self.platform_name = self.nc.attrs['platform']
            self.pps = True

        self.sensor = SENSOR.get(self.platform_name, 'seviri')

    # def get_shape(self, dsid, ds_info):
    #     """Get the shape of the data."""
    #     raise NotImplementedError
    #     #     return self.nc[dsid.name].shape

    def get_dataset(self, dsid, info, out=None):
        """Load a dataset."""

        logger.debug('Reading %s.', dsid.name)
        variable = self.nc[dsid.name]

        info = {'platform_name': self.platform_name,
                'sensor': self.sensor}

        try:
            values = np.ma.masked_equal(variable[:],
                                        variable.attrs['_FillValue'], copy=False)
        except KeyError:
            values = np.ma.array(variable[:], copy=False)
        if 'scale_factor' in variable.attrs:
            values = values * variable.attrs['scale_factor']
            info['scale_factor'] = variable.attrs['scale_factor']
        if 'add_offset' in variable.attrs:
            values = values + variable.attrs['add_offset']
            info['add_offset'] = variable.attrs['add_offset']
        if 'valid_range' in variable.attrs:
            info['valid_range'] = variable.attrs['valid_range']
        if 'units' in variable.attrs:
            info['units'] = variable.attrs['units']
        if 'standard_name' in variable.attrs:
            info['standard_name'] = variable.attrs['standard_name']

        if self.pps and dsid.name == 'ctth_alti':
            info['valid_range'] = (0., 8500.)
        if self.pps and dsid.name == 'ctth_alti_pal':
            values = values[1:, :]

        proj = Dataset(np.squeeze(values),
                       copy=False,
                       **info)
        return proj

    def get_area_def(self, dsid):
        """Get the area definition of the datasets in the file. 
        Only applicable for MSG products!"""
        if self.pps:
            # PPS:
            raise NotImplementedError

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
        try:
            # MSG:
            return datetime.strptime(self.nc.attrs['time_coverage_start'], '%Y-%m-%dT%H:%M:%SZ')
        except ValueError:
            # PPS:
            return datetime.strptime(self.nc.attrs['time_coverage_start'], '%Y%m%dT%H%M%S%fZ')

    @property
    def end_time(self):
        try:
            # MSG:
            return datetime.strptime(self.nc.attrs['time_coverage_end'], '%Y-%m-%dT%H:%M:%SZ')
        except ValueError:
            # PPS:
            return datetime.strptime(self.nc.attrs['time_coverage_end'], '%Y%m%dT%H%M%S%fZ')
