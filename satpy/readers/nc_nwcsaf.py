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

import numpy as np
import xarray as xr

from pyresample.utils import get_area_def
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
        self.nc = xr.open_dataset(filename,
                                  decode_cf=True,
                                  mask_and_scale=False,
                                  engine='h5netcdf',
                                  chunks=1000)

        self.nc = self.nc.rename({'nx': 'x', 'ny': 'y'})

        self.pps = False

        try:
            # MSG:
            sat_id = self.nc.attrs['satellite_identifier']
            try:
                self.platform_name = PLATFORM_NAMES[sat_id]
            except KeyError:
                self.platform_name = PLATFORM_NAMES[sat_id.astype(str)]
        except KeyError:
            # PPS:
            self.platform_name = self.nc.attrs['platform']
            self.pps = True

        self.sensor = SENSOR.get(self.platform_name, 'seviri')

    def get_dataset(self, dsid, info):
        """Load a dataset."""

        logger.debug('Reading %s.', dsid.name)
        variable = self.nc[dsid.name]

        if 'scale_factor' in variable.attrs or 'add_offset' in variable.attrs:
            scale = variable.attrs.get('scale_factor', 1)
            offset = variable.attrs.get('add_offset', 0)
            if np.issubdtype((scale + offset).dtype, np.floating):
                if '_FillValue' in variable.attrs:
                    variable = variable.where(
                        variable != variable.attrs['_FillValue'])
                if 'valid_range' in variable.attrs:
                    variable = variable.where(
                        variable <= variable.attrs['valid_range'][1])
                    variable = variable.where(
                        variable >= variable.attrs['valid_range'][0])
                if 'valid_max' in variable.attrs:
                    variable = variable.where(
                        variable <= variable.attrs['valid_max'])
                if 'valid_min' in variable.attrs:
                    variable = variable.where(
                        variable >= variable.attrs['valid_min'])

            variable = variable * scale + offset

        variable.attrs.update({'platform_name': self.platform_name,
                               'sensor': self.sensor})

        variable.attrs.setdefault('units', '1')

        ancillary_names = variable.attrs.get('ancillary_variables', '')
        try:
            variable.attrs['ancillary_variables'] = ancillary_names.split()
        except AttributeError:
            pass

        if 'standard_name' in info:
            import h5py
            if isinstance(variable.attrs.get('standard_name'),
                          h5py._hl.base.Empty):
                variable.attrs.pop('standard_name')
            variable.attrs.setdefault('standard_name', info['standard_name'])

        if self.pps and dsid.name == 'ctth_alti':
            variable.attrs['valid_range'] = (0., 8500.)
        if self.pps and dsid.name == 'ctth_alti_pal':
            variable = variable[1:, :]

        return variable

    def get_area_def(self, dsid):
        """Get the area definition of the datasets in the file.

        Only applicable for MSG products!
        """
        if self.pps:
            # PPS:
            raise NotImplementedError

        if dsid.name.endswith('_pal'):
            raise NotImplementedError

        try:
            proj_str = self.nc.attrs['gdal_projection'] + ' +units=km'
        except TypeError:
            proj_str = self.nc.attrs['gdal_projection'].decode() + ' +units=km'

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
            try:
                return datetime.strptime(self.nc.attrs['time_coverage_start'],
                                         '%Y-%m-%dT%H:%M:%SZ')
            except TypeError:
                return datetime.strptime(self.nc.attrs['time_coverage_start'].astype(str),
                                         '%Y-%m-%dT%H:%M:%SZ')
        except ValueError:
            # PPS:
            return datetime.strptime(self.nc.attrs['time_coverage_start'],
                                     '%Y%m%dT%H%M%S%fZ')

    @property
    def end_time(self):
        try:
            # MSG:
            try:
                return datetime.strptime(self.nc.attrs['time_coverage_end'],
                                         '%Y-%m-%dT%H:%M:%SZ')
            except TypeError:
                return datetime.strptime(self.nc.attrs['time_coverage_end'].astype(str),
                                         '%Y-%m-%dT%H:%M:%SZ')
        except ValueError:
            # PPS:
            return datetime.strptime(self.nc.attrs['time_coverage_end'],
                                     '%Y%m%dT%H%M%S%fZ')
