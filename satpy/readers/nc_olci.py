#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Martin Raspaud

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

"""Sentinel-3 OLCI reader
"""

import logging
import os
from datetime import datetime

import h5netcdf
import numpy as np

from satpy.dataset import Dataset, DatasetID
from satpy.readers.file_handlers import BaseFileHandler
from satpy.utils import angle2xyz, lonlat2xyz, xyz2angle, xyz2lonlat

logger = logging.getLogger(__name__)

PLATFORM_NAMES = {'S3A': 'Sentinel-3A',
                  'S3B': 'Sentinel-3B'}


class NCOLCIGeo(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(NCOLCIGeo, self).__init__(filename, filename_info,
                                        filetype_info)
        self.nc = h5netcdf.File(filename, 'r')

        self.cache = {}

    def get_dataset(self, key, info):
        """Load a dataset
        """

        logger.debug('Reading %s.', key.name)
        variable = self.nc[key.name]

        ds = (np.ma.masked_equal(variable[:],
                                 variable.attrs['_FillValue']) *
              (variable.attrs['scale_factor'] * 1.0) +
              variable.attrs.get('add_offset', 0))

        proj = Dataset(ds,
                       copy=False,
                       **info)
        return proj

    @property
    def start_time(self):
        return datetime.strptime(self.nc.attrs['start_time'], '%Y-%m-%dT%H:%M:%S.%fZ')

    @property
    def end_time(self):
        return datetime.strptime(self.nc.attrs['stop_time'], '%Y-%m-%dT%H:%M:%S.%fZ')


class NCOLCI1B(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(NCOLCI1B, self).__init__(filename, filename_info,
                                       filetype_info)
        self.nc = h5netcdf.File(filename, 'r')
        self.channel = filename_info['dataset_name']
        cal_file = os.path.join(os.path.dirname(
            filename), 'instrument_data.nc')
        self.cal = h5netcdf.File(cal_file, 'r')
        # TODO: get metadata from the manifest file (xfdumanifest.xml)
        self.platform_name = PLATFORM_NAMES[filename_info['mission_id']]
        self.sensor = 'olci'

    def get_dataset(self, key, info):
        """Load a dataset
        """
        if self.channel != key.name:
            return
        logger.debug('Reading %s.', key.name)
        variable = self.nc[self.channel + '_radiance']

        radiances = (np.ma.masked_equal(variable[:],
                                        variable.attrs['_FillValue'], copy=False) *
                     variable.attrs['scale_factor'] +
                     variable.attrs['add_offset'])
        units = variable.attrs['units']
        if key.calibration == 'reflectance':
            solar_flux = self.cal['solar_flux'][:]
            d_index = np.ma.masked_equal(self.cal['detector_index'][:],
                                         self.cal['detector_index'].attrs[
                                             '_FillValue'],
                                         copy=False)
            idx = int(key.name[2:]) - 1
            radiances /= solar_flux[idx, d_index]
            radiances *= np.pi * 100
            units = '%'

        proj = Dataset(radiances,
                       copy=False,
                       units=units,
                       platform_name=self.platform_name,
                       sensor=self.sensor)
        proj.info.update(key.to_dict())
        return proj

    @property
    def start_time(self):
        return datetime.strptime(self.nc.attrs['start_time'], '%Y-%m-%dT%H:%M:%S.%fZ')

    @property
    def end_time(self):
        return datetime.strptime(self.nc.attrs['stop_time'], '%Y-%m-%dT%H:%M:%S.%fZ')


class NCOLCIAngles(BaseFileHandler):

    datasets = {'satellite_azimuth_angle': 'OAA',
                'satellite_zenith_angle': 'OZA',
                'solar_azimuth_angle': 'SAA',
                'solar_zenith_angle': 'SZA'}

    def __init__(self, filename, filename_info, filetype_info):
        super(NCOLCIAngles, self).__init__(filename, filename_info,
                                           filetype_info)
        self.nc = None
        # TODO: get metadata from the manifest file (xfdumanifest.xml)
        self.platform_name = PLATFORM_NAMES[filename_info['mission_id']]
        self.sensor = 'olci'
        self.cache = {}
        self._start_time = filename_info['start_time']
        self._end_time = filename_info['end_time']

    def _get_scaled_variable(self, name):
        """Get a scaled variable."""
        variable = self.nc[name]

        values = (np.ma.masked_equal(variable[:],
                                     variable.attrs['_FillValue'], copy=False) *
                  variable.attrs.get('scale_factor', 1) +
                  variable.attrs.get('add_offset', 0))
        return values, variable.attrs

    def get_dataset(self, key, info):
        """Load a dataset."""
        if key.name not in self.datasets:
            return

        if self.nc is None:
            self.nc = h5netcdf.File(self.filename, 'r')

        logger.debug('Reading %s.', key.name)

        l_step = self.nc.attrs['al_subsampling_factor']
        c_step = self.nc.attrs['ac_subsampling_factor']

        if (c_step != 1 or l_step != 1) and key.name not in self.cache:

            if key.name.startswith('satellite'):
                zen, zattrs = self._get_scaled_variable(
                    self.datasets['satellite_zenith_angle'])
                azi, aattrs = self._get_scaled_variable(
                    self.datasets['satellite_azimuth_angle'])
            elif key.name.startswith('solar'):
                zen, zattrs = self._get_scaled_variable(
                    self.datasets['solar_zenith_angle'])
                azi, aattrs = self._get_scaled_variable(
                    self.datasets['solar_azimuth_angle'])
            else:
                raise NotImplementedError("Don't know how to read " + key.name)

            x, y, z = angle2xyz(azi, zen)
            shape = x.shape

            from geotiepoints.interpolator import Interpolator
            tie_lines = np.arange(
                0, (shape[0] - 1) * l_step + 1, l_step)
            tie_cols = np.arange(0, (shape[1] - 1) * c_step + 1, c_step)
            lines = np.arange((shape[0] - 1) * l_step + 1)
            cols = np.arange((shape[1] - 1) * c_step + 1)
            along_track_order = 1
            cross_track_order = 3
            satint = Interpolator([x, y, z],
                                  (tie_lines, tie_cols),
                                  (lines, cols),
                                  along_track_order,
                                  cross_track_order)
            (x, y, z, ) = satint.interpolate()

            azi, zen = xyz2angle(x, y, z)

            if 'zenith' in key.name:
                values, attrs = zen, zattrs
            elif 'azimuth' in key.name:
                values, attrs = azi, aattrs
            else:
                raise NotImplementedError("Don't know how to read " + key.name)

            if key.name.startswith('satellite'):
                self.cache['satellite_zenith_angle'] = zen, zattrs
                self.cache['satellite_azimuth_angle'] = azi, aattrs
            elif key.name.startswith('solar'):
                self.cache['solar_zenith_angle'] = zen, zattrs
                self.cache['solar_azimuth_angle'] = azi, aattrs

        elif key.name in self.cache:
            values, attrs = self.cache[key.name]
        else:
            values, attrs = self._get_scaled_variable(self.datasets[key.name])

        units = attrs['units']

        proj = Dataset(values,
                       copy=False,
                       units=units,
                       platform_name=self.platform_name,
                       sensor=self.sensor)
        proj.info.update(key.to_dict())
        return proj

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time
