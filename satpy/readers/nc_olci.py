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

import numpy as np

import dask.array as da
import xarray as xr
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
        self.nc = xr.open_dataset(filename,
                                  decode_cf=True,
                                  mask_and_scale=True,
                                  engine='h5netcdf',
                                  chunks={'columns': 1000, 'rows': 1000})
        self.nc = self.nc.rename({'columns': 'x', 'rows': 'y'})

    def get_dataset(self, key, info):
        """Load a dataset."""

        logger.debug('Reading %s.', key.name)
        variable = self.nc[key.name]

        return variable

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
        self.nc = xr.open_dataset(filename,
                                  decode_cf=True,
                                  mask_and_scale=True,
                                  engine='h5netcdf',
                                  chunks={'columns': 1000, 'rows': 1000})

        self.nc = self.nc.rename({'columns': 'x', 'rows': 'y'})

        self.channel = filename_info['dataset_name']
        cal_file = os.path.join(os.path.dirname(
            filename), 'instrument_data.nc')
        # self.cal = h5netcdf.File(cal_file, 'r')
        self.cal = xr.open_dataset(cal_file,
                                   decode_cf=True,
                                   mask_and_scale=True,
                                   engine='h5netcdf',
                                   chunks={'columns': 1000, 'rows': 1000})

        self.cal = self.cal.rename({'columns': 'x', 'rows': 'y'})

        # TODO: get metadata from the manifest file (xfdumanifest.xml)
        self.platform_name = PLATFORM_NAMES[filename_info['mission_id']]
        self.sensor = 'olci'

    def get_dataset(self, key, info):
        """Load a dataset
        """
        if self.channel != key.name:
            return
        logger.debug('Reading %s.', key.name)
        radiances = self.nc[self.channel + '_radiance']

        if key.calibration == 'reflectance':
            solar_flux = self.cal['solar_flux']
            d_index = self.cal['detector_index']

            idx = int(key.name[2:]) - 1
            sflux = solar_flux.values[idx, d_index.fillna(0).values.astype(int)]
            radiances = radiances / sflux * np.pi * 100
            radiances.attrs['units'] = '%'

        radiances.attrs['platform_name'] = self.platform_name
        radiances.attrs['sensor'] = self.sensor
        radiances.attrs.update(key.to_dict())
        return radiances

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
            self.nc = xr.open_dataset(self.filename,
                                      decode_cf=True,
                                      mask_and_scale=True,
                                      engine='h5netcdf',
                                      chunks={'tie_columns': 1000,
                                              'tie_rows': 1000})

            self.nc = self.nc.rename({'tie_columns': 'x', 'tie_rows': 'y'})
        logger.debug('Reading %s.', key.name)

        l_step = self.nc.attrs['al_subsampling_factor']
        c_step = self.nc.attrs['ac_subsampling_factor']

        if (c_step != 1 or l_step != 1) and key.name not in self.cache:

            if key.name.startswith('satellite'):
                zen = self.nc[self.datasets['satellite_zenith_angle']]
                zattrs = zen.attrs
                azi = self.nc[self.datasets['satellite_azimuth_angle']]
                aattrs = azi.attrs
            elif key.name.startswith('solar'):
                zen = self.nc[self.datasets['solar_zenith_angle']]
                zattrs = zen.attrs
                azi = self.nc[self.datasets['solar_azimuth_angle']]
                aattrs = azi.attrs
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
            satint = Interpolator([x.values, y.values, z.values],
                                  (tie_lines, tie_cols),
                                  (lines, cols),
                                  along_track_order,
                                  cross_track_order)
            (x, y, z, ) = satint.interpolate()
            x = xr.DataArray(da.from_array(x, chunks=(1000, 1000)),
                             dims=['y', 'x'])
            y = xr.DataArray(da.from_array(y, chunks=(1000, 1000)),
                             dims=['y', 'x'])
            z = xr.DataArray(da.from_array(z, chunks=(1000, 1000)),
                             dims=['y', 'x'])

            azi, zen = xyz2angle(x, y, z)
            azi.attrs = aattrs
            zen.attrs = zattrs

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

        values.attrs['units'] = attrs['units']
        values.attrs['platform_name'] = self.platform_name
        values.attrs['sensor'] = self.sensor

        values.attrs.update(key.to_dict())
        return values

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time
