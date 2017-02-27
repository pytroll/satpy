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
"""Compact viirs format.
"""

import logging
import os
from datetime import datetime

import h5netcdf
import numpy as np

from satpy.dataset import Dataset, DatasetID
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)

PLATFORM_NAMES = {'S3A': 'Sentinel-3A',
                  'S3B': 'Sentinel-3B'}


class NCSLSTRGeo(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(NCSLSTRGeo, self).__init__(filename, filename_info,
                                         filetype_info)
        self.nc = h5netcdf.File(filename, 'r')
        self.cache = {}

    def get_dataset(self, key, info):
        """Load a dataset
        """

        logger.debug('Reading %s.', key.name)

        try:
            variable = self.nc[info['file_key']]
        except KeyError:
            return

        ds = (np.ma.masked_equal(variable[:],
                                 variable.attrs['_FillValue']) *
              (variable.attrs.get('scale_factor', 1) * 1.0) +
              variable.attrs.get('add_offset', 0))
        ds.mask = np.ma.getmaskarray(ds)

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


class NCSLSTR1B(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(NCSLSTR1B, self).__init__(filename, filename_info,
                                        filetype_info)
        self.nc = h5netcdf.File(filename, 'r')
        self.channel = filename_info['dataset_name']
        self.view = 'n'  # n for nadir, o for oblique
        cal_file = os.path.join(os.path.dirname(
            filename), 'viscal.nc')
        self.cal = h5netcdf.File(cal_file, 'r')
        indices_file = os.path.join(os.path.dirname(filename),
                                    'indices_a{}.nc'.format(self.view))
        self.indices = h5netcdf.File(indices_file, 'r')
        # TODO: get metadata from the manifest file (xfdumanifest.xml)
        self.platform_name = PLATFORM_NAMES[filename_info['mission_id']]
        self.sensor = 'slstr'

    def get_dataset(self, key, info):
        """Load a dataset."""
        if self.channel != key.name:
            return
        logger.debug('Reading %s.', key.name)
        if key.calibration == 'brightness_temperature':
            variable = self.nc[self.channel + '_BT_i' + self.view]
        else:
            variable = self.nc[self.channel + '_radiance_a' + self.view]

        radiances = (np.ma.masked_equal(variable[:],
                                        variable.attrs['_FillValue'], copy=False) *
                     variable.attrs.get('scale_factor', 1) +
                     variable.attrs.get('add_offset', 0))
        units = variable.attrs['units']
        if key.calibration == 'reflectance':
            # TODO take into account sun-earth distance
            solar_flux = self.cal[key.name + '_solar_irradiances'][:]
            d_index = np.ma.masked_equal(self.indices['detector_a'
                                                      + self.view][:],
                                         self.indices['detector_a'
                                                      + self.view].attrs[
                                             '_FillValue'],
                                         copy=False)
            idx = 0  # Nadir view
            radiances[np.logical_not(np.ma.getmaskarray(
                d_index))] /= solar_flux[d_index.compressed(), idx]
            radiances *= np.pi * 100
            units = '%'

        proj = Dataset(radiances,
                       name=key.name,
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


class NCSLSTRAngles(BaseFileHandler):

    view = 'n'

    datasets = {'satellite_azimuth_angle': 'satellite_azimuth_t' + view,
                'satellite_zenith_angle': 'satellite_zenith_t' + view,
                'solar_azimuth_angle': 'solar_azimuth_t' + view,
                'solar_zenith_angle': 'solar_zenith_t' + view}

    def __init__(self, filename, filename_info, filetype_info):
        super(NCSLSTRAngles, self).__init__(filename, filename_info,
                                            filetype_info)
        self.nc = None
        # TODO: get metadata from the manifest file (xfdumanifest.xml)
        self.platform_name = PLATFORM_NAMES[filename_info['mission_id']]
        self.sensor = 'slstr'
        self._start_time = filename_info['start_time']
        self._end_time = filename_info['end_time']
        cart_file = os.path.join(
            os.path.dirname(self.filename), 'cartesian_i{}.nc'.format(self.view))
        self.cart = h5netcdf.File(cart_file, 'r')
        cartx_file = os.path.join(
            os.path.dirname(self.filename), 'cartesian_tx.nc')
        self.cartx = h5netcdf.File(cartx_file, 'r')

    def get_dataset(self, key, info):
        """Load a dataset
        """
        if key.name not in self.datasets:
            return

        if self.nc is None:
            self.nc = h5netcdf.File(self.filename, 'r')

        logger.debug('Reading %s.', key.name)
        variable = self.nc[self.datasets[key.name]]

        values = (np.ma.masked_equal(variable[:],
                                     variable.attrs['_FillValue'], copy=False) *
                  variable.attrs.get('scale_factor', 1) +
                  variable.attrs.get('add_offset', 0))
        values = np.ma.masked_invalid(values, copy=False)
        units = variable.attrs['units']

        l_step = self.nc.attrs.get('al_subsampling_factor', 1)
        c_step = self.nc.attrs.get('ac_subsampling_factor', 16)

        if c_step != 1 or l_step != 1:
            logger.debug('Interpolating %s.', key.name)

            # TODO: do it in cartesian coordinates ! pbs at date line and
            # possible
            tie_x_var = self.cartx['x_tx']
            tie_x = (np.ma.masked_equal(tie_x_var[0, :],
                                        tie_x_var.attrs['_FillValue'],
                                        copy=False) *
                     tie_x_var.attrs.get('scale_factor', 1) +
                     tie_x_var.attrs.get('add_offset', 0))

            tie_y_var = self.cartx['y_tx']
            tie_y = (np.ma.masked_equal(tie_y_var[:, 0],
                                        tie_y_var.attrs['_FillValue'],
                                        copy=False) *
                     tie_y_var.attrs.get('scale_factor', 1) +
                     tie_y_var.attrs.get('add_offset', 0))

            full_x_var = self.cart['x_i' + self.view]
            full_x = (np.ma.masked_equal(full_x_var[:],
                                         full_x_var.attrs['_FillValue'],
                                         copy=False) *
                      full_x_var.attrs.get('scale_factor', 1) +
                      full_x_var.attrs.get('add_offset', 0))

            full_y_var = self.cart['y_i' + self.view]
            full_y = (np.ma.masked_equal(full_y_var[:],
                                         full_y_var.attrs['_FillValue'],
                                         copy=False) *
                      full_y_var.attrs.get('scale_factor', 1) +
                      full_y_var.attrs.get('add_offset', 0))

            from scipy.interpolate import RectBivariateSpline
            spl = RectBivariateSpline(
                tie_y, tie_x[::-1], values[:, ::-1].filled(0))

            interpolated = spl.ev(full_y.compressed(),
                                  full_x.compressed())
            interpolated = np.ma.masked_invalid(interpolated, copy=False)
            values = np.ma.empty(full_y.shape,
                                 dtype=values.dtype)
            values[np.logical_not(np.ma.getmaskarray(full_y))] = interpolated
            values.mask = full_y.mask

        proj = Dataset(values,
                       copy=False,
                       units=units,
                       platform_name=self.platform_name,
                       standard_name=variable.attrs['standard_name'],
                       sensor=self.sensor)
        return proj

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time
