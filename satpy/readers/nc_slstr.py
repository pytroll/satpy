#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Martin Raspaud

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   Ulrik Egede <u.egede@imperial.ac.uk>

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
import re

from datetime import datetime

import numpy as np
import xarray as xr
import dask.array as da

from satpy.readers.file_handlers import BaseFileHandler
from satpy import CHUNK_SIZE

logger = logging.getLogger(__name__)

PLATFORM_NAMES = {'S3A': 'Sentinel-3A',
                  'S3B': 'Sentinel-3B'}


class NCSLSTRGeo(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(NCSLSTRGeo, self).__init__(filename, filename_info,
                                         filetype_info)
        self.nc = xr.open_dataset(self.filename,
                                  decode_cf=True,
                                  mask_and_scale=True,
                                  chunks={'columns': CHUNK_SIZE,
                                          'rows': CHUNK_SIZE})
        self.nc = self.nc.rename({'columns': 'x', 'rows': 'y'})

        self.cache = {}

    def get_dataset(self, key, info):
        """Load a dataset
        """

        logger.debug('Reading %s.', key.name)

        try:
            variable = self.nc[info['file_key']]
        except KeyError:
            return

        info.update(variable.attrs)

        variable.attrs = info
        return variable

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

        self.nc = xr.open_dataset(self.filename,
                                  decode_cf=True,
                                  mask_and_scale=True,
                                  chunks={'columns': CHUNK_SIZE,
                                          'rows': CHUNK_SIZE})
        self.nc = self.nc.rename({'columns': 'x', 'rows': 'y'})
        self.channel = filename_info['dataset_name']
        self.stripe = self.filename[-5]
        self.view = self.filename[-4]
        cal_file = os.path.join(os.path.dirname(self.filename), 'viscal.nc')
        self.cal = xr.open_dataset(cal_file,
                                   decode_cf=True,
                                   mask_and_scale=True,
                                   chunks={'views': CHUNK_SIZE})
        indices_file = os.path.join(os.path.dirname(self.filename),
                                    'indices_{}{}.nc'.format(self.stripe, self.view))
        self.indices = xr.open_dataset(indices_file,
                                       decode_cf=True,
                                       mask_and_scale=True,
                                       chunks={'columns': CHUNK_SIZE,
                                               'rows': CHUNK_SIZE})
        self.indices = self.indices.rename({'columns': 'x', 'rows': 'y'})

        self.platform_name = PLATFORM_NAMES[filename_info['mission_id']]
        self.sensor = 'slstr'

    def get_dataset(self, key, info):
        """Load a dataset."""

        if self.channel not in key.name:
            return

        logger.debug('Reading %s.', key.name)
        if key.calibration == 'brightness_temperature':
            variable = self.nc['{}_BT_{}{}'.format(self.channel, self.stripe, self.view)]
        else:
            variable = self.nc['{}_radiance_{}{}'.format(self.channel, self.stripe, self.view)]

        radiances = variable
        units = variable.attrs['units']

        if key.calibration == 'reflectance':
            # TODO take into account sun-earth distance
            solar_flux = self.cal[re.sub('_[^_]*$', '', key.name) + '_solar_irradiances']
            d_index = self.indices['detector_{}{}'.format(self.stripe, self.view)]
            idx = 0 if self.view == 'n' else 1   # 0: Nadir view, 1: oblique (check).

            def cal_rad(rad, didx, solar_flux=None):
                indices = np.isfinite(didx)
                rad[indices] /= solar_flux[didx[indices].astype(int)]
                return rad

            radiances.data = da.map_blocks(
                cal_rad, radiances.data, d_index.data, solar_flux=solar_flux[:, idx].values)

            radiances *= np.pi * 100
            units = '%'

        info.update(radiances.attrs)
        info.update(key.to_dict())
        info.update(dict(units=units,
                         platform_name=self.platform_name,
                         sensor=self.sensor,
                         view=self.view))

        radiances.attrs = info
        return radiances

    @property
    def start_time(self):
        return datetime.strptime(self.nc.attrs['start_time'], '%Y-%m-%dT%H:%M:%S.%fZ')

    @property
    def end_time(self):
        return datetime.strptime(self.nc.attrs['stop_time'], '%Y-%m-%dT%H:%M:%S.%fZ')


class NCSLSTRAngles(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):

        super(NCSLSTRAngles, self).__init__(filename, filename_info,
                                            filetype_info)

        self.nc = xr.open_dataset(self.filename,
                                  decode_cf=True,
                                  mask_and_scale=True,
                                  chunks={'columns': CHUNK_SIZE,
                                          'rows': CHUNK_SIZE})

        # TODO: get metadata from the manifest file (xfdumanifest.xml)
        self.platform_name = PLATFORM_NAMES[filename_info['mission_id']]
        self.sensor = 'slstr'

        self.view = filename_info['view']
        self._start_time = filename_info['start_time']
        self._end_time = filename_info['end_time']

        cart_file = os.path.join(
            os.path.dirname(self.filename), 'cartesian_i{}.nc'.format(self.view))
        self.cart = xr.open_dataset(cart_file,
                                    decode_cf=True,
                                    mask_and_scale=True,
                                    chunks={'columns': CHUNK_SIZE,
                                            'rows': CHUNK_SIZE})
        cartx_file = os.path.join(
            os.path.dirname(self.filename), 'cartesian_tx.nc')
        self.cartx = xr.open_dataset(cartx_file,
                                     decode_cf=True,
                                     mask_and_scale=True,
                                     chunks={'columns': CHUNK_SIZE,
                                             'rows': CHUNK_SIZE})

    def get_dataset(self, key, info):
        """Load a dataset
        """
        logger.debug('Reading %s.', key.name)

        # Check if file_key is specified in the yaml
        file_key = info.get('file_key', key.name)

        variable = self.nc[file_key]

        l_step = self.nc.attrs.get('al_subsampling_factor', 1)
        c_step = self.nc.attrs.get('ac_subsampling_factor', 16)

        if c_step != 1 or l_step != 1:
            logger.debug('Interpolating %s.', key.name)

            # TODO: do it in cartesian coordinates ! pbs at date line and
            # possible
            tie_x = self.cartx['x_tx'].data[0, :][::-1]
            tie_y = self.cartx['y_tx'].data[:, 0]
            full_x = self.cart['x_i' + self.view].data
            full_y = self.cart['y_i' + self.view].data

            variable = variable.fillna(0)

            from scipy.interpolate import RectBivariateSpline
            spl = RectBivariateSpline(
                tie_y, tie_x, variable.data[:, ::-1])

            valid = np.isfinite(full_y)

            interpolated = spl.ev(full_y[valid],
                                  full_x[valid])

            values = np.full_like(full_y, np.nan,
                                  dtype=variable.dtype)

            values[valid] = interpolated
            values = np.ma.masked_invalid(values, copy=False)

            variable = xr.DataArray(da.from_array(values, chunks=(CHUNK_SIZE, CHUNK_SIZE)),
                                    dims=['y', 'x'], attrs=variable.attrs)

        variable.attrs['platform_name'] = self.platform_name
        variable.attrs['sensor'] = self.sensor

        if 'units' not in variable.attrs:
            variable.attrs['units'] = 'degrees'

        variable.attrs.update(key.to_dict())

        return variable

    @property
    def start_time(self):
        return datetime.strptime(self.nc.attrs['start_time'], '%Y-%m-%dT%H:%M:%S.%fZ')

    @property
    def end_time(self):
        return datetime.strptime(self.nc.attrs['stop_time'], '%Y-%m-%dT%H:%M:%S.%fZ')


class NCSLSTRFlag(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(NCSLSTRFlag, self).__init__(filename, filename_info,
                                          filetype_info)
        self.nc = xr.open_dataset(self.filename,
                                  decode_cf=True,
                                  mask_and_scale=True,
                                  chunks={'columns': CHUNK_SIZE,
                                          'rows': CHUNK_SIZE})
        self.nc = self.nc.rename({'columns': 'x', 'rows': 'y'})
        self.stripe = self.filename[-5]
        self.view = self.filename[-4]
        # TODO: get metadata from the manifest file (xfdumanifest.xml)
        self.platform_name = PLATFORM_NAMES[filename_info['mission_id']]
        self.sensor = 'slstr'

    def get_dataset(self, key, info):
        """Load a dataset."""
        logger.debug('Reading %s.', key.name)

        variable = self.nc[key.name]

        info.update(variable.attrs)
        info.update(key.to_dict())
        info.update(dict(platform_name=self.platform_name,
                         sensor=self.sensor))

        variable.attrs = info
        return variable

    @property
    def start_time(self):
        return datetime.strptime(self.nc.attrs['start_time'], '%Y-%m-%dT%H:%M:%S.%fZ')

    @property
    def end_time(self):
        return datetime.strptime(self.nc.attrs['stop_time'], '%Y-%m-%dT%H:%M:%S.%fZ')
