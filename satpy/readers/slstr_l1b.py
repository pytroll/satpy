#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016-2020 Satpy developers
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
"""SLSTR L1b reader."""

import logging
import os
import re
import warnings
from datetime import datetime

import dask.array as da
import numpy as np
import xarray as xr

from satpy import CHUNK_SIZE
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)

PLATFORM_NAMES = {'S3A': 'Sentinel-3A',
                  'S3B': 'Sentinel-3B'}

# These are the default channel adjustment factors.
# Defined in the product notice: S3.PN-SLSTR-L1.08
# https://sentinel.esa.int/documents/247904/2731673/Sentinel-3A-and-3B-SLSTR-Product-Notice-Level-1B-SL-1-RBT-at-NRT-and-NTC.pdf
CHANCALIB_FACTORS = {'S1_nadir': 0.97,
                     'S2_nadir': 0.98,
                     'S3_nadir': 0.98,
                     'S4_nadir': 1.0,
                     'S5_nadir': 1.11,
                     'S6_nadir': 1.13,
                     'S7_nadir': 1.0,
                     'S8_nadir': 1.0,
                     'S9_nadir': 1.0,
                     'S1_oblique': 0.94,
                     'S2_oblique': 0.95,
                     'S3_oblique': 0.95,
                     'S4_oblique': 1.0,
                     'S5_oblique': 1.04,
                     'S6_oblique': 1.07,
                     'S7_oblique': 1.0,
                     'S8_oblique': 1.0,
                     'S9_oblique': 1.0, }


class NCSLSTRGeo(BaseFileHandler):
    """Filehandler for geo info."""

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the geo filehandler."""
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
        """Load a dataset."""
        logger.debug('Reading %s.', key['name'])
        file_key = info['file_key'].format(view=key['view'].name[0],
                                           stripe=key['stripe'].name)
        try:
            variable = self.nc[file_key]
        except KeyError:
            return

        info = info.copy()
        info.update(variable.attrs)

        variable.attrs = info
        return variable

    @property
    def start_time(self):
        """Get the start time."""
        return datetime.strptime(self.nc.attrs['start_time'], '%Y-%m-%dT%H:%M:%S.%fZ')

    @property
    def end_time(self):
        """Get the end time."""
        return datetime.strptime(self.nc.attrs['stop_time'], '%Y-%m-%dT%H:%M:%S.%fZ')


class NCSLSTR1B(BaseFileHandler):
    """Filehandler for l1 SLSTR data.

    By default, the calibration factors recommended by EUMETSAT are applied.
    This is required as the SLSTR VIS channels are producing slightly incorrect
    radiances that require adjustment.
    Satpy uses the radiance corrections in S3.PN-SLSTR-L1.08, checked 11/03/2022.
    User-supplied coefficients can be passed via the `user_calibration` kwarg
    This should be a dict of channel names (such as `S1_nadir`, `S8_oblique`).

    For example::

        calib_dict = {'S1_nadir': 1.12}
        scene = satpy.Scene(filenames,
                            reader='slstr-l1b',
                            reader_kwargs={'user_calib': calib_dict})

    Will multiply S1 nadir radiances by 1.12.
    """

    def __init__(self, filename, filename_info, filetype_info,
                 user_calibration=None):
        """Initialize the SLSTR l1 data filehandler."""
        super(NCSLSTR1B, self).__init__(filename, filename_info,
                                        filetype_info)

        self.nc = xr.open_dataset(self.filename,
                                  decode_cf=True,
                                  mask_and_scale=True,
                                  chunks={'columns': CHUNK_SIZE,
                                          'rows': CHUNK_SIZE})
        self.nc = self.nc.rename({'columns': 'x', 'rows': 'y'})
        self.channel = filename_info['dataset_name']
        self.stripe = filename_info['stripe']
        views = {'n': 'nadir', 'o': 'oblique'}
        self.view = views[filename_info['view']]
        cal_file = os.path.join(os.path.dirname(self.filename), 'viscal.nc')
        self.cal = xr.open_dataset(cal_file,
                                   decode_cf=True,
                                   mask_and_scale=True,
                                   chunks={'views': CHUNK_SIZE})
        indices_file = os.path.join(os.path.dirname(self.filename),
                                    'indices_{}{}.nc'.format(self.stripe, self.view[0]))
        self.indices = xr.open_dataset(indices_file,
                                       decode_cf=True,
                                       mask_and_scale=True,
                                       chunks={'columns': CHUNK_SIZE,
                                               'rows': CHUNK_SIZE})
        self.indices = self.indices.rename({'columns': 'x', 'rows': 'y'})

        self.platform_name = PLATFORM_NAMES[filename_info['mission_id']]
        self.sensor = 'slstr'
        if isinstance(user_calibration, dict):
            self.usercalib = user_calibration
        else:
            self.usercalib = None

    def _apply_radiance_adjustment(self, radiances):
        """Adjust SLSTR radiances with default or user supplied values."""
        chan_name = self.channel + '_' + self.view
        adjust_fac = None
        if self.usercalib is not None:
            # If user supplied adjustment, use it.
            if chan_name in self.usercalib:
                adjust_fac = self.usercalib[chan_name]
        if adjust_fac is None:
            if chan_name in CHANCALIB_FACTORS:
                adjust_fac = CHANCALIB_FACTORS[chan_name]
            else:
                warnings.warn("Warning: No radiance adjustment supplied " +
                              "for channel " + chan_name)
                return radiances
        return radiances * adjust_fac

    @staticmethod
    def _cal_rad(rad, didx, solar_flux=None):
        """Calibrate."""
        indices = np.isfinite(didx)
        rad[indices] /= solar_flux[didx[indices].astype(int)]
        return rad

    def get_dataset(self, key, info):
        """Load a dataset."""
        if (self.channel not in key['name'] or
                self.stripe != key['stripe'].name or
                self.view != key['view'].name):
            return
        logger.debug('Reading %s.', key['name'])
        if key['calibration'] == 'brightness_temperature':
            variable = self.nc['{}_BT_{}{}'.format(self.channel, self.stripe, self.view[0])]
        else:
            variable = self.nc['{}_radiance_{}{}'.format(self.channel, self.stripe, self.view[0])]
        radiances = self._apply_radiance_adjustment(variable)
        units = variable.attrs['units']
        if key['calibration'] == 'reflectance':
            # TODO take into account sun-earth distance
            solar_flux = self.cal[re.sub('_[^_]*$', '', key['name']) + '_solar_irradiances']
            d_index = self.indices['detector_{}{}'.format(self.stripe, self.view[0])]
            idx = 0 if self.view[0] == 'n' else 1  # 0: Nadir view, 1: oblique (check).
            radiances.data = da.map_blocks(
                self._cal_rad, radiances.data, d_index.data, solar_flux=solar_flux[:, idx].values)
            radiances *= np.pi * 100
            units = '%'

        info = info.copy()
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
        """Get the start time."""
        return datetime.strptime(self.nc.attrs['start_time'], '%Y-%m-%dT%H:%M:%S.%fZ')

    @property
    def end_time(self):
        """Get the end time."""
        return datetime.strptime(self.nc.attrs['stop_time'], '%Y-%m-%dT%H:%M:%S.%fZ')


class NCSLSTRAngles(BaseFileHandler):
    """Filehandler for angles."""

    def _loadcart(self, fname):
        """Load a cartesian file of appropriate type."""
        cartf = xr.open_dataset(fname,
                                decode_cf=True,
                                mask_and_scale=True,
                                chunks={'columns': CHUNK_SIZE,
                                        'rows': CHUNK_SIZE})
        return cartf

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the angles reader."""
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

        carta_file = os.path.join(
            os.path.dirname(self.filename), 'cartesian_a{}.nc'.format(self.view[0]))
        carti_file = os.path.join(
            os.path.dirname(self.filename), 'cartesian_i{}.nc'.format(self.view[0]))
        cartx_file = os.path.join(
            os.path.dirname(self.filename), 'cartesian_tx.nc')
        self.carta = self._loadcart(carta_file)
        self.carti = self._loadcart(carti_file)
        self.cartx = self._loadcart(cartx_file)

    def get_dataset(self, key, info):
        """Load a dataset."""
        if not key['view'].name.startswith(self.view[0]):
            return
        logger.debug('Reading %s.', key['name'])
        # Check if file_key is specified in the yaml
        file_key = info['file_key'].format(view=key['view'].name[0])

        variable = self.nc[file_key]
        l_step = self.nc.attrs.get('al_subsampling_factor', 1)
        c_step = self.nc.attrs.get('ac_subsampling_factor', 16)

        if key.get('resolution', 1000) == 500:
            l_step *= 2
            c_step *= 2

        if c_step != 1 or l_step != 1:
            logger.debug('Interpolating %s.', key['name'])
            # TODO: do it in cartesian coordinates ! pbs at date line and
            # possible
            tie_x = self.cartx['x_tx'].data[0, :][::-1]
            tie_y = self.cartx['y_tx'].data[:, 0]
            if key.get('resolution', 1000) == 500:
                full_x = self.carta['x_a' + self.view[0]].data
                full_y = self.carta['y_a' + self.view[0]].data
            else:
                full_x = self.carti['x_i' + self.view[0]].data
                full_y = self.carti['y_i' + self.view[0]].data

            variable = variable.fillna(0)
            variable.attrs['resolution'] = key.get('resolution', 1000)

            from scipy.interpolate import RectBivariateSpline
            spl = RectBivariateSpline(
                tie_y, tie_x, variable.data[:, ::-1])

            values = spl.ev(full_y, full_x)

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
        """Get the start time."""
        return datetime.strptime(self.nc.attrs['start_time'], '%Y-%m-%dT%H:%M:%S.%fZ')

    @property
    def end_time(self):
        """Get the end time."""
        return datetime.strptime(self.nc.attrs['stop_time'], '%Y-%m-%dT%H:%M:%S.%fZ')


class NCSLSTRFlag(BaseFileHandler):
    """File handler for flags."""

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the flag reader."""
        super(NCSLSTRFlag, self).__init__(filename, filename_info,
                                          filetype_info)
        self.nc = xr.open_dataset(self.filename,
                                  decode_cf=True,
                                  mask_and_scale=True,
                                  chunks={'columns': CHUNK_SIZE,
                                          'rows': CHUNK_SIZE})
        self.nc = self.nc.rename({'columns': 'x', 'rows': 'y'})
        self.stripe = filename_info['stripe']
        views = {'n': 'nadir', 'o': 'oblique'}
        self.view = views[filename_info['view']]
        # TODO: get metadata from the manifest file (xfdumanifest.xml)
        self.platform_name = PLATFORM_NAMES[filename_info['mission_id']]
        self.sensor = 'slstr'

    def get_dataset(self, key, info):
        """Load a dataset."""
        if (self.stripe != key['stripe'].name or
                self.view != key['view'].name):
            return
        logger.debug('Reading %s.', key['name'])
        file_key = info['file_key'].format(view=key['view'].name[0],
                                           stripe=key['stripe'].name)
        variable = self.nc[file_key]

        info = info.copy()
        info.update(variable.attrs)
        info.update(key.to_dict())
        info.update(dict(platform_name=self.platform_name,
                         sensor=self.sensor))

        variable.attrs = info
        return variable

    @property
    def start_time(self):
        """Get the start time."""
        return datetime.strptime(self.nc.attrs['start_time'], '%Y-%m-%dT%H:%M:%S.%fZ')

    @property
    def end_time(self):
        """Get the end time."""
        return datetime.strptime(self.nc.attrs['stop_time'], '%Y-%m-%dT%H:%M:%S.%fZ')
