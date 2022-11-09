#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 Satpy developers
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
"""SAFE SAR L2 OCN format reader.

The OCN data contains various parameters, but mainly the wind speed and direction
calculated from SAR data and input model data from ECMWF

Implemented in this reader is the OWI, Ocean Wind field.

See more at ESA webpage https://sentinel.esa.int/web/sentinel/ocean-wind-field-component
"""

import logging

import numpy as np
import xarray as xr

from satpy import CHUNK_SIZE
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)


class SAFENC(BaseFileHandler):
    """Measurement file reader."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init the file reader."""
        super(SAFENC, self).__init__(filename, filename_info,
                                     filetype_info)

        self._start_time = filename_info['start_time']
        self._end_time = filename_info['end_time']
        # For some SAFE packages, fstart_time differs, but start_time is the same
        # To avoid over writing exiting file with same start_time, a solution is to
        # use fstart_time
        self._fstart_time = filename_info['fstart_time']
        self._fend_time = filename_info['fend_time']

        self._polarization = filename_info['polarization']

        self.lats = None
        self.lons = None
        self._shape = None
        self.area = None

        self.nc = xr.open_dataset(filename,
                                  decode_cf=True,
                                  mask_and_scale=False,
                                  chunks={'owiAzSize': CHUNK_SIZE,
                                          'owiRaSize': CHUNK_SIZE})
        self.nc = self.nc.rename({'owiAzSize': 'y'})
        self.nc = self.nc.rename({'owiRaSize': 'x'})
        self.filename = filename

    def get_dataset(self, key, info):
        """Load a dataset."""
        if key['name'] in ['owiLat', 'owiLon']:
            if self.lons is None or self.lats is None:
                self.lons = self.nc['owiLon']
                self.lats = self.nc['owiLat']
            if key['name'] == 'owiLat':
                res = self.lats
            else:
                res = self.lons
            res.attrs = info
        else:
            res = self._get_data_channels(key, info)

        if 'missionName' in self.nc.attrs:
            res.attrs.update({'platform_name': self.nc.attrs['missionName']})

        res.attrs.update({'fstart_time': self._fstart_time})
        res.attrs.update({'fend_time': self._fend_time})

        if not self._shape:
            self._shape = res.shape

        return res

    def _get_data_channels(self, key, info):
        res = self.nc[key['name']]
        if key['name'] in ['owiHs', 'owiWl', 'owiDirmet']:
            res = xr.DataArray(res, dims=['y', 'x', 'oswPartitions'])
        elif key['name'] in ['owiNrcs', 'owiNesz', 'owiNrcsNeszCorr']:
            res = xr.DataArray(res, dims=['y', 'x', 'oswPolarisation'])
        elif key['name'] in ['owiPolarisationName']:
            res = xr.DataArray(res, dims=['owiPolarisation'])
        elif key['name'] in ['owiCalConstObsi', 'owiCalConstInci']:
            res = xr.DataArray(res, dims=['owiIncSize'])
        elif key['name'].startswith('owi'):
            res = xr.DataArray(res, dims=['y', 'x'])
        else:
            res = xr.DataArray(res, dims=['y', 'x'])
        res.attrs.update(info)
        if '_FillValue' in res.attrs:
            res = res.where(res != res.attrs['_FillValue'])
            res.attrs['_FillValue'] = np.nan
        return res

    @property
    def start_time(self):
        """Product start_time, parsed from the measurement file name."""
        return self._start_time

    @property
    def end_time(self):
        """Product end_time, parsed from the measurement file name."""
        return self._end_time

    @property
    def fstart_time(self):
        """Product fstart_time meaning the start time parsed from the SAFE directory."""
        return self._fstart_time

    @property
    def fend_time(self):
        """Product fend_time meaning the end time parsed from the SAFE directory."""
        return self._fend_time
