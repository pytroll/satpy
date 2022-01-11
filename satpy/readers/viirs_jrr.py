#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2022 Satpy developers
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
"""VIIRS NOAA enterprise L2 product reader.

This module implements readers for the NOAA enterprise level 2 products for the
VIIRS instrument. These replace the 'old' EDR products.
"""


from satpy.readers.file_handlers import BaseFileHandler
from satpy import CHUNK_SIZE
import xarray as xr
import logging

# map platform attributes to Oscar standard name
PLATFORM_MAP = {
    "NPP": "Suomi-NPP",
    "J01": "NOAA-20",
    "J02": "NOAA-21",
}

LOG = logging.getLogger(__name__)


class VIIRSJRRFileHandler(BaseFileHandler):
    """NetCDF4 reader for VIIRS Active Fires."""

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the geo filehandler."""
        super(VIIRSJRRFileHandler, self).__init__(filename, filename_info,
                                                  filetype_info)
        self.nc = xr.open_dataset(self.filename,
                                  decode_cf=True,
                                  mask_and_scale=True,
                                  chunks={'Columns': CHUNK_SIZE,
                                          'Rows': CHUNK_SIZE})
        if 'columns' in self.nc.dims:
            self.nc = self.nc.rename({'Columns': 'x', 'Rows': 'y'})
        elif 'Along_Track_375m' in self.nc.dims:
            self.nc = self.nc.rename({'Along_Scan_375m': 'x', 'Along_Track_375m': 'y'})
            self.nc = self.nc.rename({'Along_Scan_750m': 'x', 'Along_Track_750m': 'y'})

        # For some reason, no 'standard_name' is defined in some netCDF files, so
        # here we manually make the definitions.
        if 'Latitude' in self.nc:
            self.nc['Latitude'].attrs.update({'standard_name': 'latitude'})
        if 'Longitude' in self.nc:
            self.nc['Longitude'].attrs.update({'standard_name': 'longitude'})

        self.algorithm_version = filename_info['platform_shortname']

    def get_dataset(self, dataset_id, info):
        """Get the dataset."""
        ds = self.nc[info['file_key']]

        return ds

    @property
    def start_time(self):
        """Get first date/time when observations were recorded."""
        return self.filename_info['start_time']

    @property
    def end_time(self):
        """Get last date/time when observations were recorded."""
        return self.filename_info.get('end_time', self.start_time)

    @property
    def platform_name(self):
        """Get platform name."""
        platform_path = self.filetype_info['platform_name']
        platform_dict = {'NPP': 'Suomi-NPP',
                         'JPSS-1': 'NOAA-20',
                         'J01': 'NOAA-20',
                         'JPSS-2': 'NOAA-21',
                         'J02': 'NOAA-21'}
        return platform_dict[platform_path]
