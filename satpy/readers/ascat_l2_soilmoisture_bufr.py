#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2021 Satpy developers
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
"""ASCAT Soil moisture product reader for BUFR messages.

Based on the IASI L2 SO2 BUFR reader.

"""

import logging
from datetime import datetime

import dask.array as da
import numpy as np
import xarray as xr

try:
    import eccodes as ec
except ImportError as e:
    raise ImportError(
        """Missing eccodes-python and/or eccodes C-library installation. Use conda to install eccodes.
           Error: """, e)

from satpy import CHUNK_SIZE
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger('AscatSoilMoistureBufr')


class AscatSoilMoistureBufr(BaseFileHandler):
    """File handler for the ASCAT Soil Moisture BUFR product."""

    def __init__(self, filename, filename_info, filetype_info, **kwargs):
        """Initialise the file handler for the ASCAT Soil Moisture BUFR data."""
        super(AscatSoilMoistureBufr, self).__init__(filename, filename_info, filetype_info)

        start_time, end_time = self.get_start_end_date()

        self.metadata = {}
        self.metadata['start_time'] = start_time
        self.metadata['end_time'] = end_time

    @property
    def start_time(self):
        """Return the start time of data acqusition."""
        return self.metadata['start_time']

    @property
    def end_time(self):
        """Return the end time of data acquisition."""
        return self.metadata['end_time']

    @property
    def platform_name(self):
        """Return spacecraft name."""
        return self.filename_info['platform']

    def extract_msg_date_extremes(self, bufr, date_min=None, date_max=None):
        """Extract the minimum and maximum dates from a single bufr message."""
        ec.codes_set(bufr, 'unpack', 1)
        size = ec.codes_get(bufr, 'numberOfSubsets')
        years = np.resize(ec.codes_get_array(bufr, 'year'), size)
        months = np.resize(ec.codes_get_array(bufr, 'month'), size)
        days = np.resize(ec.codes_get_array(bufr, 'day'), size)
        hours = np.resize(ec.codes_get_array(bufr, 'hour'), size)
        minutes = np.resize(ec.codes_get_array(bufr, 'minute'), size)
        seconds = np.resize(ec.codes_get_array(bufr, 'second'), size)
        for year, month, day, hour, minute, second in zip(years, months, days, hours, minutes, seconds):
            time_stamp = datetime(year, month, day, hour, minute, second)
            date_min = time_stamp if not date_min else min(date_min, time_stamp)
            date_max = time_stamp if not date_max else max(date_max, time_stamp)
        return date_min, date_max

    def get_start_end_date(self):
        """Get the first and last date from the bufr file."""
        with open(self.filename, 'rb') as fh:
            date_min = None
            date_max = None
            while True:
                # get handle for message
                bufr = ec.codes_bufr_new_from_file(fh)
                if bufr is None:
                    break
                date_min, date_max = self.extract_msg_date_extremes(bufr, date_min, date_max)
            return date_min, date_max

    def get_bufr_data(self, key):
        """Get BUFR data by key."""
        attr = np.array([])
        with open(self.filename, 'rb') as fh:
            while True:
                # get handle for message
                bufr = ec.codes_bufr_new_from_file(fh)
                if bufr is None:
                    break
                ec.codes_set(bufr, 'unpack', 1)
                tmp = ec.codes_get_array(bufr, key, float)
                if len(tmp) == 1:
                    size = ec.codes_get(bufr, 'numberOfSubsets')
                    tmp = np.resize(tmp, size)
                attr = np.append(attr, tmp)
                ec.codes_release(bufr)
        return attr

    def get_dataset(self, dataset_id, dataset_info):
        """Get dataset using the BUFR key in dataset_info."""
        arr = self.get_bufr_data(dataset_info['key'])
        if 'fill_value' in dataset_info:
            arr[arr == dataset_info['fill_value']] = np.nan
        arr = da.from_array(arr, chunks=CHUNK_SIZE)
        xarr = xr.DataArray(arr, dims=["y"], name=dataset_info['name'])
        xarr.attrs['platform_name'] = self.platform_name
        xarr.attrs.update(dataset_info)

        return xarr
