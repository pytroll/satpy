#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017 Panu Lahtinen

# Author(s):

#   Panu Lahtinen <panu.lahtinen@fmi.fi

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
"""IASI L2 HDF5 files.
"""

import h5py
import numpy as np
import datetime as dt
import logging

from satpy.dataset import Dataset
from satpy.readers.file_handlers import BaseFileHandler

# Scan timing values taken from
# http://oiswww.eumetsat.org/WEBOPS/eps-pg/IASI-L1/IASIL1-PG-4ProdOverview.htm
# Time between each scan in one scanline [ms]
SCAN_STEP_TIME = 8. / 37.
# Duration of one measurement [ms]
SCAN_STARE_DURATION = 151.0
# Time correction used between each 4-footprint measurements
VIEW_TIME_ADJUSTMENT = SCAN_STEP_TIME + SCAN_STARE_DURATION / 2.

VALUES_PER_SCAN_LINE = 120

# Epoch for the dates
EPOCH = dt.datetime(2000, 1, 1)

SHORT_NAMES = {'M01': 'Metop-B',
               'M02': 'Metop-A',
               'M03': 'Metop-C'}

DSET_NAMES = {'ozone_mixing_ratio': 'O',
              'ozone_mixing_ratio_quality': 'QO',
              'pressure': 'P',
              'pressure_quality': 'QP',
              'temperature': 'T',
              'temperature_quality': 'QT',
              'water_mixing_ratio': 'W',
              'water_mixing_ratio_quality': 'QW'}

GEO_NAMES = {'latitude': 'Latitude',
             'longitude': 'Longitude',
             'satellite_azimuth_angle': 'SatAzimuth',
             'satellite_zenith_angle': 'SatZenith',
             'sensing_time': {'day': 'SensingTime_day',
                              'msec': 'SensingTime_msec'},
             'solar_azimuth_angle': 'SunAzimuth',
             'solar_zenith_angle': 'SunZenith'}


LOGGER = logging.getLogger(__name__)


class IASIL2HDF5(BaseFileHandler):

    """File handler for IASI L2 HDF5 files."""

    def __init__(self, filename, filename_info, filetype_info):
        super(IASIL2HDF5, self).__init__(filename, filename_info,
                                         filetype_info)

        self.filename = filename
        self.finfo = filename_info
        self.lons = None
        self.lats = None

        self.mda = {}
        short_name = filename_info['platform_id']
        self.mda['platform_name'] = SHORT_NAMES.get(short_name, short_name)
        self.mda['sensor'] = 'iasi'

    @property
    def start_time(self):
        return self.finfo['start_time']

    @property
    def end_time(self):
        end_time = dt.datetime.combine(self.start_time.date(),
                                       self.finfo['end_time'].time())
        if end_time < self.start_time:
            end_time += dt.timedelta(days=1)
        return end_time

    def get_dataset(self, key, info):
        """Load a dataset"""
        with h5py.File(self.filename, 'r') as fid:
            LOGGER.debug('Reading %s.', key.name)
            if key.name in DSET_NAMES:
                m_data = read_dataset(fid, key, info)
            else:
                m_data = read_geo(fid, key, info)
        m_data.info.update(info)

        return m_data


def read_dataset(fid, key, info):
    """Read dataset"""
    dsid = DSET_NAMES[key.name]
    data = fid["/PWLR/" + dsid].value
    try:
        unit = fid["/PWLR/" + dsid].attrs['units']
        long_name = fid["/PWLR/" + dsid].attrs['long_name']
    except KeyError:
        unit = ''
        long_name = ''

    data = np.ma.masked_where(data > 1e30, data)

    return Dataset(data, copy=False, long_name=long_name,
                   **info)


def read_geo(fid, key, info):
    """Read geolocation and related datasets."""
    dsid = GEO_NAMES[key.name]
    if "time" in key.name:
        days = fid["/L1C/" + dsid["day"]].value
        msecs = fid["/L1C/" + dsid["msec"]].value
        unit = ""
        data = _form_datetimes(days, msecs)
    else:
        data = fid["/L1C/" + dsid].value
        unit = fid["/L1C/" + dsid].attrs['units']

    data = Dataset(data, copy=False, **info)

    return data


def _form_datetimes(days, msecs):
    """Form datetimes from days and milliseconds relative to EPOCH for
    each of IASI scans."""

    all_datetimes = []
    for i in range(days.size):
        day = int(days[i])
        msec = msecs[i]
        scanline_datetimes = []
        for j in range(VALUES_PER_SCAN_LINE / 4):
            usec = 1000 * (j * VIEW_TIME_ADJUSTMENT + msec)
            for k in range(4):
                delta = (dt.timedelta(days=day, microseconds=usec))
                scanline_datetimes.append(EPOCH + delta)
        all_datetimes.append(np.array(scanline_datetimes))

    return np.array(all_datetimes)
