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
import xarray as xr
import dask.array as da
import datetime as dt
import logging

from satpy.readers.file_handlers import BaseFileHandler
from satpy import CHUNK_SIZE

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
              'water_mixing_ratio_quality': 'QW',
              'water_total_column': 'WC',
              'ozone_total_column': 'OC',
              'surface_skin_temperature': 'Ts',
              'surface_skin_temperature_quality': 'QTs',
              'emissivity': 'E',
              'emissivity_quality': 'QE'}

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

        self.finfo = filename_info
        self.lons = None
        self.lats = None
        self.sensor = 'iasi'

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
                m_data = read_dataset(fid, key)
            else:
                m_data = read_geo(fid, key)
        m_data.attrs.update(info)
        m_data.attrs['sensor'] = self.sensor

        return m_data


def read_dataset(fid, key):
    """Read dataset"""
    dsid = DSET_NAMES[key.name]
    dset = fid["/PWLR/" + dsid]
    if dset.ndim == 3:
        dims = ['y', 'x', 'level']
    else:
        dims = ['y', 'x']
    data = xr.DataArray(da.from_array(dset.value, chunks=CHUNK_SIZE),
                        name=key.name, dims=dims).astype(np.float32)
    data = xr.where(data > 1e30, np.nan, data)

    dset_attrs = dict(dset.attrs)
    data.attrs.update(dset_attrs)

    return data


def read_geo(fid, key):
    """Read geolocation and related datasets."""
    dsid = GEO_NAMES[key.name]
    add_epoch = False
    if "time" in key.name:
        days = fid["/L1C/" + dsid["day"]].value
        msecs = fid["/L1C/" + dsid["msec"]].value
        data = _form_datetimes(days, msecs)
        add_epoch = True
        dtype = np.float64
    else:
        data = fid["/L1C/" + dsid].value
        dtype = np.float32
    data = xr.DataArray(da.from_array(data, chunks=CHUNK_SIZE),
                        name=key.name, dims=['y', 'x']).astype(dtype)

    if add_epoch:
        data.attrs['sensing_time_epoch'] = EPOCH

    return data


def _form_datetimes(days, msecs):
    """Calculate seconds since EPOCH from days and milliseconds for each of IASI scan."""

    all_datetimes = []
    for i in range(days.size):
        day = int(days[i])
        msec = msecs[i]
        scanline_datetimes = []
        for j in range(int(VALUES_PER_SCAN_LINE / 4)):
            usec = 1000 * (j * VIEW_TIME_ADJUSTMENT + msec)
            delta = (dt.timedelta(days=day, microseconds=usec))
            for k in range(4):
                scanline_datetimes.append(delta.total_seconds())
        all_datetimes.append(scanline_datetimes)

    return np.array(all_datetimes, dtype=np.float64)
