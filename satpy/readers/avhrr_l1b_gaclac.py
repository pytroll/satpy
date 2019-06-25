#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009-2019 Satpy developers
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
"""Reading and calibrating GAC and LAC avhrr data.

.. todo::

    Fine grained calibration

"""

import logging
from datetime import datetime, timedelta
import xarray as xr
import dask.array as da
from pygac.gac_klm import GACKLMReader
from pygac.gac_pod import GACPODReader
from satpy import CHUNK_SIZE
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)


spacecrafts = {7: "NOAA 15", 3: "NOAA 16", 13: "NOAA 18", 15: "NOAA 19"}

AVHRR3_CHANNEL_NAMES = {"1": 0, "2": 1, "3A": 2, "3B": 3, "4": 4, "5": 5}
AVHRR2_CHANNEL_NAMES = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
AVHRR_CHANNEL_NAMES = {"1": 0, "2": 1, "3": 2, "4": 3}


class GACLACFile(BaseFileHandler):
    """Reader for GAC and LAC data."""

    def __init__(self, filename, filename_info, filetype_info):
        super(GACLACFile, self).__init__(
            filename, filename_info, filetype_info)

        self.reader = None
        self.channels = None
        self._start_time = filename_info['start_time']
        self._end_time = datetime.combine(filename_info['start_time'].date(),
                                          filename_info['end_time'].time())
        if self._end_time < self._start_time:
            self._end_time += timedelta(days=1)
        self.platform_id = filename_info['platform_id']
        if self.platform_id in ['NK', 'NL', 'NM', 'NN', 'NP']:
            self.reader_class = GACKLMReader
            self.chn_dict = AVHRR3_CHANNEL_NAMES
            self.sensor = 'avhrr-3'
        elif self.platform_id in ['NC', 'ND', 'NF', 'NH', 'NJ']:
            self.reader_class = GACPODReader
            self.chn_dict = AVHRR2_CHANNEL_NAMES
            self.sensor = 'avhrr-2'
        else:
            self.reader_class = GACPODReader
            self.chn_dict = AVHRR_CHANNEL_NAMES
            self.sensor = 'avhrr'
        self.filename_info = filename_info

    def get_dataset(self, key, info):
        if self.reader is None:
            self.reader = self.reader_class()
            self.reader.read(self.filename)

        if key.name in ['latitude', 'longitude']:
            if self.reader.lons is None or self.reader.lats is None:
                # self.reader.get_lonlat(clock_drift_adjust=False)
                self.reader.get_lonlat()
            if key.name == 'latitude':
                data = self.reader.lats
            else:
                data = self.reader.lons
        elif key.name in ['sensor_zenith_angle', 'solar_zenith_angle',
                          'sun_sensor_azimuth_difference_angle']:
            sat_azi, sat_zenith, sun_azi, sun_zenith, rel_azi = self.reader.get_angles()
            if key.name == 'sensor_zenith_angle':
                data = sat_zenith
            elif key.name == 'solar_zenith_angle':
                data = sun_zenith
            elif key.name == 'sun_sensor_azimuth_difference_angle':
                data = rel_azi
        else:
            if self.channels is None:
                self.channels = self.reader.get_calibrated_channels()
            data = self.channels[:, :, self.chn_dict[key.name]]

        chunk_cols = data.shape[1]
        chunk_lines = int((CHUNK_SIZE ** 2) / chunk_cols)
        res = xr.DataArray(da.from_array(data, chunks=(chunk_lines, chunk_cols)),
                           dims=['y', 'x'], attrs=info)
        res.attrs['platform_name'] = self.reader.spacecraft_name
        res.attrs['orbit_number'] = self.filename_info['orbit_number']
        res.attrs['sensor'] = self.sensor
        res.attrs['orbital_parameters'] = {'tle': self.reader.get_tle_lines()}
        res['acq_time'] = ('y', self.reader.get_times())
        res['acq_time'].attrs['long_name'] = 'Mean scanline acquisition time'
        return res

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time
