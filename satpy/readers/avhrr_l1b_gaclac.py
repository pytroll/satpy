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
"""Reading and calibrating GAC and LAC AVHRR data.

.. todo::

    Fine grained calibration

"""

import logging
from datetime import datetime, timedelta
import xarray as xr
import dask.array as da
import numpy as np
from pygac.gac_klm import GACKLMReader
from pygac.gac_pod import GACPODReader
import pygac.utils
from satpy import CHUNK_SIZE
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)


spacecrafts = {7: "NOAA 15", 3: "NOAA 16", 13: "NOAA 18", 15: "NOAA 19"}

AVHRR3_CHANNEL_NAMES = {"1": 0, "2": 1, "3A": 2, "3B": 3, "4": 4, "5": 5}
AVHRR2_CHANNEL_NAMES = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
AVHRR_CHANNEL_NAMES = {"1": 0, "2": 1, "3": 2, "4": 3}
ANGLES = ('sensor_zenith_angle', 'sensor_azimuth_angle', 'solar_zenith_angle',
          'solar_azimuth_angle', 'sun_sensor_azimuth_difference_angle')


class GACLACFile(BaseFileHandler):
    """Reader for GAC and LAC data."""

    def __init__(self, filename, filename_info, filetype_info,
                 start_line=None, end_line=None, strip_invalid_coords=True,
                 interpolate_coords=True, adjust_clock_drift=True,
                 tle_dir=None, tle_name=None, tle_thresh=7):
        """
        Args:
            start_line: User defined start scanline
            end_line: User defined end scanline
            strip_invalid_coords: Strip scanlines with invalid coordinates in
                the beginning/end of the orbit
            interpolate_coords: Interpolate coordinates from every eighth pixel
                to all pixels.
            adjust_clock_drift: Adjust the geolocation to compensate for the
                clock error (POD satellites only).
            tle_dir: Directory holding Two-Line-Element (TLE) files
            tle_name: Filename pattern of TLE files.
            tle_thresh: Maximum number of days between observation and nearest
                TLE
        """
        super(GACLACFile, self).__init__(
            filename, filename_info, filetype_info)

        self.start_line = start_line
        self.end_line = end_line
        self.strip_invalid_coords = strip_invalid_coords
        self.interpolate_coords = interpolate_coords
        self.adjust_clock_drift = adjust_clock_drift
        self.tle_dir = tle_dir
        self.tle_name = tle_name
        self.tle_thresh = tle_thresh
        self.reader = None
        self.channels = None
        self.angles = None
        self.qual_flags = None
        self.midnight_scanline = None
        self.missing_scanlines = None
        self.first_valid_lat = None
        self.last_valid_lat = None
        self._start_time = filename_info['start_time']
        self._end_time = datetime.combine(filename_info['start_time'].date(),
                                          filename_info['end_time'].time())
        if self._end_time < self._start_time:
            self._end_time += timedelta(days=1)
        self.platform_id = filename_info['platform_id']
        if self.platform_id in ['NK', 'NL', 'NM', 'NN', 'NP', 'M1', 'M2',
                                'M3']:
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
            self.reader = self.reader_class(
                interpolate_coords=self.interpolate_coords,
                adjust_clock_drift=self.adjust_clock_drift,
                tle_dir=self.tle_dir,
                tle_name=self.tle_name,
                tle_thresh=self.tle_thresh)
            self.reader.read(self.filename)
        if np.all(self.reader.mask):
            raise ValueError('All data is masked out')

        if key.name in ['latitude', 'longitude']:
            # Lats/lons are buffered by the reader
            if key.name == 'latitude':
                _, data = self.reader.get_lonlat()
            else:
                data, _ = self.reader.get_lonlat()

            # If coordinate interpolation is disabled, only every eighth
            # pixel has a lat/lon coordinate
            xdim = 'x' if self.interpolate_coords else 'x_every_eighth'
            xcoords = None
        elif key.name in ANGLES:
            data = self._get_angle(key.name)
            xdim = 'x' if self.interpolate_coords else 'x_every_eighth'
            xcoords = None
        elif key.name == 'qual_flags':
            data = self.reader.get_qual_flags()
            xdim = 'num_flags'
            xcoords = ['Scan line number',
                       'Fatal error flag',
                       'Insufficient data for calibration',
                       'Insufficient data for calibration',
                       'Solar contamination of blackbody in channels 3',
                       'Solar contamination of blackbody in channels 4',
                       'Solar contamination of blackbody in channels 5']
        else:
            data = self._get_channel(key.name)
            xdim = 'x'
            xcoords = None

        # Update start/end time using the actual scanline timestamps
        times = self.reader.get_times()
        self._start_time = times[0].astype(datetime)
        self._end_time = times[-1].astype(datetime)

        # Select user-defined scanlines and/or strip invalid coordinates
        self.midnight_scanline = self.reader.get_midnight_scanline()
        self.missing_scanlines = self.reader.get_miss_lines().astype(int)
        if (self.start_line is not None or self.end_line is not None
                or self.strip_invalid_coords):
            data, times = self.slice(data=data, times=times)

        chunk_cols = data.shape[1]
        chunk_lines = int((CHUNK_SIZE ** 2) / chunk_cols)
        res = xr.DataArray(da.from_array(data, chunks=(chunk_lines, chunk_cols)),
                           dims=['y', xdim], attrs=info)
        if xcoords:
            res[xdim] = xcoords
        res.attrs['platform_name'] = self.reader.spacecraft_name
        res.attrs['orbit_number'] = self.filename_info['orbit_number']
        res.attrs['sensor'] = self.sensor
        res.attrs['orbital_parameters'] = {'tle': self.reader.get_tle_lines()}
        res.attrs['midnight_scanline'] = self.midnight_scanline
        res.attrs['missing_scanlines'] = self.missing_scanlines
        res['acq_time'] = ('y', times)
        res['acq_time'].attrs['long_name'] = 'Mean scanline acquisition time'
        return res

    def slice(self, data, times):
        """Select user-defined scanlines and/or strip invalid coordinates.

        Furthermore, update scanline timestamps and auxiliary information.

        Args:
            data: Data to be sliced
            times: Scanline timestamps
        Returns:
            Sliced data and timestamps
        """
        # Slice data, update midnight scanline & list of missing scanlines
        sliced, self.midnight_scanline, miss_lines = self._slice(data)
        self.missing_scanlines = miss_lines.astype(int)

        # Slice timestamps, update start/end time
        times, _, _ = self._slice(times)
        self._start_time = times[0].astype(datetime)
        self._end_time = times[-1].astype(datetime)

        return sliced, times

    def _slice(self, data):
        """Select user-defined scanlines and/or strip invalid coordinates.

        Returns:
            Sliced data, updated midnight scanline & list of missing scanlines
        """
        start_line = self.start_line if self.start_line is not None else 0
        end_line = self.end_line if self.end_line is not None else 0

        # Strip scanlines with invalid coordinates
        if self.strip_invalid_coords:
            first_valid_lat, last_valid_lat = self._strip_invalid_lat()
        else:
            first_valid_lat = last_valid_lat = None

        # Check and correct user-defined scanlines, if possible
        start_line, end_line = pygac.utils.check_user_scanlines(
            start_line=start_line,
            end_line=end_line,
            first_valid_lat=first_valid_lat,
            last_valid_lat=last_valid_lat,
            along_track=data.shape[0]
        )

        # Slice data, update missing lines and midnight scanline to new
        # scanline range
        sliced, miss_lines, midnight_scanline = pygac.utils.slice_channel(
            data,
            start_line=start_line,
            end_line=end_line,
            first_valid_lat=first_valid_lat,
            last_valid_lat=last_valid_lat,
            midnight_scanline=self.midnight_scanline,
            miss_lines=self.missing_scanlines,
            qual_flags=self._get_qual_flags()
        )

        return sliced, midnight_scanline, miss_lines

    def _get_channel(self, name):
        """Get channel by name and buffer results."""
        if self.channels is None:
            self.channels = self.reader.get_calibrated_channels()
        return self.channels[:, :, self.chn_dict[name.upper()]]

    def _get_qual_flags(self):
        """Get quality flags and buffer results."""
        if self.qual_flags is None:
            self.qual_flags = self.reader.get_qual_flags()
        return self.qual_flags

    def _get_angle(self, name):
        """Get angle by name and buffer results."""
        if self.angles is None:
            sat_azi, sat_zenith, sun_azi, sun_zenith, rel_azi = self.reader.get_angles()
            self.angles = {'sensor_zenith_angle': sat_zenith,
                           'sensor_azimuth_angle': sat_azi,
                           'solar_zenith_angle': sun_zenith,
                           'solar_azimuth_angle': sun_azi,
                           'sun_sensor_azimuth_difference_angle': rel_azi}
        return self.angles[name]

    def _strip_invalid_lat(self):
        """Strip scanlines with invalid coordinates in the beginning/end of the orbit.

        Returns:
            First and last scanline with valid latitudes.
        """
        if self.first_valid_lat is None:
            _, lats = self.reader.get_lonlat()
            start, end = pygac.utils.strip_invalid_lat(lats)
            self.first_valid_lat, self.last_valid_lat = start, end
        return self.first_valid_lat, self.last_valid_lat

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time
