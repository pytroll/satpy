#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009-2016.

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

# This file is part of satpy.

# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""
Reading and calibrating hrpt avhrr data.
Todo:
- AMSU
- Compare output with AAPP

Reading:
http://www.ncdc.noaa.gov/oa/pod-guide/ncdc/docs/klm/html/c4/sec4-1.htm#t413-1

Calibration:
http://www.ncdc.noaa.gov/oa/pod-guide/ncdc/docs/klm/html/c7/sec7-1.htm

"""

import logging
from datetime import datetime

import numpy as np

from pygac.gac_calibration import calibrate_solar, calibrate_thermal
from satpy.dataset import Dataset
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)

AVHRR_CHANNEL_NAMES = ("1", "2", "3a", "3b", "4", "5")

dtype = np.dtype([('frame_sync', '>u2', (6, )),
                  ('id', [('id', '>u2'),
                          ('spare', '>u2')]),
                  ('timecode', '>u2', (4, )),
                  ('telemetry', [("ramp_calibration", '>u2', (5, )),
                                 ("PRT", '>u2', (3, )),
                                 ("ch3_patch_temp", '>u2'),
                                 ("spare", '>u2'), ]),
                  ('back_scan', '>u2', (10, 3)),
                  ('space_data', '>u2', (10, 5)),
                  ('sync', '>u2'),
                  ('TIP_data', '>u2', (520, )),
                  ('spare', '>u2', (127, )),
                  ('image_data', '>u2', (2048, 5)),
                  ('aux_sync', '>u2', (100, ))])


def time_seconds(tc_array, year):
    """Return the time object from the timecodes
    """
    tc_array = np.array(tc_array, copy=True)
    word = tc_array[:, 0]
    day = word >> 1
    word = tc_array[:, 1].astype(np.uint64)
    msecs = ((127) & word) * 1024
    word = tc_array[:, 2]
    msecs += word & 1023
    msecs *= 1024
    word = tc_array[:, 3]
    msecs += word & 1023
    return (np.datetime64(
        str(year) + '-01-01T00:00:00Z', 's') +
        msecs[:].astype('timedelta64[ms]') +
        (day - 1)[:].astype('timedelta64[D]'))


def bfield(array, bit):
    """return the bit array.
    """
    return (array & 2**(9 - bit + 1)).astype(np.bool)


spacecrafts = {7: "NOAA 15", 3: "NOAA 16", 13: "NOAA 18", 15: "NOAA 19"}


def geo_interpolate(lons32km, lats32km):
    from geotiepoints import SatelliteInterpolator
    cols32km = np.arange(0, 2048, 32)
    cols1km = np.arange(2048)
    lines = lons32km.shape[0]
    rows32km = np.arange(lines)
    rows1km = np.arange(lines)

    along_track_order = 1
    cross_track_order = 3

    satint = SatelliteInterpolator(
        (lons32km, lats32km), (rows32km, cols32km), (rows1km, cols1km),
        along_track_order, cross_track_order)
    lons, lats = satint.interpolate()
    return lons, lats


class HRPTFile(BaseFileHandler):
    """Reader for HRPT Minor Frame, 10 bits data expanded to 16 bits.
    """

    def __init__(self, filename, filename_info, filetype_info):
        super(HRPTFile, self).__init__(filename, filename_info, filetype_info)
        self.channels = {i: None for i in AVHRR_CHANNEL_NAMES}
        self.units = {i: 'counts' for i in AVHRR_CHANNEL_NAMES}

        self._data = None
        self._is3b = None
        self.lons = None
        self.lats = None
        self.area = None
        self.platform_name = None
        self.year = filename_info.get('start_time', datetime.utcnow()).year
        self.times = None
        self.prt = None
        self.ict = None
        self.space = None
        self.read()

    def read(self):

        with open(self.filename, "rb") as fp_:
            self._data = np.memmap(fp_, dtype=dtype, mode="r")
        if np.all(self._data['frame_sync'][0] > 1024):
            self._data = self._data.newbyteorder()
        self.platform_name = spacecrafts[
            (self._data["id"]["id"][0] >> 3) & 15]

    def get_dataset(self, key, info):
        if self._data is None:
            self.read()

        if key.name in ['latitude', 'longitude']:
            lons, lats = self.get_lonlats()
            if key.name == 'latitude':
                return Dataset(lats, id=key)
            else:
                return Dataset(lons, id=key)

        avhrr_channel_index = {'1': 0,
                               '2': 1,
                               '3a': 2,
                               '3b': 2,
                               '4': 3,
                               '5': 4}
        index = avhrr_channel_index[key.name]
        mask = False
        if key.name in ['3a', '3b'] and self._is3b is None:
            ch3a = bfield(self._data["id"]["id"], 10)
            self._is3b = np.logical_not(ch3a)

        if key.name == '3a':
            mask = np.tile(self._is3b, (1, 2048))
        elif key.name == '3b':
            mask = np.tile(np.logical_not(self._is3b), (1, 2048))

        data = self._data["image_data"][:, :, index]
        if key.calibration == 'counts':
            return Dataset(data,
                           mask=mask,
                           area=self.get_lonlats(),
                           units='1')

        pg_spacecraft = ''.join(self.platform_name.split()).lower()

        jdays = (np.datetime64(self.start_time) - np.datetime64(str(
            self.year) + '-01-01T00:00:00Z')) / np.timedelta64(1, 'D')
        if index < 2 or key.name == '3a':
            data = calibrate_solar(data, index, self.year, jdays,
                                   pg_spacecraft)
            units = '%'

        if index > 2 or key.name == '3b':
            if self.times is None:
                self.times = time_seconds(self._data["timecode"], self.year)
            line_numbers = (
                np.round((self.times - self.times[-1]) /
                         np.timedelta64(166666667, 'ns'))).astype(np.int)
            line_numbers -= line_numbers[0]
            if self.prt is None:
                self.prt, self.ict, self.space = self.get_telemetry()
            chan = index + 1
            data = calibrate_thermal(data, self.prt, self.ict[:, chan - 3],
                                     self.space[:, chan - 3], line_numbers,
                                     chan, pg_spacecraft)
            units = 'K'
        # TODO: check if entirely masked before returning
        return Dataset(data, mask=mask, units=units)

    def get_telemetry(self):
        prt = np.mean(self._data["telemetry"]['PRT'], axis=1)

        ict = np.empty((len(self._data), 3))
        for i in range(3):
            ict[:, i] = np.mean(self._data['back_scan'][:, :, i], axis=1)

        space = np.empty((len(self._data), 3))
        for i in range(3):
            space[:, i] = np.mean(self._data['space_data'][
                                  :, :, i + 2], axis=1)

        return prt, ict, space

    def get_lonlats(self):
        if self.lons is not None and self.lats is not None:
            return self.lons, self.lats
        from pyorbital.orbital import Orbital
        from pyorbital.geoloc import compute_pixels, get_lonlatalt
        from pyorbital.geoloc_instrument_definitions import avhrr

        if self.times is None:
            self.times = time_seconds(self._data["timecode"], self.year)
        scanline_nb = len(self.times)
        scan_points = np.arange(0, 2048, 32)
        # scan_points = np.arange(2048)

        sgeom = avhrr(scanline_nb, scan_points, apply_offset=False)
        # no attitude error
        rpy = [0, 0, 0]
        s_times = sgeom.times(
            self.times[:, np.newaxis]).ravel()
        # s_times = (np.tile(sgeom._times[0, :], (scanline_nb, 1)).astype(
        #    'timedelta64[s]') + self.times[:, np.newaxis]).ravel()

        orb = Orbital(self.platform_name)

        pixels_pos = compute_pixels(orb, sgeom, s_times, rpy)
        lons, lats, alts = get_lonlatalt(pixels_pos, s_times)
        self.lons, self.lats = geo_interpolate(
            lons.reshape((scanline_nb, -1)), lats.reshape((scanline_nb, -1)))

        return self.lons, self.lats

    @property
    def start_time(self):
        return time_seconds(self._data["timecode"][0, np.newaxis, :],
                            self.year).astype(datetime)[0]

    @property
    def end_time(self):
        return time_seconds(self._data["timecode"][-1, np.newaxis, :],
                            self.year).astype(datetime)[0]
