#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009-2021 Satpy developers
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
"""Reading and calibrating hrpt avhrr data.

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

import dask.array as da
import numpy as np
import xarray as xr
from geotiepoints import SatelliteInterpolator
from pyorbital.geoloc import compute_pixels, get_lonlatalt
from pyorbital.geoloc_instrument_definitions import avhrr
from pyorbital.orbital import Orbital

from satpy._compat import cached_property
from satpy.readers.aapp_l1b import get_avhrr_lac_chunks
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
    """Return the time object from the timecodes."""
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
    """Return the bit array."""
    return (array & 2**(9 - bit + 1)).astype(bool)


spacecrafts = {7: "NOAA 15", 3: "NOAA 16", 13: "NOAA 18", 15: "NOAA 19"}


def geo_interpolate(lons32km, lats32km):
    """Interpolate geo data."""
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


def _get_channel_index(key):
    """Get the avhrr channel index."""
    avhrr_channel_index = {'1': 0,
                           '2': 1,
                           '3a': 2,
                           '3b': 2,
                           '4': 3,
                           '5': 4}
    index = avhrr_channel_index[key['name']]
    return index


class HRPTFile(BaseFileHandler):
    """Reader for HRPT Minor Frame, 10 bits data expanded to 16 bits."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init the file handler."""
        super(HRPTFile, self).__init__(filename, filename_info, filetype_info)
        self.channels = {i: None for i in AVHRR_CHANNEL_NAMES}
        self.units = {i: 'counts' for i in AVHRR_CHANNEL_NAMES}

        self.year = filename_info.get('start_time', datetime.utcnow()).year

    @cached_property
    def times(self):
        """Get the timestamps for each line."""
        return time_seconds(self._data["timecode"], self.year)

    @cached_property
    def _chunks(self):
        """Get the best chunks for this data."""
        return get_avhrr_lac_chunks((self._data.shape[0], 2048), float)

    @cached_property
    def _data(self):
        """Get the data."""
        return self.read()

    def read(self):
        """Read the file."""
        with open(self.filename, "rb") as fp_:
            data = np.memmap(fp_, dtype=dtype, mode="r")
        if np.all(np.median(data['frame_sync'], axis=0) > 1024):
            data = self._data.newbyteorder()
        return data

    @cached_property
    def platform_name(self):
        """Get the platform name."""
        return spacecrafts[np.median((self._data["id"]["id"] >> 3) & 15)]

    def get_dataset(self, key, info):
        """Get the dataset."""
        attrs = info.copy()
        attrs['platform_name'] = self.platform_name

        if key['name'] in ['latitude', 'longitude']:
            data = self._get_navigation_data(key)
        else:
            data = self._get_channel_data(key)

        result = xr.DataArray(data, dims=['y', 'x'], attrs=attrs)
        mask = self._get_ch3_mask_or_true(key)
        return result.where(mask)

    def _get_channel_data(self, key):
        """Get channel data."""
        data = da.from_array(self._data["image_data"][:, :, _get_channel_index(key)], chunks=self._chunks)
        if key['calibration'] != 'counts':
            if key['name'] in ['1', '2', '3a']:
                data = self.calibrate_solar_channel(data, key)

            if key['name'] in ['3b', '4', '5']:
                data = self.calibrate_thermal_channel(data, key)
        return data

    def _get_navigation_data(self, key):
        """Get navigation data."""
        lons, lats = self.lons_lats
        if key['name'] == 'latitude':
            data = da.from_array(lats, chunks=self._chunks)
        else:
            data = da.from_array(lons, chunks=self._chunks)
        return data

    def _get_ch3_mask_or_true(self, key):
        mask = True
        if key['name'] == '3a':
            mask = np.tile(np.logical_not(self._is3b), (2048, 1)).T
        elif key['name'] == '3b':
            mask = np.tile(self._is3b, (2048, 1)).T
        return mask

    @cached_property
    def _is3b(self):
        return bfield(self._data["id"]["id"], 10) == 0

    def calibrate_thermal_channel(self, data, key):
        """Calibrate a thermal channel."""
        from pygac.calibration import calibrate_thermal
        line_numbers = (
            np.round((self.times - self.times[-1]) /
                     np.timedelta64(166666667, 'ns'))).astype(int)
        line_numbers -= line_numbers[0]
        prt, ict, space = self.telemetry
        index = _get_channel_index(key)
        data = calibrate_thermal(data, prt, ict[:, index - 2],
                                 space[:, index], line_numbers,
                                 index + 1, self.calibrator)
        return data

    def calibrate_solar_channel(self, data, key):
        """Calibrate a solar channel."""
        from pygac.calibration import calibrate_solar
        julian_days = ((np.datetime64(self.start_time)
                        - np.datetime64(str(self.year) + '-01-01T00:00:00Z'))
                       / np.timedelta64(1, 'D'))
        data = calibrate_solar(data, _get_channel_index(key), self.year, julian_days,
                               self.calibrator)
        return data

    @cached_property
    def calibrator(self):
        """Create a calibrator for the data."""
        from pygac.calibration import Calibrator
        pg_spacecraft = ''.join(self.platform_name.split()).lower()
        return Calibrator(pg_spacecraft)

    @cached_property
    def telemetry(self):
        """Get the telemetry."""
        # This isn't converted to dask arrays as it does not work with pygac
        prt = np.mean(self._data["telemetry"]['PRT'], axis=1)
        ict = np.mean(self._data['back_scan'], axis=1)
        space = np.mean(self._data['space_data'][:, :], axis=1)

        return prt, ict, space

    @cached_property
    def lons_lats(self):
        """Get the lons and lats."""
        scanline_nb = len(self.times)
        scan_points = np.arange(0, 2048, 32)
        lons, lats = self._get_avhrr_tiepoints(scan_points, scanline_nb)

        lons, lats = geo_interpolate(
            lons.reshape((scanline_nb, -1)), lats.reshape((scanline_nb, -1)))
        return lons, lats

    def _get_avhrr_tiepoints(self, scan_points, scanline_nb):
        sgeom = avhrr(scanline_nb, scan_points, apply_offset=False)
        # no attitude error
        rpy = [0, 0, 0]
        s_times = sgeom.times(self.times[:, np.newaxis])
        orb = Orbital(self.platform_name)
        pixels_pos = compute_pixels(orb, sgeom, s_times, rpy)
        lons, lats, alts = get_lonlatalt(pixels_pos, s_times)
        return lons, lats

    @property
    def start_time(self):
        """Get the start time."""
        return time_seconds(self._data["timecode"][0, np.newaxis, :],
                            self.year).astype(datetime)[0]

    @property
    def end_time(self):
        """Get the end time."""
        return time_seconds(self._data["timecode"][-1, np.newaxis, :],
                            self.year).astype(datetime)[0]
