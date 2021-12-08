#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020, 2021 Pytroll developers

# Author(s):

#   Adam Dybbroe <Firstname.Lastname@smhi.se>

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

"""Reader for the AAPP AMSU-B/MHS level-1c data.

https://nwp-saf.eumetsat.int/site/download/documentation/aapp/NWPSAF-MF-UD-003_Formats_v8.0.pdf

"""

import logging
import numbers
from contextlib import suppress
from typing import NamedTuple

import dask.array as da
import numpy as np

from satpy import CHUNK_SIZE
from satpy.readers.aapp_l1b import AAPPL1BaseFileHandler, create_xarray

logger = logging.getLogger(__name__)


LINE_CHUNK = CHUNK_SIZE ** 2 // 90

MHS_AMSUB_CHANNEL_NAMES = ['1', '2', '3', '4', '5']
MHS_AMSUB_ANGLE_NAMES = ['sensor_zenith_angle', 'sensor_azimuth_angle',
                         'solar_zenith_angle', 'solar_azimuth_difference_angle']

MHS_AMSUB_PLATFORM_IDS2NAMES = {15: 'NOAA-15',
                                16: 'NOAA-16',
                                17: 'NOAA-17',
                                18: 'NOAA-18',
                                19: 'NOAA-19',
                                1: 'Metop-B',
                                2: 'Metop-A',
                                3: 'Metop-C',
                                4: 'Metop simulator'}

MHS_AMSUB_PLATFORMS = ['Metop-A', 'Metop-B', 'Metop-C', 'NOAA-18', 'NOAA-19']


class FrequencyDoubleSideBandBase(NamedTuple):
    """Base class for a frequency double side band.

    Frequency Double Side Band is supposed to describe the special type of bands
    commonly used in humidty sounding from Passive Microwave Sensors. When the
    absorption band being observed is symmetrical it is advantageous (giving
    better NeDT) to sense in a band both right and left of the central
    absorption frequency.

    This is needed because of this bug: https://bugs.python.org/issue41629

    """

    central: float
    side: float
    bandwidth: float
    unit: str = "GHz"


class FrequencyDoubleSideBand(FrequencyDoubleSideBandBase):
    """The frequency double side band class.

    The elements of the double-side-band type frequency band are the central
    frquency, the relative side band frequency (relative to the center - left
    and right) and their bandwidths, and optionally a unit (defaults to
    GHz). No clever unit conversion is done here, it's just used for checking
    that two ranges are comparable.

    Frequency Double Side Band is supposed to describe the special type of bands
    commonly used in humidty sounding from Passive Microwave Sensors. When the
    absorption band being observed is symmetrical it is advantageous (giving
    better NeDT) to sense in a band both right and left of the central
    absorption frequency.

    """

    def __eq__(self, other):
        """Return if two channel frequencies are equal.

        Args:
            other (tuple or scalar): (central frq, side band frq and band width frq) or scalar frq

        Return:
            True if other is a scalar and min <= other <= max, or if other is
            a tuple equal to self, False otherwise.

        """
        if other is None:
            return False
        if isinstance(other, numbers.Number):
            return other in self
        if isinstance(other, (tuple, list)) and len(other) == 3:
            return other in self
        return super().__eq__(other)

    def __ne__(self, other):
        """Return the opposite of `__eq__`."""
        return not self == other

    def __lt__(self, other):
        """Compare to another frequency."""
        if other is None:
            return False
        return super().__lt__(other)

    def __gt__(self, other):
        """Compare to another frequency."""
        if other is None:
            return True
        return super().__gt__(other)

    def __hash__(self):
        """Hash this tuple."""
        return tuple.__hash__(self)

    def __str__(self):
        """Format for print out."""
        return "{0.central} {0.unit} ({0.side}_{0.bandwidth} {0.unit})".format(self)

    def __contains__(self, other):
        """Check if this double-side-band 'contains' *other*."""
        if other is None:
            return False
        if isinstance(other, numbers.Number):
            if (self.central + self.side - self.bandwidth/2. <= other
                    <= self.central + self.side + self.bandwidth/2.):
                return True
            if (self.central - self.side - self.bandwidth/2. <= other
                    <= self.central - self.side + self.bandwidth/2.):
                return True
            return False

        if isinstance(other, (tuple, list)) and len(other) == 3:
            return ((self.central - self.side - self.bandwidth/2. <=
                     other[0] - other[1] - other[2]/2. and
                     self.central - self.side + self.bandwidth/2. >=
                     other[0] - other[1] + other[2]/2.) or
                    (self.central + self.side - self.bandwidth/2. <=
                     other[0] + other[1] - other[2]/2. and
                     self.central + self.side + self.bandwidth/2. >=
                     other[0] + other[1] + other[2]/2.))

        with suppress(AttributeError):
            if self.unit != other.unit:
                raise NotImplementedError("Can't compare frequency ranges with different units.")
            return ((self.central - self.side - self.bandwidth/2. <=
                     other.central - other.side - other.bandwidth/2. and
                     self.central - self.side + self.bandwidth/2. >=
                     other.central - other.side + other.bandwidth/2.) or
                    (self.central + self.side - self.bandwidth/2. <=
                     other.central + other.side - other.bandwidth/2. and
                     self.central + self.side + self.bandwidth/2. >=
                     other.central + other.side + other.bandwidth/2.))

        return False

    def distance(self, value):
        """Get the distance from value."""
        if self == value:
            try:
                left_side_dist = abs(value.central - value.side - (self.central - self.side))
                right_side_dist = abs(value.central + value.side - (self.central + self.side))
                return min(left_side_dist, right_side_dist)
            except AttributeError:
                if isinstance(value, (tuple, list)):
                    return abs((value[0] - value[1]) - (self.central - self.side))

                left_side_dist = abs(value - (self.central - self.side))
                right_side_dist = abs(value - (self.central + self.side))
                return min(left_side_dist, right_side_dist)
        else:
            return np.inf

    @classmethod
    def convert(cls, frq):
        """Convert `frq` to this type if possible."""
        if isinstance(frq, dict):
            return cls(**frq)
        return frq


class FrequencyRangeBase(NamedTuple):
    """Base class for frequency ranges.

    This is needed because of this bug: https://bugs.python.org/issue41629
    """

    central: float
    bandwidth: float
    unit: str = "GHz"


class FrequencyRange(FrequencyRangeBase):
    """The Frequency range class.

    The elements of the range are central and bandwidth values, and optionally
    a unit (defaults to GHz). No clever unit conversion is done here, it's just
    used for checking that two ranges are comparable.

    This type is used for passive microwave sensors.

    """

    def __eq__(self, other):
        """Return if two channel frequencies are equal.

        Args:
            other (tuple or scalar): (central frq, band width frq) or scalar frq

        Return:
            True if other is a scalar and min <= other <= max, or if other is
            a tuple equal to self, False otherwise.

        """
        if other is None:
            return False
        if isinstance(other, numbers.Number):
            return other in self
        if isinstance(other, (tuple, list)) and len(other) == 2:
            return self[:2] == other
        return super().__eq__(other)

    def __ne__(self, other):
        """Return the opposite of `__eq__`."""
        return not self == other

    def __lt__(self, other):
        """Compare to another frequency."""
        if other is None:
            return False
        return super().__lt__(other)

    def __gt__(self, other):
        """Compare to another frequency."""
        if other is None:
            return True
        return super().__gt__(other)

    def __hash__(self):
        """Hash this tuple."""
        return tuple.__hash__(self)

    def __str__(self):
        """Format for print out."""
        return "{0.central} {0.unit} ({0.bandwidth} {0.unit})".format(self)

    def __contains__(self, other):
        """Check if this range contains *other*."""
        if other is None:
            return False
        if isinstance(other, numbers.Number):
            return self.central - self.bandwidth/2. <= other <= self.central + self.bandwidth/2.

        with suppress(AttributeError):
            if self.unit != other.unit:
                raise NotImplementedError("Can't compare frequency ranges with different units.")
            return (self.central - self.bandwidth/2. <= other.central - other.bandwidth/2. and
                    self.central + self.bandwidth/2. >= other.central + other.bandwidth/2.)
        return False

    def distance(self, value):
        """Get the distance from value."""
        if self == value:
            try:
                return abs(value.central - self.central)
            except AttributeError:
                if isinstance(value, (tuple, list)):
                    return abs(value[0] - self.central)
                return abs(value - self.central)
        else:
            return np.inf

    @classmethod
    def convert(cls, frq):
        """Convert `frq` to this type if possible."""
        if isinstance(frq, dict):
            return cls(**frq)
        return frq


class MHS_AMSUB_AAPPL1CFile(AAPPL1BaseFileHandler):
    """Reader for AMSU-B/MHS L1C files created from the AAPP software."""

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize object information by reading the input file."""
        super(MHS_AMSUB_AAPPL1CFile, self).__init__(filename, filename_info,
                                                    filetype_info)

        self.channels = {i: None for i in MHS_AMSUB_CHANNEL_NAMES}
        self.units = {i: 'brightness_temperature' for i in MHS_AMSUB_CHANNEL_NAMES}

        self._channel_names = MHS_AMSUB_CHANNEL_NAMES
        self._angle_names = MHS_AMSUB_ANGLE_NAMES

        self._set_filedata_layout()
        self.read()

        self._get_platform_name(MHS_AMSUB_PLATFORM_IDS2NAMES)
        self._get_sensorname()

    def _set_filedata_layout(self):
        """Set the file data type/layout."""
        self._header_offset = HEADER_LENGTH
        self._scan_type = _SCANTYPE
        self._header_type = _HEADERTYPE

    def _get_sensorname(self):
        """Get the sensor name from the header."""
        if self._header['instrument'][0] == 11:
            self.sensor = 'amsub'
        elif self._header['instrument'][0] == 12:
            self.sensor = 'mhs'
        else:
            raise IOError("Sensor neither MHS nor AMSU-B!")

    def get_angles(self, angle_id):
        """Get sun-satellite viewing angles."""
        satz = self._data["angles"][:, :, 0] * 1e-2
        sata = self._data["angles"][:, :, 1] * 1e-2

        sunz = self._data["angles"][:, :, 2] * 1e-2
        suna = self._data["angles"][:, :, 3] * 1e-2

        name_to_variable = dict(zip(MHS_AMSUB_ANGLE_NAMES, (satz, sata, sunz, suna)))
        return create_xarray(name_to_variable[angle_id])

    def navigate(self, coordinate_id):
        """Get the longitudes and latitudes of the scene."""
        lons, lats = self._get_coordinates_in_degrees()
        if coordinate_id == 'longitude':
            return create_xarray(lons)
        if coordinate_id == 'latitude':
            return create_xarray(lats)

        raise KeyError("Coordinate {} unknown.".format(coordinate_id))

    def _get_coordinates_in_degrees(self):
        lons = self._data["latlon"][:, :, 1] * 1e-4
        lats = self._data["latlon"][:, :, 0] * 1e-4
        return lons, lats

    def _calibrate_active_channel_data(self, key):
        """Calibrate active channel data only."""
        return self.calibrate(key)

    def calibrate(self, dataset_id):
        """Calibrate the data."""
        units = {'brightness_temperature': 'K'}

        mask = True
        idx = ['1', '2', '3', '4', '5'].index(dataset_id['name'])

        ds = create_xarray(
            _calibrate(self._data, idx,
                       dataset_id['calibration'],
                       mask=mask))

        ds.attrs['units'] = units[dataset_id['calibration']]
        ds.attrs.update(dataset_id._asdict())
        return ds


def _calibrate(data,
               chn,
               calib_type,
               mask=True):
    """Calibrate channel data.

    *calib_type* in brightness_temperature.

    """
    if calib_type not in ['brightness_temperature']:
        raise ValueError('Calibration ' + calib_type + ' unknown!')

    channel = da.from_array(data["btemps"][:, :, chn] / 100., chunks=(LINE_CHUNK, 90))
    mask &= channel != 0

    if calib_type == 'counts':
        return channel

    channel = channel.astype(np.float)

    return da.where(mask, channel, np.nan)


HEADER_LENGTH = 1152*4

_HEADERTYPE = np.dtype([("siteid", "S3"),
                        ("cfill_1", "S1"),
                        ("l1bsite", "S3"),
                        ("cfill_2", "S1"),
                        ("versnb", "<i4"),
                        ("versyr", "<i4"),
                        ("versdy", "<i4"),
                        ("hdrcnt", "<i4"),
                        ("satid", "<i4"),
                        ("instrument", "<i4"),
                        ("satht", "<i4"),
                        ("period", "<i4"),
                        ("startorbit", "<i4"),
                        ("startdatayr", "<i4"),
                        ("startdatady", "<i4"),
                        ("startdatatime", "<i4"),
                        ("endorbit", "<i4"),
                        ("enddatayr", "<i4"),
                        ("enddatady", "<i4"),
                        ("enddatatime", "<i4"),
                        ("scnlin", "<i4"),
                        ("misscnlin", "<i4"),
                        ("vnantennacorr", "<i4"),
                        ("spare", "<i4"),
                        ("tempradcnv", "<i4", (3, 5)),
                        ("wmosatid", "<i4"),
                        ("filler", "<i4", (1114,)),
                        ])

_SCANTYPE = np.dtype([("scnlin", "<i4"),
                      ("scnlinyr", "<i4"),
                      ("scnlindy", "<i4"),
                      ("scnlintime", "<i4"),
                      ("qualind", "<i4"),
                      ("scnlinqual", "<i4"),
                      ("chanqual", "<i4", (5, )),
                      ("instrtemp", "<i4"),
                      ("spare1", "<i4", (2, )),
                      # Navigation
                      ("latlon", "<i4", (90, 2)),  # lat/lon in degrees for Bnfovs:
                      # first : 10^4 x (latitude)
                      # second : 10^4 x (longitude)
                      ("angles", "<i4", (90, 4)),  # scan angles for Bnfovs:
                      # first: 10^2 x (local zenith angle)
                      # second: 10^2 x (local azimuth angle)
                      # third: 10^2 x (solar zenith angle)
                      # fourth: 10^2 x (solar azimuth angle)
                      ("scalti", "<i4"),  # sat altitude above reference ellipsoid, km*10
                      ("spare2", "<i4", (2, )),
                      # Calibration
                      ("btemps", "<i4", (90, 5)),  # BT data for Bnfovs 10^2 x scene Tb (K), channels 1-5
                      ("dataqual", "<i4", (90, )),
                      ("filler", "<i4", (55, ))
                      ])
