#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020 Adam.Dybbroe

# Author(s):

#   Adam.Dybbroe <a000680@c21856.ad.smhi.se>

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

import dask.array as da
import numpy as np
from datetime import datetime, timedelta

from satpy import CHUNK_SIZE
from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.aapp_l1b import create_xarray
import logging

logger = logging.getLogger(__name__)


LINE_CHUNK = CHUNK_SIZE ** 2 // 90

MHS_CHANNEL_NAMES = ['1', '2', '3', '4', '5']
MHS_CHANNEL_NAMES_SET = ('1', '2', '3', '4', '5')

PLATFORM_NAMES = {15: 'NOAA-15',
                  16: 'NOAA-16',
                  17: 'NOAA-17',
                  18: 'NOAA-18',
                  19: 'NOAA-19',
                  1: 'Metop-B',
                  2: 'Metop-A',
                  3: 'Metop-C',
                  4: 'Metop simulator'}

MHS_PLATFORMS = ['Metop-A', 'Metop-B', 'Metop-c', 'NOAA-18', 'NOAA-19']

ANGLES = ['sensor_zenith_angle', 'sensor_azimuth_angle',
          'solar_zenith_angle', 'solar_azimuth_difference_angle']


class FrequencyStripes(tuple):
    """A tuple for frequency stripes and band identificationswavelength ranges.

    The elements of the range are min, central and max values, and optionally a unit
    (defaults to µm). No clever unit conversion is done here, it's just used for checking
    that two ranges are comparable.
    """

    def __eq__(self, other):
        """Return if two wavelengths are equal.

        Args:
            other (tuple or scalar): (min wl, nominal wl, max wl) or scalar wl

        Return:
            True if other is a scalar and min <= other <= max, or if other is
            a tuple equal to self, False otherwise.

        """
        if other is None:
            return False
        elif isinstance(other, numbers.Number):
            return other in self
        elif isinstance(other, (tuple, list)) and len(other) == 3:
            return self[:3] == other
        return super().__eq__(other)

    def __ne__(self, other):
        """Return the opposite of `__eq__`."""
        return not self == other

    def __lt__(self, other):
        """Compare to another wavelength."""
        if other is None:
            return False
        return super().__lt__(other)

    def __gt__(self, other):
        """Compare to another wavelength."""
        if other is None:
            return True
        return super().__gt__(other)

    def __hash__(self):
        """Hash this tuple."""
        return tuple.__hash__(self)

    def __str__(self):
        """Format for print out."""
        return "{0.central} {0.unit} ({0.min}-{0.max} {0.unit})".format(self)

    def __contains__(self, other):
        """Check if this range contains *other*."""
        if other is None:
            return False
        elif isinstance(other, numbers.Number):
            return self.min <= other <= self.max
        with suppress(AttributeError):
            if self.unit != other.unit:
                raise NotImplementedError("Can't compare wavelength ranges with different units.")
            return self.min <= other.min and self.max >= other.max
        return False

    def distance(self, value):
        """Get the distance from value."""
        if self == value:
            try:
                return abs(value.central - self.central)
            except AttributeError:
                if isinstance(value, (tuple, list)):
                    return abs(value[1] - self.central)
                return abs(value - self.central)
        else:
            return np.inf

    @classmethod
    def convert(cls, wl):
        """Convert `wl` to this type if possible."""
        if isinstance(wl, (dict)):
            return cls(*wl)
        return wl


class MHSAAPPL1CFile(BaseFileHandler):
    """Reader for AMSU-B/MHS L1C files created from the AAPP software."""

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize object information by reading the input file."""
        super(MHSAAPPL1CFile, self).__init__(filename, filename_info,
                                             filetype_info)
        self.channels = {i: None for i in MHS_CHANNEL_NAMES_SET}
        self.units = {i: 'brightness_temperature' for i in MHS_CHANNEL_NAMES_SET}

        self._data = None
        self._header = None

        self._shape = None
        self.area = None

        self.read()

        self.platform_name = PLATFORM_NAMES.get(self._header['satid'][0], None)
        if self.platform_name is None:
            raise ValueError("Unsupported platform ID: %d" % self._header['satid'])

        if self.platform_name in MHS_PLATFORMS:
            self.sensor = 'mhs'
        else:
            self.sensor = 'amsub'

    @property
    def start_time(self):
        """Get the time of the first observation."""
        return datetime(self._data['scnlinyr'][0], 1, 1) + timedelta(
            days=int(self._data['scnlindy'][0]) - 1,
            milliseconds=int(self._data['scnlintime'][0]))

    @property
    def end_time(self):
        """Get the time of the final observation."""
        return datetime(self._data['scnlinyr'][-1], 1, 1) + timedelta(
            days=int(self._data['scnlindy'][-1]) - 1,
            milliseconds=int(self._data['scnlintime'][-1]))

    def get_dataset(self, key, info):
        """Get a dataset from the file."""
        if key['name'] in MHS_CHANNEL_NAMES:
            dataset = self.calibrate(key)

        elif key['name'] in ['longitude', 'latitude']:
            dataset = self.navigate(key['name'])
            dataset.attrs = info
        elif key['name'] in ANGLES:
            dataset = self.get_angles(key['name'])
        else:
            raise ValueError("Not a supported dataset: %s", key['name'])

        self._update_dataset_attributes(dataset, key, info)

        if not self._shape:
            self._shape = dataset.shape

        return dataset

    def _update_dataset_attributes(self, dataset, key, info):
        dataset.attrs.update({'platform_name': self.platform_name,
                              'sensor': self.sensor})
        dataset.attrs.update(key.to_dict())
        for meta_key in ('standard_name', 'units'):
            if meta_key in info:
                dataset.attrs.setdefault(meta_key, info[meta_key])

    def read(self):
        """Read the data."""
        tic = datetime.now()
        header = np.memmap(self.filename, dtype=_HEADERTYPE, mode="r", shape=(1, ))
        data = np.memmap(self.filename, dtype=_SCANTYPE, offset=HEADER_LENGTH, mode="r")
        logger.debug("Reading time %s", str(datetime.now() - tic))

        self._header = header
        self._data = data

    def get_angles(self, angle_id):
        """Get sun-satellite viewing angles."""
        satz = self._data["angles"][:, :, 0] * 1e-2
        sata = self._data["angles"][:, :, 1] * 1e-2

        sunz = self._data["angles"][:, :, 2] * 1e-2
        suna = self._data["angles"][:, :, 3] * 1e-2

        name_to_variable = dict(zip(ANGLES, (satz, sata, sunz, suna)))
        return create_xarray(name_to_variable[angle_id])

    def navigate(self, coordinate_id):
        """Get the longitudes and latitudes of the scene."""
        lons, lats = self._get_coordinates_in_degrees()
        if coordinate_id == 'longitude':
            return create_xarray(lons)
        elif coordinate_id == 'latitude':
            return create_xarray(lats)
        else:
            raise KeyError("Coordinate {} unknown.".format(coordinate_id))

    def _get_coordinates_in_degrees(self):
        lons = self._data["latlon"][:, :, 1] * 1e-4
        lats = self._data["latlon"][:, :, 0] * 1e-4
        return lons, lats

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
    mask &= channel != 0  # What does this do? FIXME!

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
                      ("btemps", "<i4", (90, 5)),  # BT data for Bnfovs 10^2 x scene brightness temperature (K), channels 1-5
                      ("dataqual", "<i4", (90, )),
                      ("filler", "<i4", (55, ))
                      ])
