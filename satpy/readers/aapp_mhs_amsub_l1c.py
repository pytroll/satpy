#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020, 2021, 2022 Pytroll developers

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


class MHS_AMSUB_AAPPL1CFile(AAPPL1BaseFileHandler):
    """Reader for AMSU-B/MHS L1C files created from the AAPP software."""

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize object information by reading the input file."""
        super().__init__(filename, filename_info, filetype_info)

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

    channel = channel.astype(np.float_)

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
