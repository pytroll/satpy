#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010-2017 Satpy developers
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
"""HRIT format reader for JMA data.

Introduction
------------
The JMA HRIT format is described in the `JMA HRIT - Mission Specific
Implementation`_. There are three readers for this format in Satpy:

- ``jami_hrit``: For data from the `JAMI` instrument on MTSAT-1R
- ``mtsat2-imager_hrit``: For data from the `Imager` instrument on MTSAT-2
- ``ahi_hrit``: For data from the `AHI` instrument on Himawari-8/9

Although the data format is identical, the instruments have different
characteristics, which is why there is a dedicated reader for each of them.
Sample data is available here:

- `JAMI/Imager sample data`_
- `AHI sample data`_


Example
-------
Here is an example how to read Himwari-8 HRIT data with Satpy:

.. code-block:: python

    from satpy import Scene
    import glob

    filenames = glob.glob('data/IMG_DK01B14_2018011109*')
    scn = Scene(filenames=filenames, reader='ahi_hrit')
    scn.load(['B14'])
    print(scn['B14'])


Output:

.. code-block:: none

    <xarray.DataArray (y: 5500, x: 5500)>
    dask.array<concatenate, shape=(5500, 5500), dtype=float64, chunksize=(550, 4096), ...
    Coordinates:
        acq_time  (y) datetime64[ns] 2018-01-11T09:00:20.995200 ... 2018-01-11T09:09:40.348800
        crs       object +proj=geos +lon_0=140.7 +h=35785831 +x_0=0 +y_0=0 +a=6378169 ...
      * y         (y) float64 5.5e+06 5.498e+06 5.496e+06 ... -5.496e+06 -5.498e+06
      * x         (x) float64 -5.498e+06 -5.496e+06 -5.494e+06 ... 5.498e+06 5.5e+06
    Attributes:
        orbital_parameters:   {'projection_longitude': 140.7, 'projection_latitud...
        standard_name:        toa_brightness_temperature
        level:                None
        wavelength:           (11.0, 11.2, 11.4)
        units:                K
        calibration:          brightness_temperature
        file_type:            ['hrit_b14_seg', 'hrit_b14_fd']
        modifiers:            ()
        polarization:         None
        sensor:               ahi
        name:                 B14
        platform_name:        Himawari-8
        resolution:           4000
        start_time:           2018-01-11 09:00:20.995200
        end_time:             2018-01-11 09:09:40.348800
        area:                 Area ID: FLDK, Description: Full Disk, Projection I...
        ancillary_variables:  []

JMA HRIT data contain the scanline acquisition time for only a subset of scanlines. Timestamps of
the remaining scanlines are computed using linear interpolation. This is what you'll find in the
``acq_time`` coordinate of the dataset.

Compression
-----------

Gzip-compressed MTSAT files can be decompressed on the fly using
:class:`~satpy.readers.FSFile`:

.. code-block:: python

    import fsspec
    from satpy import Scene
    from satpy.readers import FSFile

    filename = "/data/HRIT_MTSAT1_20090101_0630_DK01IR1.gz"
    open_file = fsspec.open(filename, compression="gzip")
    fs_file = FSFile(open_file)
    scn = Scene([fs_file], reader="jami_hrit")
    scn.load(["IR1"])


.. _JMA HRIT - Mission Specific Implementation: http://www.jma.go.jp/jma/jma-eng/satellite/introduction/4_2HRIT.pdf
.. _JAMI/Imager sample data: https://www.data.jma.go.jp/mscweb/en/operation/hrit_sample.html
.. _AHI sample data: https://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/sample_hrit.html
"""

import logging
from datetime import datetime

import numpy as np
import xarray as xr

from satpy.readers._geos_area import get_area_definition, get_area_extent
from satpy.readers.hrit_base import (
    HRITFileHandler,
    ancillary_text,
    annotation_header,
    base_hdr_map,
    image_data_function,
)
from satpy.readers.utils import get_geostationary_mask

logger = logging.getLogger('hrit_jma')


# JMA implementation:
key_header = np.dtype([('key_number', 'u4')])

segment_identification = np.dtype([('image_segm_seq_no', '>u1'),
                                   ('total_no_image_segm', '>u1'),
                                   ('line_no_image_segm', '>u2')])

encryption_key_message = np.dtype([('station_number', '>u2')])

image_compensation_information = np.dtype([('compensation', '|S1')])

image_observation_time = np.dtype([('times', '|S1')])

image_quality_information = np.dtype([('quality', '|S1')])


jma_variable_length_headers: dict = {}

jma_text_headers = {image_data_function: 'image_data_function',
                    annotation_header: 'annotation_header',
                    ancillary_text: 'ancillary_text',
                    image_compensation_information: 'image_compensation_information',
                    image_observation_time: 'image_observation_time',
                    image_quality_information: 'image_quality_information'}

jma_hdr_map = base_hdr_map.copy()
jma_hdr_map.update({7: key_header,
                    128: segment_identification,
                    129: encryption_key_message,
                    130: image_compensation_information,
                    131: image_observation_time,
                    132: image_quality_information
                    })


cuc_time = np.dtype([('coarse', 'u1', (4, )),
                     ('fine', 'u1', (3, ))])

time_cds_expanded = np.dtype([('days', '>u2'),
                              ('milliseconds', '>u4'),
                              ('microseconds', '>u2'),
                              ('nanoseconds', '>u2')])

FULL_DISK = 1
NORTH_HEMIS = 2
SOUTH_HEMIS = 3
UNKNOWN_AREA = -1
AREA_NAMES = {FULL_DISK: {'short': 'FLDK', 'long': 'Full Disk'},
              NORTH_HEMIS: {'short': 'NH', 'long': 'Northern Hemisphere'},
              SOUTH_HEMIS: {'short': 'SH', 'long': 'Southern Hemisphere'},
              UNKNOWN_AREA: {'short': 'UNKNOWN', 'long': 'Unknown Area'}}

MTSAT1R = 'MTSAT-1R'
MTSAT2 = 'MTSAT-2'
HIMAWARI8 = 'Himawari-8'
UNKNOWN_PLATFORM = 'Unknown Platform'
PLATFORMS = {
    'GEOS(140.00)': MTSAT1R,
    'GEOS(140.25)': MTSAT1R,
    'GEOS(140.70)': HIMAWARI8,
    'GEOS(145.00)': MTSAT2,
}
SENSORS = {
    MTSAT1R: 'jami',
    MTSAT2: 'mtsat2_imager',
    HIMAWARI8: 'ahi'
}


def mjd2datetime64(mjd):
    """Convert Modified Julian Day (MJD) to datetime64."""
    epoch = np.datetime64('1858-11-17 00:00')
    day2usec = 24 * 3600 * 1E6
    mjd_usec = (mjd * day2usec).astype(np.int64).astype('timedelta64[us]')
    return epoch + mjd_usec


class HRITJMAFileHandler(HRITFileHandler):
    """JMA HRIT format reader.

    By default, the reader uses the start time parsed from the filename. To use exact time, computed
    from the metadata, the user can define a keyword argument::

        scene = Scene(filenames=filenames,
                      reader='ahi_hrit',
                      reader_kwargs={'use_acquisition_time_as_start_time': True})

    As this time is different for every channel, time-dependent calculations like SZA correction
    can be pretty slow when multiple channels are used.

    The exact scanline times are always available as coordinates of an individual channels::

        scene.load(["B03"])
        print(scene["B03].coords["acq_time"].data)

    would print something similar to::

        array(['2021-12-08T06:00:20.131200000', '2021-12-08T06:00:20.191948000',
               '2021-12-08T06:00:20.252695000', ...,
               '2021-12-08T06:09:39.449390000', '2021-12-08T06:09:39.510295000',
               '2021-12-08T06:09:39.571200000'], dtype='datetime64[ns]')

    The first value represents the exact start time, and the last one the exact end time of the data
    acquisition.

    """

    def __init__(self, filename, filename_info, filetype_info, use_acquisition_time_as_start_time=False):
        """Initialize the reader."""
        super(HRITJMAFileHandler, self).__init__(filename, filename_info,
                                                 filetype_info,
                                                 (jma_hdr_map,
                                                  jma_variable_length_headers,
                                                  jma_text_headers))

        self._use_acquisition_time_as_start_time = use_acquisition_time_as_start_time
        self.mda['segment_sequence_number'] = self.mda['image_segm_seq_no']
        self.mda['planned_end_segment_number'] = self.mda['total_no_image_segm']
        self.mda['planned_start_segment_number'] = 1

        items = self.mda['image_data_function'].decode().split('\r')
        if items[0].startswith('$HALFTONE'):
            self.calibration_table = []
            for item in items[1:]:
                if item == '':
                    continue
                key, value = item.split(':=')
                if key.startswith('_UNIT'):
                    self.mda['unit'] = item.split(':=')[1]
                elif key.startswith('_NAME'):
                    pass
                elif key.isdigit():
                    key = int(key)
                    value = float(value)
                    self.calibration_table.append((key, value))

            self.calibration_table = np.array(self.calibration_table)

        self.projection_name = self.mda['projection_name'].decode().strip()
        sublon = float(self.projection_name.split('(')[1][:-1])
        self.mda['projection_parameters']['SSP_longitude'] = sublon
        self.platform = self._get_platform()
        self.is_segmented = self.mda['segment_sequence_number'] > 0
        self.area_id = filename_info.get('area', UNKNOWN_AREA)
        if self.area_id not in AREA_NAMES:
            self.area_id = UNKNOWN_AREA
        self.area = self._get_area_def()
        self.acq_time = self._get_acq_time()

    def _get_platform(self):
        """Get the platform name.

        The platform is not specified explicitly in JMA HRIT files. For
        segmented data it is not even specified in the filename. But it
        can be derived indirectly from the projection name:

            GEOS(140.00): MTSAT-1R
            GEOS(140.25): MTSAT-1R    # TODO: Check if there is more...
            GEOS(140.70): Himawari-8
            GEOS(145.00): MTSAT-2

        See [MTSAT], section 3.1. Unfortunately Himawari-8 and 9 are not
        distinguishable using that method at the moment. From [HIMAWARI]:

        "HRIT/LRIT files have the same file naming convention in the same
        format in Himawari-8 and Himawari-9, so there is no particular
        difference."

        TODO: Find another way to distinguish Himawari-8 and 9.

        References:
        [MTSAT] http://www.data.jma.go.jp/mscweb/notice/Himawari7_e.html
        [HIMAWARI] http://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/sample_hrit.html

        """
        try:
            return PLATFORMS[self.projection_name]
        except KeyError:
            logger.error('Unable to determine platform: Unknown projection '
                         'name "{}"'.format(self.projection_name))
            return UNKNOWN_PLATFORM

    def _check_sensor_platform_consistency(self, sensor):
        """Make sure sensor and platform are consistent.

        Args:
            sensor (str) : Sensor name from YAML dataset definition

        Raises:
            ValueError if they don't match

        """
        ref_sensor = SENSORS.get(self.platform, None)
        if ref_sensor and not sensor == ref_sensor:
            logger.error('Sensor-Platform mismatch: {} is not a payload '
                         'of {}. Did you choose the correct reader?'
                         .format(sensor, self.platform))

    def _get_line_offset(self):
        """Get line offset for the current segment.

        Read line offset from the file and adapt it to the current segment
        or half disk scan so that

            y(l) ~ l - loff

        because this is what get_geostationary_area_extent() expects.
        """
        # Get line offset from the file
        nlines = int(self.mda['number_of_lines'])
        loff = np.float32(self.mda['loff'])

        # Adapt it to the current segment
        if self.is_segmented:
            # loff in the file specifies the offset of the full disk image
            # centre (1375/2750 for VIS/IR)
            segment_number = self.mda['segment_sequence_number'] - 1
            loff -= (self.mda['total_no_image_segm'] - segment_number - 1) * nlines
        elif self.area_id in (NORTH_HEMIS, SOUTH_HEMIS):
            # loff in the file specifies the start line of the half disk image
            # in the full disk image
            loff = nlines - loff
        elif self.area_id == UNKNOWN_AREA:
            logger.error('Cannot compute line offset for unknown area')

        return loff

    def _get_area_def(self):
        """Get the area definition of the band."""
        pdict = {
            'cfac': np.int32(self.mda['cfac']),
            'lfac': np.int32(self.mda['lfac']),
            'coff': np.float32(self.mda['coff']),
            'loff': self._get_line_offset(),
            'ncols': int(self.mda['number_of_columns']),
            'nlines': int(self.mda['number_of_lines']),
            'scandir': 'N2S',
            'a': float(self.mda['projection_parameters']['a']),
            'b': float(self.mda['projection_parameters']['b']),
            'h': float(self.mda['projection_parameters']['h']),
            'ssp_lon': float(self.mda['projection_parameters']['SSP_longitude']),
            'a_name': AREA_NAMES[self.area_id]['short'],
            'a_desc': AREA_NAMES[self.area_id]['long'],
            'p_id': 'geosmsg'
        }
        area_extent = get_area_extent(pdict)
        return get_area_definition(pdict, area_extent)

    def get_area_def(self, dsid):
        """Get the area definition of the band."""
        return self.area

    def get_dataset(self, key, info):
        """Get the dataset designated by *key*."""
        res = super(HRITJMAFileHandler, self).get_dataset(key, info)

        # Filenames of segmented data is identical for MTSAT-1R, MTSAT-2
        # and Himawari-8/9. Make sure we have the correct reader for the data
        # at hand.
        self._check_sensor_platform_consistency(info['sensor'])

        # Calibrate and mask space pixels
        res = self._mask_space(self.calibrate(res, key.calibration))

        # Add scanline acquisition time
        res.coords['acq_time'] = ('y', self.acq_time)
        res.coords['acq_time'].attrs['long_name'] = 'Scanline acquisition time'

        # Update attributes
        res.attrs.update(info)
        res.attrs['platform_name'] = self.platform
        res.attrs['orbital_parameters'] = {
            'projection_longitude': float(self.mda['projection_parameters']['SSP_longitude']),
            'projection_latitude': 0.,
            'projection_altitude': float(self.mda['projection_parameters']['h'])}

        return res

    def _mask_space(self, data):
        """Mask space pixels."""
        geomask = get_geostationary_mask(area=self.area)
        return data.where(geomask)

    def _get_acq_time(self):
        r"""Get the acquisition times from the file.

        Acquisition times for a subset of scanlines are stored in the header
        as follows:

        b'LINE:=1\rTIME:=54365.022558\rLINE:=21\rTIME:=54365.022664\r...'

        Missing timestamps in between are computed using linear interpolation.
        """
        buf_b = np.frombuffer(self.mda['image_observation_time'],
                              dtype=image_observation_time)

        # Replace \r by \n before encoding, otherwise encoding will drop all
        # elements except the last one
        buf_s = b''.join(buf_b['times']).replace(b'\r', b'\n').decode()

        # Split into key:=value pairs; then extract line number and timestamp
        splits = buf_s.strip().split('\n')
        lines_sparse = [int(s.split(':=')[1]) for s in splits[0::2]]
        times_sparse = [float(s.split(':=')[1]) for s in splits[1::2]]

        if self.platform == HIMAWARI8:
            # Only a couple of timestamps in the header, and only the first
            # and last are usable (duplicates inbetween).
            lines_sparse = [lines_sparse[0], lines_sparse[-1]]
            times_sparse = [times_sparse[0], times_sparse[-1]]

        # Compute missing timestamps using linear interpolation.
        lines = np.arange(lines_sparse[0], lines_sparse[-1]+1)
        times = np.interp(lines, lines_sparse, times_sparse)

        # Convert to np.datetime64
        times64 = mjd2datetime64(times)

        return times64

    @staticmethod
    def _interp(arr, cal):
        return np.interp(arr.ravel(), cal[:, 0], cal[:, 1]).reshape(arr.shape)

    def calibrate(self, data, calibration):
        """Calibrate the data."""
        tic = datetime.now()

        if calibration == 'counts':
            return data
        if calibration == 'radiance':
            raise NotImplementedError("Can't calibrate to radiance.")

        cal = self.calibration_table
        res = data.data.map_blocks(self._interp, cal, dtype=cal[:, 0].dtype)
        res = xr.DataArray(res,
                           dims=data.dims, attrs=data.attrs,
                           coords=data.coords)
        res = res.where(data < 65535)
        logger.debug("Calibration time " + str(datetime.now() - tic))
        return res

    @property
    def start_time(self):
        """Get start time of the scan."""
        if self._use_acquisition_time_as_start_time:
            return self.acq_time[0].astype(datetime)
        return self._start_time

    @property
    def end_time(self):
        """Get end time of the scan."""
        return self.acq_time[-1].astype(datetime)
