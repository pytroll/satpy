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
"""HRIT format reader for JMA data
************************************

References:
    JMA HRIT - Mission Specific Implementation
    http://www.jma.go.jp/jma/jma-eng/satellite/introduction/4_2HRIT.pdf

"""

import logging
from datetime import datetime

import numpy as np
import xarray as xr

from pyresample import geometry
from satpy.readers.hrit_base import (HRITFileHandler, ancillary_text,
                                     annotation_header, base_hdr_map,
                                     image_data_function)
from .utils import get_geostationary_mask

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


jma_variable_length_headers = {}

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


class HRITJMAFileHandler(HRITFileHandler):
    """JMA HRIT format reader."""

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the reader."""
        super(HRITJMAFileHandler, self).__init__(filename, filename_info,
                                                 filetype_info,
                                                 (jma_hdr_map,
                                                  jma_variable_length_headers,
                                                  jma_text_headers))

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

    def _get_platform(self):
        """Get the platform name

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
        """Make sure sensor and platform are consistent

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
        """Get line offset for the current segment

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
        cfac = np.int32(self.mda['cfac'])
        lfac = np.int32(self.mda['lfac'])
        coff = np.float32(self.mda['coff'])
        loff = self._get_line_offset()

        a = self.mda['projection_parameters']['a']
        b = self.mda['projection_parameters']['b']
        h = self.mda['projection_parameters']['h']
        lon_0 = self.mda['projection_parameters']['SSP_longitude']

        nlines = int(self.mda['number_of_lines'])
        ncols = int(self.mda['number_of_columns'])

        area_extent = self.get_area_extent((nlines, ncols),
                                           (loff, coff),
                                           (lfac, cfac),
                                           h)

        proj_dict = {'a': float(a),
                     'b': float(b),
                     'lon_0': float(lon_0),
                     'h': float(h),
                     'proj': 'geos',
                     'units': 'm'}

        area = geometry.AreaDefinition(
            AREA_NAMES[self.area_id]['short'],
            AREA_NAMES[self.area_id]['long'],
            'geosmsg',
            proj_dict,
            ncols,
            nlines,
            area_extent)

        return area

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

        # Update attributes
        res.attrs.update(info)
        res.attrs['platform_name'] = self.platform
        res.attrs['satellite_longitude'] = float(self.mda['projection_parameters']['SSP_longitude'])
        res.attrs['satellite_latitude'] = 0.
        res.attrs['satellite_altitude'] = float(self.mda['projection_parameters']['h'])
        res.attrs['orbital_parameters'] = {
            'projection_longitude': float(self.mda['projection_parameters']['SSP_longitude']),
            'projection_latitude': 0.,
            'projection_altitude': float(self.mda['projection_parameters']['h'])}

        return res

    def _mask_space(self, data):
        """Mask space pixels"""
        geomask = get_geostationary_mask(area=self.area)
        return data.where(geomask)

    @staticmethod
    def _interp(arr, cal):
        return np.interp(arr.ravel(), cal[:, 0], cal[:, 1]).reshape(arr.shape)

    def calibrate(self, data, calibration):
        """Calibrate the data."""
        tic = datetime.now()

        if calibration == 'counts':
            return data
        elif calibration == 'radiance':
            raise NotImplementedError("Can't calibrate to radiance.")
        else:
            cal = self.calibration_table
            res = data.data.map_blocks(self._interp, cal, dtype=cal[:, 0].dtype)
            res = xr.DataArray(res,
                               dims=data.dims, attrs=data.attrs,
                               coords=data.coords)
        res = res.where(data > 0)
        logger.debug("Calibration time " + str(datetime.now() - tic))
        return res
