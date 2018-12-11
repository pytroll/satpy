#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2010-2017

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

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

    def get_area_def(self, dsid):
        """Get the area definition of the band."""
        cfac = np.int32(self.mda['cfac'])
        lfac = np.int32(self.mda['lfac'])
        coff = np.float32(self.mda['coff'])
        loff = np.float32(self.mda['loff'])

        a = self.mda['projection_parameters']['a']
        b = self.mda['projection_parameters']['b']
        h = self.mda['projection_parameters']['h']
        lon_0 = self.mda['projection_parameters']['SSP_longitude']

        nlines = int(self.mda['number_of_lines'])
        ncols = int(self.mda['number_of_columns'])

        segment_number = self.mda['segment_sequence_number'] - 1

        loff -= segment_number * nlines

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
            "FLDK",
            "HRIT FLDK Area: {}".format(self.projection_name),
            'geosmsg',
            proj_dict,
            ncols,
            nlines,
            area_extent)
        return area

    def get_dataset(self, key, info):
        """Get the dataset designated by *key*."""
        res = super(HRITJMAFileHandler, self).get_dataset(key, info)

        res = self.calibrate(res, key.calibration)
        res.attrs.update(info)
        res.attrs['platform_name'] = 'Himawari-8'
        res.attrs['sensor'] = 'ahi'
        res.attrs['satellite_longitude'] = float(self.mda['projection_parameters']['SSP_longitude'])
        res.attrs['satellite_latitude'] = 0.
        res.attrs['satellite_altitude'] = float(self.mda['projection_parameters']['h'])
        return res

    def calibrate(self, data, calibration):
        """Calibrate the data."""
        tic = datetime.now()

        if calibration == 'counts':
            return data
        elif calibration == 'radiance':
            raise NotImplementedError("Can't calibrate to radiance.")
        else:
            cal = self.calibration_table

            def interp(arr):
                return np.interp(arr.ravel(),
                                 cal[:, 0], cal[:, 1]).reshape(arr.shape)

            res = data.data.map_blocks(interp, dtype=cal[:, 0].dtype)

            res = xr.DataArray(res,
                               dims=data.dims, attrs=data.attrs,
                               coords=data.coords)
        res = res.where(data > 0)
        logger.debug("Calibration time " + str(datetime.now() - tic))
        return res
