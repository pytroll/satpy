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

"""HRIT format reader for JMA data.

References:
    JMA HRIT - Mission Specific Implementation
    http://www.jma.go.jp/jma/jma-eng/satellite/introduction/4_2HRIT.pdf

"""

import logging
from datetime import datetime, timedelta

import numpy as np

from pyresample import geometry
from satpy.readers.hrit_base import (HRITFileHandler, ancillary_text,
                                     annotation_header, base_hdr_map,
                                     image_data_function, make_time_cds_short,
                                     time_cds_short)

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


def make_time_cds_expanded(tcds_array):
    return (datetime(1958, 1, 1) +
            timedelta(days=int(tcds_array['days']),
                      milliseconds=int(tcds_array['milliseconds']),
                      microseconds=float(tcds_array['microseconds'] +
                                         tcds_array['nanoseconds'] / 1000.)))


def recarray2dict(arr):
    res = {}
    for dtuple in arr.dtype.descr:
        key = dtuple[0]
        ntype = dtuple[1]
        data = arr[key]
        if isinstance(ntype, list):
            res[key] = recarray2dict(data)
        else:
            res[key] = data

    return res


class HRITJMAFileHandler(HRITFileHandler):

    """JMA HRIT format reader
    """

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

        items = self.mda['image_data_function'].split('\r')
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

        sublon = float(self.mda['projection_name'].strip().split('(')[1][:-1])
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
        total_segments = (self.mda['planned_end_segment_number'] -
                          self.mda['planned_start_segment_number'] + 1)

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
            'some_area_name',
            "On-the-fly area",
            'geosmsg',
            proj_dict,
            ncols,
            nlines,
            area_extent)

        self.area = area

        return area

    def get_dataset(self, key, info, out=None,
                    xslice=slice(None), yslice=slice(None)):
        """Get the dataset designated by *key*."""
        res = super(HRITJMAFileHandler, self).get_dataset(key, info, out,
                                                          xslice, yslice)
        if res is not None:
            out = res

        self.calibrate(out, key.calibration)
        out.info['units'] = info['units']
        out.info['standard_name'] = info['standard_name']
        #out.info['platform_name'] = self.platform_name
        out.info['platform_name'] = 'Himawari-8'
        out.info['sensor'] = 'ahi'

    def calibrate(self, data, calibration):
        """Calibrate the data."""
        tic = datetime.now()

        if calibration == 'counts':
            return
        elif calibration == 'radiance':
            raise NotImplementedError("Can't calibrate to radiance.")
        else:
            cal = self.calibration_table

            data.data[:] = np.interp(data.data.ravel(),
                                     cal[:, 0], cal[:, 1]).reshape(data.data.shape)
        logger.debug("Calibration time " + str(datetime.now() - tic))


def show(data, negate=False):
    """Show the stretched data.
    """
    from PIL import Image as pil
    data = np.array((data - data.min()) * 255.0 /
                    (data.max() - data.min()), np.uint8)
    if negate:
        data = 255 - data
    img = pil.fromarray(data)
    img.show()


if __name__ == "__main__":

    # TESTFILE = ("/media/My Passport/HIMAWARI-8/HISD/Hsfd/" +
    #            "201502/07/201502070200/00/B13/" +
    #            "HS_H08_20150207_0200_B13_FLDK_R20_S0101.DAT")
    TESTFILE = ("/local_disk/data/himawari8/testdata/" +
                "HS_H08_20130710_0300_B13_FLDK_R20_S1010.DAT")
    #"HS_H08_20130710_0300_B01_FLDK_R10_S1010.DAT")
    SCENE = ahisf([TESTFILE])
    SCENE.read_band(TESTFILE)
    SCENE.calibrate(['13'])
    # SCENE.calibrate(['13'], calibrate=0)

    # print SCENE._data['13']['counts'][0].shape

    show(SCENE.channels['13'], negate=False)

    import matplotlib.pyplot as plt
    plt.imshow(SCENE.channels['13'])
    plt.colorbar()
    plt.show()
