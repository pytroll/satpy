#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2014, 2015, 2016, 2017 Adam.Dybbroe

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

"""HRIT format reader

"""

import logging
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

import dask.array as da
from pyresample import geometry
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger('hrit_base')


common_hdr = np.dtype([('hdr_id', 'u1'),
                       ('record_length', '>u2')])

primary_header = np.dtype([('file_type', 'u1'),
                           ('total_header_length', '>u4'),
                           ('data_field_length', '>u8')])

image_structure = np.dtype([('number_of_bits_per_pixel', 'u1'),
                            ('number_of_columns', '>u2'),
                            ('number_of_lines', '>u2'),
                            ('compression_flag_for_data', 'u1')])

image_navigation = np.dtype([('projection_name', 'S32'),
                             ('cfac', '>i4'),
                             ('lfac', '>i4'),
                             ('coff', '>i4'),
                             ('loff', '>i4')])

image_data_function = np.dtype([('function', '|S1')])

annotation_header = np.dtype([('annotation', '|S1')])

time_cds_short = np.dtype([('days', '>u2'),
                           ('milliseconds', '>u4')])


def make_time_cds_short(tcds_array):
    return (datetime(1958, 1, 1) +
            timedelta(days=int(tcds_array['days']),
                      milliseconds=int(tcds_array['milliseconds'])))


timestamp_record = np.dtype([('cds_p_field', 'u1'),
                             ('timestamp', time_cds_short)])

ancillary_text = np.dtype([('ancillary', '|S1')])

key_header = np.dtype([('key', '|S1')])

base_variable_length_headers = {}

base_text_headers = {image_data_function: 'image_data_function',
                     annotation_header: 'annotation_header',
                     ancillary_text: 'ancillary_text',
                     key_header: 'key_header'}

base_hdr_map = {0: primary_header,
                1: image_structure,
                2: image_navigation,
                3: image_data_function,
                4: annotation_header,
                5: timestamp_record,
                6: ancillary_text,
                7: key_header,
                }


def dec10216(inbuf):
    """Decode 10 bits data into 16 bits words.

    ::

        /*
         * pack 4 10-bit words in 5 bytes into 4 16-bit words
         *
         * 0       1       2       3       4       5
         * 01234567890123456789012345678901234567890
         * 0         1         2         3         4
         */
        ip = &in_buffer[i];
        op = &out_buffer[j];
        op[0] = ip[0]*4 + ip[1]/64;
        op[1] = (ip[1] & 0x3F)*16 + ip[2]/16;
        op[2] = (ip[2] & 0x0F)*64 + ip[3]/4;
        op[3] = (ip[3] & 0x03)*256 +ip[4];

    """
    arr10 = inbuf.astype(np.uint16)
    arr16_len = int(len(arr10) * 4 / 5)
    arr10_len = int((arr16_len * 5) / 4)
    arr10 = arr10[:arr10_len]  # adjust size

    # dask is slow with indexing
    arr10_0 = arr10[::5]
    arr10_1 = arr10[1::5]
    arr10_2 = arr10[2::5]
    arr10_3 = arr10[3::5]
    arr10_4 = arr10[4::5]

    arr16_0 = (arr10_0 << 2) + (arr10_1 >> 6)
    arr16_1 = ((arr10_1 & 63) << 4) + (arr10_2 >> 4)
    arr16_2 = ((arr10_2 & 15) << 6) + (arr10_3 >> 2)
    arr16_3 = ((arr10_3 & 3) << 8) + arr10_4
    arr16 = da.stack([arr16_0, arr16_1, arr16_2, arr16_3], axis=-1).ravel()
    arr16 = da.rechunk(arr16, arr16.shape[0])

    return arr16


class HRITFileHandler(BaseFileHandler):
    """HRIT standard format reader."""

    def __init__(self, filename, filename_info, filetype_info, hdr_info):
        """Initialize the reader."""
        super(HRITFileHandler, self).__init__(filename, filename_info,
                                              filetype_info)

        self.mda = {}

        hdr_map, variable_length_headers, text_headers = hdr_info

        with open(self.filename) as fp:
            total_header_length = 16
            while fp.tell() < total_header_length:
                hdr_id = np.fromfile(fp, dtype=common_hdr, count=1)[0]
                the_type = hdr_map[hdr_id['hdr_id']]
                if the_type in variable_length_headers:
                    field_length = int((hdr_id['record_length'] - 3) /
                                       the_type.itemsize)
                    current_hdr = np.fromfile(fp,
                                              dtype=the_type,
                                              count=field_length)
                    key = variable_length_headers[the_type]
                    if key in self.mda:
                        if not isinstance(self.mda[key], list):
                            self.mda[key] = [self.mda[key]]
                        self.mda[key].append(current_hdr)
                    else:
                        self.mda[key] = current_hdr
                elif the_type in text_headers:
                    field_length = int((hdr_id['record_length'] - 3) /
                                       the_type.itemsize)
                    char = list(the_type.fields.values())[0][0].char
                    new_type = np.dtype(char + str(field_length))
                    current_hdr = np.fromfile(fp,
                                              dtype=new_type,
                                              count=1)[0]
                    self.mda[text_headers[the_type]] = current_hdr
                else:
                    current_hdr = np.fromfile(fp,
                                              dtype=the_type,
                                              count=1)[0]
                    self.mda.update(
                        dict(zip(current_hdr.dtype.names, current_hdr)))

                total_header_length = self.mda['total_header_length']

            self._start_time = filename_info['start_time']
            self._end_time = self._start_time + timedelta(minutes=15)

            self.mda.setdefault('number_of_bits_per_pixel', 10)

            self.mda['projection_parameters'] = {'a': 6378169.00,
                                                 'b': 6356583.80,
                                                 'h': 35785831.00,
                                                 # FIXME: find a reasonable SSP
                                                 'SSP_longitude': 0.0}

    def get_shape(self, dsid, ds_info):
        return int(self.mda['number_of_lines']), int(self.mda['number_of_columns'])

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time

    def get_dataset(self, key, info):
        """Load a dataset."""
        # Read bands
        data = self.read_band(key, info)

        # Convert to xarray
        xdata = xr.DataArray(data, dims=['y', 'x'])

        return xdata

    def get_xy_from_linecol(self, line, col, offsets, factors):
        """Get the intermediate coordinates from line & col.

        Intermediate coordinates are actually the instruments scanning angles.
        """
        loff, coff = offsets
        lfac, cfac = factors
        x__ = (col - coff) / cfac * 2**16
        y__ = (line - loff) / lfac * 2**16

        return x__, y__

    def get_area_extent(self, size, offsets, factors, platform_height):
        """Get the area extent of the file."""
        nlines, ncols = size
        h = platform_height

        # count starts at 1
        cols = 1 - 0.5
        lines = 1 - 0.5
        ll_x, ll_y = self.get_xy_from_linecol(lines, cols, offsets, factors)

        cols += ncols
        lines += nlines
        ur_x, ur_y = self.get_xy_from_linecol(lines, cols, offsets, factors)

        return (np.deg2rad(ll_x) * h, np.deg2rad(ll_y) * h,
                np.deg2rad(ur_x) * h, np.deg2rad(ur_y) * h)

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

    def read_band(self, key, info):
        """Read the data."""
        # TODO slicing !
        tic = datetime.now()

        shape = int(np.ceil(self.mda['data_field_length'] / 8.))
        if self.mda['number_of_bits_per_pixel'] == 16:
            dtype = '>u2'
            shape /= 2
        elif self.mda['number_of_bits_per_pixel'] in [8, 10]:
            dtype = np.uint8
        shape = (shape, )

        data = np.memmap(self.filename, mode='r',
                         offset=self.mda['total_header_length'],
                         dtype=dtype,
                         shape=shape)
        data = da.from_array(data, chunks=shape[0])
        if self.mda['number_of_bits_per_pixel'] == 10:
            data = dec10216(data)
        data = data.reshape((self.mda['number_of_lines'],
                             self.mda['number_of_columns']))
        logger.debug("Reading time " + str(datetime.now() - tic))
        return data
