#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2014, 2015, 2016 Adam.Dybbroe

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
from pyresample import geometry

from satpy.dataset import Dataset
from satpy.readers.file_handlers import BaseFileHandler


class CalibrationError(Exception):
    pass

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

image_data_function = np.dtype('|S1')

annotation_header = np.dtype('|S1')

time_cds_short = np.dtype([('days', '>u2'),
                           ('milliseconds', '>u4')])


def make_time_cds_short(tcds_array):
    return (datetime(1958, 1, 1) +
            timedelta(days=int(tcds_array['days']),
                      milliseconds=int(tcds_array['milliseconds'])))

timestamp_record = np.dtype([('cds_p_field', 'u1'),
                             ('timestamp', time_cds_short)])

ancillary_text = np.dtype('|S1')

key_header = np.dtype('|S1')

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
    arr10 = inbuf.astype(np.uint16)
    arr16 = np.zeros((len(arr10) * 4 / 5,), dtype=np.uint16)
    arr10_len = (len(arr16) * 5) / 4
    arr10 = arr10[:arr10_len]  # adjust size
    """
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
    arr16.flat[::4] = np.left_shift(arr10[::5], 2) + \
        np.right_shift((arr10[1::5]), 6)
    arr16.flat[1::4] = np.left_shift((arr10[1::5] & 63), 4) + \
        np.right_shift((arr10[2::5]), 4)
    arr16.flat[2::4] = np.left_shift(arr10[2::5] & 15, 6) + \
        np.right_shift((arr10[3::5]), 2)
    arr16.flat[3::4] = np.left_shift(arr10[3::5] & 3, 8) + \
        arr10[4::5]
    return arr16


class HRITFileHandler(BaseFileHandler):

    """HRIT standard format reader
    """

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
                    field_length = (
                        hdr_id['record_length'] - 3) / the_type.itemsize
                    current_hdr = np.fromfile(fp,
                                              dtype=the_type,
                                              count=field_length)
                    self.mda[variable_length_headers[
                        the_type]] = current_hdr
                elif the_type in text_headers:
                    field_length = (
                        hdr_id['record_length'] - 3) / the_type.itemsize
                    new_type = np.dtype(the_type.char + str(field_length))
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
            try:
                self.mda['timestamp'] = make_time_cds_short(
                    self.mda['timestamp'])

                self._end_time = self.mda['timestamp']
            except KeyError:
                self._end_time = self._start_time + timedelta(minutes=15)

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

    def get_dataset(self, key, info, out=None, xslice=slice(None), yslice=slice(None)):
        to_return = out is None
        if out is None:
            nlines = int(self.mda['number_of_lines'])
            ncols = int(self.mda['number_of_columns'])
            out = Dataset(np.ma.empty((nlines, ncols), dtype=np.float32))

        self.read_band(key, info, out, xslice, yslice)

        if to_return:
            from satpy.yaml_reader import Shuttle
            return Shuttle(out.data, out.mask, out.info)

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

        segment_number = self.mda['segment_sequence_number']
        total_segments = self.mda[
            'planned_end_segment_number'] - self.mda['planned_start_segment_number'] + 1

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

    def read_band(self, key, info,
                  out=None, xslice=slice(None), yslice=slice(None)):
        """Read the data"""
        # TODO slicing !
        tic = datetime.now()
        with open(self.filename, "rb") as fp_:
            fp_.seek(self.mda['total_header_length'])
            data = np.fromfile(fp_, dtype=np.uint8, count=int(np.ceil(
                self.mda['data_field_length'] / 8.)))
            out.data[:] = dec10216(data).reshape((self.mda['number_of_lines'],
                                                  self.mda['number_of_columns']))[yslice, xslice] * 1.0
            out.mask[:] = out.data == 0
        logger.debug("Reading time " + str(datetime.now() - tic))

        # new_info = dict(units=info['units'],
        #                 standard_name=info['standard_name'],
        #                 wavelength=info['wavelength'],
        #                 resolution='resolution',
        #                 id=key,
        #                 name=key.name,
        #                 platform_name=self.platform_name,
        #                 sensor=self.sensor,
        #                 satellite_longitude=float(
        #                     self.nav_info['SSP_longitude']),
        #                 satellite_latitude=float(
        #                     self.nav_info['SSP_latitude']),
        #                 satellite_altitude=float(self.nav_info['distance_earth_center_to_satellite'] -
        #                                          self.proj_info['earth_equatorial_radius']) * 1000)
        # out.info.update(new_info)
