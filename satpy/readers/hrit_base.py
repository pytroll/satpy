#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2014-2018 Satpy developers
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
"""HRIT/LRIT format reader
***************************

This module is the base module for all HRIT-based formats. Here, you will find
the common building blocks for hrit reading.

One of the features here is the on-the-fly decompression of hrit files. It needs
a path to the xRITDecompress binary to be provided through the environment
variable called XRIT_DECOMPRESS_PATH. When compressed hrit files are then
encountered (files finishing with `.C_`), they are decompressed to the system's
temporary directory for reading.

"""

import logging
from datetime import timedelta
from tempfile import gettempdir
import os
from six import BytesIO
from subprocess import Popen, PIPE

import numpy as np
import xarray as xr

import dask.array as da
from pyresample import geometry
from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.eum_base import time_cds_short
from satpy.readers.seviri_base import dec10216

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


def get_xritdecompress_cmd():
    """Find a valid binary for the xRITDecompress command."""
    cmd = os.environ.get('XRIT_DECOMPRESS_PATH', None)
    if not cmd:
        raise IOError("XRIT_DECOMPRESS_PATH is not defined (complete path to xRITDecompress)")

    question = ("Did you set the environment variable XRIT_DECOMPRESS_PATH correctly?")
    if not os.path.exists(cmd):
        raise IOError(str(cmd) + " does not exist!\n" + question)
    elif os.path.isdir(cmd):
        raise IOError(str(cmd) + " is a directory!\n" + question)

    return cmd


def get_xritdecompress_outfile(stdout):
    """Analyse the output of the xRITDecompress command call and return the file."""
    outfile = b''
    for line in stdout:
        try:
            k, v = [x.strip() for x in line.split(b':', 1)]
        except ValueError:
            break
        if k == b'Decompressed file':
            outfile = v
            break

    return outfile


def decompress(infile, outdir='.'):
    """Decompress an XRIT data file and return the path to the decompressed file.

    It expect to find Eumetsat's xRITDecompress through the environment variable
    XRIT_DECOMPRESS_PATH.
    """
    cmd = get_xritdecompress_cmd()
    infile = os.path.abspath(infile)
    cwd = os.getcwd()
    os.chdir(outdir)

    p = Popen([cmd, infile], stdout=PIPE)
    stdout = BytesIO(p.communicate()[0])
    status = p.returncode
    os.chdir(cwd)

    if status != 0:
        raise IOError("xrit_decompress '%s', failed, status=%d" % (infile, status))

    outfile = get_xritdecompress_outfile(stdout)

    if not outfile:
        raise IOError("xrit_decompress '%s', failed, no output file is generated" % infile)

    return os.path.join(outdir, outfile.decode('utf-8'))


class HRITFileHandler(BaseFileHandler):

    """HRIT standard format reader."""

    def __init__(self, filename, filename_info, filetype_info, hdr_info):
        """Initialize the reader."""
        super(HRITFileHandler, self).__init__(filename, filename_info,
                                              filetype_info)
        self.mda = {}
        self._get_hd(hdr_info)

        if self.mda.get('compression_flag_for_data'):
            logger.debug('Unpacking %s', filename)
            try:
                self.filename = decompress(filename, gettempdir())
            except IOError as err:
                logger.warning("Unpacking failed: %s", str(err))
            self.mda = {}
            self._get_hd(hdr_info)

        self._start_time = filename_info['start_time']
        self._end_time = self._start_time + timedelta(minutes=15)

    def _get_hd(self, hdr_info):
        """Open the file, read and get the basic file header info and set the mda
           dictionary
        """

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

        self.mda.setdefault('number_of_bits_per_pixel', 10)

        self.mda['projection_parameters'] = {'a': 6378169.00,
                                             'b': 6356583.80,
                                             'h': 35785831.00,
                                             # FIXME: find a reasonable SSP
                                             'SSP_longitude': 0.0}
        self.mda['orbital_parameters'] = {}

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
        shape = int(np.ceil(self.mda['data_field_length'] / 8.))
        if self.mda['number_of_bits_per_pixel'] == 16:
            dtype = '>u2'
            shape //= 2
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
        return data
