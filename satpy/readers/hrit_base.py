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
"""HRIT/LRIT format reader.

This module is the base module for all HRIT-based formats. Here, you will find
the common building blocks for hrit reading.

One of the features here is the on-the-fly decompression of hrit files. It needs
a path to the xRITDecompress binary to be provided through the environment
variable called XRIT_DECOMPRESS_PATH. When compressed hrit files are then
encountered (files finishing with `.C_`), they are decompressed to the system's
temporary directory for reading.

"""

import logging
import os
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from io import BytesIO
from subprocess import PIPE, Popen  # nosec B404

import dask
import dask.array as da
import numpy as np
import xarray as xr
from pyresample import geometry

import satpy.readers.utils as utils
from satpy import config
from satpy.readers import FSFile
from satpy.readers.eum_base import time_cds_short
from satpy.readers.file_handlers import BaseFileHandler
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

    p = Popen([cmd, infile], stdout=PIPE)  # nosec B603
    stdout = BytesIO(p.communicate()[0])
    status = p.returncode
    os.chdir(cwd)

    if status != 0:
        raise IOError("xrit_decompress '%s', failed, status=%d" % (infile, status))

    outfile = get_xritdecompress_outfile(stdout)

    if not outfile:
        raise IOError("xrit_decompress '%s', failed, no output file is generated" % infile)

    return os.path.join(outdir, outfile.decode('utf-8'))


def get_header_id(fp):
    """Return the HRIT header common data."""
    data = fp.read(common_hdr.itemsize)
    return np.frombuffer(data, dtype=common_hdr, count=1)[0]


def get_header_content(fp, header_dtype, count=1):
    """Return the content of the HRIT header."""
    data = fp.read(header_dtype.itemsize * count)
    return np.frombuffer(data, dtype=header_dtype, count=count)


class HRITFileHandler(BaseFileHandler):
    """HRIT standard format reader."""

    def __init__(self, filename, filename_info, filetype_info, hdr_info):
        """Initialize the reader."""
        super(HRITFileHandler, self).__init__(filename, filename_info,
                                              filetype_info)

        self.mda = {}
        self.hdr_info = hdr_info
        self._get_hd(self.hdr_info)

        self._start_time = filename_info['start_time']
        self._end_time = self._start_time + timedelta(minutes=15)

    def _get_hd(self, hdr_info):
        """Open the file, read and get the basic file header info and set the mda dictionary."""
        hdr_map, variable_length_headers, text_headers = hdr_info

        with utils.generic_open(self.filename, mode='rb') as fp:
            total_header_length = 16
            while fp.tell() < total_header_length:
                hdr_id = get_header_id(fp)
                the_type = hdr_map[hdr_id['hdr_id']]
                if the_type in variable_length_headers:
                    field_length = int((hdr_id['record_length'] - 3) /
                                       the_type.itemsize)
                    current_hdr = get_header_content(fp, the_type, field_length)
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
                    current_hdr = get_header_content(fp, new_type)[0]
                    self.mda[text_headers[the_type]] = current_hdr
                else:
                    current_hdr = get_header_content(fp, the_type)[0]
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

    @property
    def start_time(self):
        """Get start time."""
        return self._start_time

    @property
    def end_time(self):
        """Get end time."""
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
        output_dtype, output_shape = self._get_output_info()
        return da.from_delayed(_read_data(self.filename, self.mda),
                               shape=output_shape,
                               dtype=output_dtype)

    def _get_output_info(self):
        bpp = self.mda['number_of_bits_per_pixel']
        if bpp in [10, 16]:
            output_dtype = np.uint16
        elif bpp == 8:
            output_dtype = np.uint8
        else:
            raise ValueError(f"Unexpected number of bits per pixel: {bpp}")
        output_shape = (self.mda['number_of_lines'], self.mda['number_of_columns'])
        return output_dtype, output_shape


@dask.delayed
def _read_data(filename, mda):
    return HRITSegment(filename, mda).read_data()


@contextmanager
def decompressed(filename):
    """Decompress context manager."""
    try:
        new_filename = decompress(filename, config["tmp_dir"])
    except IOError as err:
        logger.error("Unpacking failed: %s", str(err))
        raise
    yield new_filename
    os.remove(new_filename)


class HRITSegment:
    """An HRIT segment with data."""

    def __init__(self, filename, mda):
        """Set up the segment."""
        self.filename = filename
        self.mda = mda
        self.lines = mda['number_of_lines']
        self.cols = mda['number_of_columns']
        self.bpp = mda['number_of_bits_per_pixel']
        self.compressed = mda['compression_flag_for_data'] == 1
        self.offset = mda['total_header_length']
        self.zipped = os.fspath(filename).endswith('.bz2')

    def read_data(self):
        """Read the data."""
        data = self._read_data_from_file()
        if self.bpp == 10:
            data = dec10216(data)
        data = data.reshape((self.lines, self.cols))
        return data

    def _read_data_from_file(self):
        if self._is_file_like():
            return self._read_file_like()
        return self._read_data_from_disk()

    def _is_file_like(self):
        return isinstance(self.filename, FSFile)

    def _read_data_from_disk(self):
        # For reading the image data, unzip_context is faster than generic_open
        dtype, shape = self._get_input_info()
        with utils.unzip_context(self.filename) as fn:
            with decompressed(fn) if self.compressed else nullcontext(fn) as filename:
                return np.fromfile(filename,
                                   offset=self.offset,
                                   dtype=dtype,
                                   count=np.prod(shape))

    def _read_file_like(self):
        # filename is likely to be a file-like object, already in memory
        dtype, shape = self._get_input_info()
        with utils.generic_open(self.filename, mode="rb") as fp:
            no_elements = np.prod(shape)
            fp.seek(self.offset)
            return np.frombuffer(
                fp.read(np.dtype(dtype).itemsize * no_elements),
                dtype=np.dtype(dtype),
                count=no_elements.item()
            ).reshape(shape)

    def _get_input_info(self):
        total_bits = int(self.lines) * int(self.cols) * int(self.bpp)
        input_shape = int(np.ceil(total_bits / 8.))
        if self.bpp == 16:
            input_dtype = '>u2'
            input_shape //= 2
        elif self.bpp in [8, 10]:
            input_dtype = np.uint8
        else:
            raise ValueError(f"Unexpected number of bits per pixel: {self.bpp}")
        input_shape = (input_shape,)
        return input_dtype, input_shape
