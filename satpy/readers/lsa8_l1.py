#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016-2019 Satpy developers
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
"""Landsat 8 L1 reader.

TODO:
  - Finish converting yaml file from MSI to OLI/TIRS
  - Factorize metadata reading with MODIS readers
  - Quality flags
  - Angles
  - Tests
  - DataArrays' attributes
  - Read directly from tar file

References:
      - *Level 1 Product Formatting*
      https://prd-wret.s3-us-west-2.amazonaws.com/assets/palladium/production/atoms/files/LSDS-1574_L8_Data_Users_Handbook_v4.0.pdf

"""

import logging
from datetime import datetime
import ast

import numpy as np
import rasterio
from rasterio.windows import Window
import dask.array as da
from xarray import DataArray
from dask.base import tokenize
from threading import Lock

from satpy.readers.file_handlers import BaseFileHandler
from satpy import CHUNK_SIZE

logger = logging.getLogger(__name__)


class LSA8MetaReader(BaseFileHandler):
    """Metadata file reader.

    This uses the EOS metadata format.
    """

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the reader."""
        super().__init__(filename, filename_info, filetype_info)
        with open(filename, 'r') as fd:
            self.mda = self.read_mda(fd.read())

        date = self.mda['L1_METADATA_FILE']['PRODUCT_METADATA']['DATE_ACQUIRED']
        time = self.mda['L1_METADATA_FILE']['PRODUCT_METADATA']['SCENE_CENTER_TIME']

        self._start_time = datetime.strptime(' '.join((date, time)), '%Y-%m-%d %H:%M:%S.%f0Z')
        self._end_time = self._start_time

    @property
    def start_time(self):
        """Get the start time."""
        return self._start_time

    @property
    def end_time(self):
        """Get the end time."""
        return self._end_time

    @staticmethod
    def read_mda(attribute):
        """Read the EOS metadata."""
        lines = attribute.split('\n')
        mda = {}
        current_dict = mda
        path = []
        prev_line = None
        for line in lines:
            if not line:
                continue
            if line == 'END':
                break
            if prev_line:
                line = prev_line + line
            key, val = line.split('=')
            key = key.strip()
            val = val.strip()
            try:
                val = ast.literal_eval(val)
            except ValueError:
                pass
            except SyntaxError as err:
                if "EOL" in err.text:
                    prev_line = line
                    continue
            prev_line = None
            if key in ['GROUP', 'OBJECT']:
                new_dict = {}
                path.append(val)
                current_dict[val] = new_dict
                current_dict = new_dict
            elif key in ['END_GROUP', 'END_OBJECT']:
                if val != path[-1]:
                    raise SyntaxError
                path = path[:-1]
                current_dict = mda
                for item in path:
                    current_dict = current_dict[item]
            elif key in ['CLASS', 'NUM_VAL']:
                pass
            else:
                current_dict[key] = val
        return mda


class LSA8BandReader(BaseFileHandler):
    """Band file reader.

    The band files are in geotiff format and read using rasterio. For
    performance reasons, the reading adapts the chunk size to match the file's
    block size.
    """

    def __init__(self, filename, filename_info, filetype_info, mda_fh):
        """Initialize the reader."""
        super().__init__(filename, filename_info, filetype_info)

        self.mda = mda_fh.mda

        date = self.mda['L1_METADATA_FILE']['PRODUCT_METADATA']['DATE_ACQUIRED']
        time = self.mda['L1_METADATA_FILE']['PRODUCT_METADATA']['SCENE_CENTER_TIME']

        self._start_time = datetime.strptime(' '.join((date, time)), '%Y-%m-%d %H:%M:%S.%f0Z')
        self._end_time = self._start_time

        self.lats = None
        self.lons = None
        self.alts = None

        self.channel = filename_info['channel']

        self.read_lock = Lock()

        self.filehandle = rasterio.open(self.filename, 'r', sharing=False)

    def get_dataset(self, key, info):
        """Load a dataset."""
        if self.channel != key.name:
            return

        logger.debug('Reading %s.', key.name)

        data = self.read_band()
        band = key.name[1:]
        minmax = self.mda['L1_METADATA_FILE']['MIN_MAX_PIXEL_VALUE']

        data = data.where(data >= minmax['QUANTIZE_CAL_MIN_BAND_' + band])
        data = data.where(data <= minmax['QUANTIZE_CAL_MAX_BAND_' + band])

        coeffs = self.mda['L1_METADATA_FILE']['RADIOMETRIC_RESCALING']
        consts = self.mda['L1_METADATA_FILE']['TIRS_THERMAL_CONSTANTS']
        if key.calibration == 'counts':
            pass
        elif key.calibration == 'radiance':
            scale = coeffs['RADIANCE_MULT_BAND_' + band]
            offset = coeffs['RADIANCE_ADD_BAND_' + band]
            data = data * scale + offset
            data.attrs['unit'] = 'W/(m2 * sr * μm)'
        else:
            if band in ['10', '11']:
                scale = coeffs['RADIANCE_MULT_BAND_' + band]
                offset = coeffs['RADIANCE_ADD_BAND_' + band]
                data = data * scale + offset
                k1 = consts['K1_CONSTANT_BAND_' + band]
                k2 = consts['K2_CONSTANT_BAND_' + band]
                data = k2 / (np.log(k1 / data + 1))
                data.attrs['unit'] = 'K'
            else:
                scale = coeffs['REFLECTANCE_MULT_BAND_' + band]
                offset = coeffs['REFLECTANCE_ADD_BAND_' + band]
                data = (data * scale + offset) * 100
                data.attrs['unit'] = '%'

        return data

    def read_band_blocks(self, blocksize=CHUNK_SIZE):
        """Read the band in native blocks."""
        # For sentinel 1 data, the block are 1 line, and dask seems to choke on that.
        band = self.filehandle

        shape = band.shape
        token = tokenize(blocksize, band)
        name = 'read_band-' + token
        dskx = dict()
        if len(band.block_shapes) != 1:
            raise NotImplementedError('Bands with multiple shapes not supported.')
        else:
            chunks = band.block_shapes[0]

        def do_read(the_band, the_window, the_lock):
            with the_lock:
                return the_band.read(1, None, window=the_window)

        for ji, window in band.block_windows(1):
            dskx[(name, ) + ji] = (do_read, band, window, self.read_lock)

        res = da.Array(dskx, name, shape=list(shape),
                       chunks=chunks,
                       dtype=band.dtypes[0])
        return DataArray(res, dims=('y', 'x'))

    def read_band(self, blocksize=CHUNK_SIZE):
        """Read the band in chunks."""
        band = self.filehandle

        shape = band.shape
        if len(band.block_shapes) == 1:
            total_size = blocksize * blocksize * 1.0
            lines, cols = band.block_shapes[0]
            if cols > lines:
                hblocks = cols
                vblocks = int(total_size / cols / lines)
            else:
                hblocks = int(total_size / cols / lines)
                vblocks = lines
        else:
            hblocks = blocksize
            vblocks = blocksize
        vchunks = range(0, shape[0], vblocks)
        hchunks = range(0, shape[1], hblocks)

        token = tokenize(hblocks, vblocks, band)
        name = 'read_band-' + token

        def do_read(the_band, the_window, the_lock):
            with the_lock:
                return the_band.read(1, None, window=the_window)

        dskx = {(name, i, j): (do_read, band,
                               Window(hcs, vcs,
                                      min(hblocks,  shape[1] - hcs),
                                      min(vblocks,  shape[0] - vcs)),
                               self.read_lock)
                for i, vcs in enumerate(vchunks)
                for j, hcs in enumerate(hchunks)
                }

        res = da.Array(dskx, name, shape=list(shape),
                       chunks=(vblocks, hblocks),
                       dtype=band.dtypes[0])
        return DataArray(res, dims=('y', 'x'))

    def get_area_def(self, dsid):
        """Get the area definition of the dataarray."""
        if self.channel != dsid.name:
            return
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        crs = self.filehandle.crs
        pcs_id = crs.to_string()
        x_size = self.filehandle.width
        y_size = self.filehandle.height
        area_extent = list(self.filehandle.bounds)
        area_def = AreaDefinition('geotiff_area', pcs_id, pcs_id,
                                  proj4_str_to_dict(crs.to_proj4()),
                                  x_size, y_size, area_extent)
        return area_def

    @property
    def start_time(self):
        """Get the start time."""
        return self._start_time

    @property
    def end_time(self):
        """Get the end time."""
        return self._end_time
