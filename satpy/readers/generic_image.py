#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2019 Satpy developers
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
"""
Reader for generic image (e.g. gif, png, jpg, tif, geotiff, ...).

Returns a dataset without calibration.  Includes coordinates if
available in the file (eg. geotiff).
"""

import logging
import rasterio

import xarray as xr
import dask.array as da
import numpy as np

from satpy.readers.file_handlers import BaseFileHandler
from satpy import CHUNK_SIZE
from pyresample import utils

BANDS = {1: ['L'],
         2: ['L', 'A'],
         3: ['R', 'G', 'B'],
         4: ['R', 'G', 'B', 'A']}

logger = logging.getLogger(__name__)


class GenericImageFileHandler(BaseFileHandler):
    """Handle reading of generic image files."""

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize filehandler."""
        super(GenericImageFileHandler, self).__init__(
            filename, filename_info, filetype_info)
        self.finfo = filename_info
        try:
            self.finfo['end_time'] = self.finfo['start_time']
        except KeyError:
            pass
        self.finfo['filename'] = self.filename
        self.file_content = {}
        self.area = None
        self.read()

    def read(self):
        """Read the image."""
        dataset = rasterio.open(self.finfo['filename'])

        # Create area definition
        if hasattr(dataset, 'crs') and dataset.crs is not None:
            self.area = utils.get_area_def_from_raster(dataset)

        data = xr.open_rasterio(dataset, chunks=(1, CHUNK_SIZE, CHUNK_SIZE))
        attrs = data.attrs.copy()

        # Rename to Satpy convention
        data = data.rename({'band': 'bands'})

        # Rename bands to [R, G, B, A], or a subset of those
        data['bands'] = BANDS[data.bands.size]

        # Mask data if alpha channel is present
        try:
            data = mask_image_data(data)
        except ValueError as err:
            logger.warning(err)

        data.attrs = attrs
        self.file_content['image'] = data

    def get_area_def(self, dsid):
        """Get area definition of the image."""
        if self.area is None:
            raise NotImplementedError("No CRS information available from image")
        return self.area

    @property
    def start_time(self):
        """Return start time."""
        return self.finfo['start_time']

    @property
    def end_time(self):
        """Return end time."""
        return self.finfo['end_time']

    def get_dataset(self, key, info):
        """Get a dataset from the file."""
        logger.debug("Reading %s.", key)
        return self.file_content[key.name]


def mask_image_data(data):
    """Mask image data if alpha channel is present."""
    if data.bands.size in (2, 4):
        if not np.issubdtype(data.dtype, np.integer):
            raise ValueError("Only integer datatypes can be used as a mask.")
        mask = data.data[-1, :, :] == np.iinfo(data.dtype).min
        data = data.astype(np.float64)
        masked_data = da.stack([da.where(mask, np.nan, data.data[i, :, :])
                                for i in range(data.shape[0])])
        data.data = masked_data
        data = data.sel(bands=BANDS[data.bands.size - 1])
    return data
