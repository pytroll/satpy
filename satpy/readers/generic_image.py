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
If nodata values are present (and rasterio is able to read them), it
will be preserved as attribute ``_FillValue`` in the returned dataset.
In case that nodata values should be used to mask pixels (that have
equal values) with np.nan, it has to be enabled in the reader yaml
file (key ``nodata_handling`` per dataset with value ``"nan_mask"``).
"""

import logging

import dask.array as da
import numpy as np
import rasterio
import xarray as xr
from pyresample import utils

from satpy import CHUNK_SIZE
from satpy.readers.file_handlers import BaseFileHandler

BANDS = {1: ['L'],
         2: ['L', 'A'],
         3: ['R', 'G', 'B'],
         4: ['R', 'G', 'B', 'A']}

NODATA_HANDLING_FILLVALUE = 'fill_value'
NODATA_HANDLING_NANMASK = 'nan_mask'

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
        self.dataset_name = None
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

        data.attrs = attrs
        self.dataset_name = 'image'
        self.file_content[self.dataset_name] = data

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
        ds_name = self.dataset_name if self.dataset_name else key['name']
        logger.debug("Reading '%s.'", ds_name)
        data = self.file_content[ds_name]

        # Mask data if necessary
        try:
            data = _mask_image_data(data, info)
        except ValueError as err:
            logger.warning(err)

        data.attrs.update(key.to_dict())
        data.attrs.update(info)
        return data


def _mask_image_data(data, info):
    """Mask image data if necessary.

    Masking is done if alpha channel is present or
    dataset 'nodata_handling' is set to 'nan_mask'.
    In the latter case even integer data is converted
    to float32 and masked with np.nan.
    """
    if data.bands.size in (2, 4):
        if not np.issubdtype(data.dtype, np.integer):
            raise ValueError("Only integer datatypes can be used as a mask.")
        mask = data.data[-1, :, :] == np.iinfo(data.dtype).min
        data = data.astype(np.float64)
        masked_data = da.stack([da.where(mask, np.nan, data.data[i, :, :])
                                for i in range(data.shape[0])])
        data.data = masked_data
        data = data.sel(bands=BANDS[data.bands.size - 1])
    elif hasattr(data, 'nodatavals') and data.nodatavals:
        data = _handle_nodatavals(data, info.get('nodata_handling', NODATA_HANDLING_FILLVALUE))
    return data


def _handle_nodatavals(data, nodata_handling):
    """Mask data with np.nan or only set 'attr_FillValue'."""
    if nodata_handling == NODATA_HANDLING_NANMASK:
        # data converted to float and masked with np.nan
        data = data.astype(np.float32)
        masked_data = da.stack([da.where(data.data[i, :, :] == nodataval, np.nan, data.data[i, :, :])
                                for i, nodataval in enumerate(data.nodatavals)])
        data.data = masked_data
        data.attrs['_FillValue'] = np.nan
    elif nodata_handling == NODATA_HANDLING_FILLVALUE:
        # keep data as it is but set _FillValue attribute to provided
        # nodatavalue (first one as it has to be the same for all bands at least
        # in GeoTiff, see GDAL gtiff driver documentation)
        fill_value = data.nodatavals[0]
        if np.issubdtype(data.dtype, np.integer):
            fill_value = int(fill_value)
        data.attrs['_FillValue'] = fill_value
    return data
