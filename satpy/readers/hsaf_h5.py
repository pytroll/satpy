#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019.
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
"""A reader for HDF5 Snow Cover (SC) file produced by the Hydrology SAF."""
import logging
from datetime import timedelta

import dask.array as da
import h5py
import numpy as np
import xarray as xr

from satpy import CHUNK_SIZE
from satpy.readers.file_handlers import BaseFileHandler
from satpy.resample import get_area_def

LOG = logging.getLogger(__name__)
AREA_X_OFFSET = 1211
AREA_Y_OFFSET = 62


class HSAFFileHandler(BaseFileHandler):
    """File handler for HSAF H5 files."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init the file handler."""
        super(HSAFFileHandler, self).__init__(filename,
                                              filename_info,
                                              filetype_info)
        self._h5fh = h5py.File(self.filename, 'r')

    @property
    def end_time(self):
        """Get end time."""
        return self.start_time + timedelta(hours=23, minutes=59, seconds=59)

    @property
    def start_time(self):
        """Get start time."""
        return self.filename_info['sensing_time']

    def _prepare_variable_for_palette(self, dset, ds_info):
        colormap = np.array(dset)
        return xr.DataArray(colormap, attrs=ds_info, dims=('idx', 'RGB'))

    def get_metadata(self, dset, name):
        """Get the metadata."""
        ds_info = {'name': name}
        if name == 'SC':
            ds_info.update({
                'filename': self.filename,
                'data_time': self.start_time,
                'nx': dset.shape[1],
                'ny': dset.shape[0]
            })
        return ds_info

    def get_area_def(self, dsid):
        """Area definition for h10 SC dataset.

        Since it is not available in the HDF5 message,
        using hardcoded one (it's known).
        """
        if dsid['name'] == 'SC':
            return self._get_area_def()
        raise NotImplementedError

    def _get_area_def(self):
        """Area definition for h10 - hardcoded.

        Area definition not available in the HDF5 message,
        so using hardcoded one (it's known).

        hsaf_h10:
          description: H SAF H10 area definition
          projection:
            proj: geos
            lon_0: 0
            h: 35785831
            x_0: 0
            y_0: 0
            a: 6378169
            rf: 295.488065897001
            no_defs: null
            type: crs
          shape:
            height: 916
            width: 1902
          area_extent:
            lower_left_xy: [-1936760.3163240477, 2635854.280233425]
            upper_right_xy: [3770006.7195370505, 5384223.683413638]
            units: m
        """
        fd_def = get_area_def('msg_seviri_fes_3km')
        hsaf_def = fd_def[AREA_Y_OFFSET:AREA_Y_OFFSET+916,
                          AREA_X_OFFSET:AREA_X_OFFSET+1902]

        return hsaf_def

    def _get_dataset(self, ds_name):
        if ds_name == 'SC_pal':
            _ds_name = 'colormap'
        else:
            _ds_name = ds_name
        return self._h5fh.get(_ds_name)

    def get_dataset(self, ds_id, ds_info):
        """Read a HDF5 file into an xarray DataArray."""
        ds = self._get_dataset(ds_id['name'])
        ds_info = self.get_metadata(ds, ds_id['name'])

        if ds_id['name'] == 'SC':
            ds_info['start_time'] = self.start_time
            ds_info['data_time'] = self.start_time
            ds_info['end_time'] = self.end_time

            data = da.from_array(ds, chunks=CHUNK_SIZE)
            return xr.DataArray(data, attrs=ds_info, dims=('y', 'x'))

        elif ds_id['name'] == 'SC_pal':
            return self._prepare_variable_for_palette(ds, ds_info)
