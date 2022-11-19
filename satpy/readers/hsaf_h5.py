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
"""A reader for HDF5 Snow Cover (SC)  file produced by the Hydrology SAF.

"""
import logging
import os
from datetime import datetime

import dask.array as da
import h5py
import numpy as np
import xarray as xr

from satpy import CHUNK_SIZE
from satpy.readers.file_handlers import BaseFileHandler
from satpy.resample import get_area_def

LOG = logging.getLogger(__name__)

class HSAFFileHandler(BaseFileHandler):
    """File handler for HSAF H5 files."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init the file handler."""
        super(HSAFFileHandler, self).__init__(filename,
                                              filename_info,
                                              filetype_info)

        self._msg_datasets = {}
        self._start_time = None
        self._end_time = None
        analysis_time = self._get_datetime(filename)
        self._analysis_time = analysis_time

    @staticmethod
    def _get_datetime(filename):
        fname = os.path.basename(filename)
        dtstr = fname.split('_')[1].zfill(4)
        return datetime.strptime(dtstr, "%Y%m%d%H%M")

    @property
    def analysis_time(self):
        """Get validity time of this file."""
        return self._analysis_time

    def _prepare_variable_for_palette(self, msg, ds_info):
        colormap = np.array(np.array(msg))
        return xr.DataArray(colormap, attrs=ds_info, dims=('idx', 'RGB'))

    def get_metadata(self, msg, name):
        """Get the metadata."""
        if name == 'SC':
            ds_info = {
                'name': 'SC',
                'filename': self.filename,
                'data_time': self._analysis_time,
                'nx': msg.shape[1],
                'ny': msg.shape[0]
            }
        elif name == 'SC_pal':
            ds_info = {
                'name': 'SC_pal'
            }
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
        hsaf_def = fd_def[62:62+916, 1211:1211+1902]

        return hsaf_def

    def _get_message(self, idx):
        h5file = h5py.File(self.filename, 'r')
        if idx == 1:
            msg = h5file.get('SC')
        if idx == 2:
            msg = h5file.get('colormap')
        return msg

    def get_dataset(self, ds_id, ds_info):
        """Read a HDF5 file into an xarray DataArray."""
        variable = None
        if ds_id['name'] == 'SC':
            msg = self._get_message(1)
            ds_info = self.get_metadata(msg, ds_id['name'])

            fname = os.path.basename(msg.file.filename)
            dtstr = fname.split('_')[1].zfill(4)
            h10_time = datetime.strptime(dtstr, "%Y%m%d%H%M")

            ds_info['start_time'] = h10_time
            ds_info['end_time'] = h10_time

            data = np.array(msg)
            data = da.from_array(data, chunks=CHUNK_SIZE)
            variable = xr.DataArray(data, attrs=ds_info, dims=('y', 'x'))

        elif ds_id['name'] == 'SC_pal':
            msg = self._get_message(2)
            ds_info = self.get_metadata(msg, ds_id['name'])
            variable = self._prepare_variable_for_palette(msg, ds_info)

        else:
            raise IOError("File does not contain " + ds_id['name'] + " data")

        return variable
