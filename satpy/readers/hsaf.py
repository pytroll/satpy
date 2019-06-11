#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017.
#
# Author(s):
#
#   David Hoese <david.hoese@ssec.wisc.edu>
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
"""A reader for files produced by the Hydrology SAF

Currently this reader depends on the `pygrib` python package. The `eccodes`
package from ECMWF is preferred, but does not support python 3 at the time
of writing.

"""
import logging
import numpy as np
import xarray as xr
import dask.array as da
from pyresample import geometry
from datetime import datetime

from satpy import CHUNK_SIZE
from satpy.readers.file_handlers import BaseFileHandler
import pygrib

LOG = logging.getLogger(__name__)


CF_UNITS = {
    'none': '1',
}


class HSAFFileHandler(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(HSAFFileHandler, self).__init__(filename,
                                              filename_info,
                                              filetype_info)

        self._msg_datasets = {}
        self._start_time = None
        self._end_time = None
        try:
            with pygrib.open(self.filename) as grib_file:
                first_msg = grib_file.message(1)
                analysis_time = self._get_datetime(first_msg)
                self._analysis_time = analysis_time
                self.metadata = self.get_metadata(first_msg)

        except (RuntimeError, KeyError):
            raise IOError("Unknown GRIB file format: {}".format(self.filename))

    @staticmethod
    def _get_datetime(msg):
        dtstr = str(msg.dataDate) + str(msg.dataTime).zfill(4)
        return datetime.strptime(dtstr, "%Y%m%d%H%M")

    @property
    def analysis_time(self):
        """
        Get validity time of this file
        """
        return self._analysis_time

    def get_metadata(self, msg):
        try:
            center_description = msg.centreDescription
        except (RuntimeError, KeyError):
            center_description = None
        ds_info = {
            'filename': self.filename,
            'shortName': msg.shortName,
            'long_name': msg.name,
            'units': msg.units,
            'centreDescription': center_description,
            'data_time': self._analysis_time,
            'nx': msg.Nx,
            'ny': msg.Ny,
            'projparams': msg.projparams
        }
        return ds_info

    def get_area_def(self, dsid):
        """
        Get area definition for message.
        """
        msg = self._get_message(1)
        try:
            return self._get_area_def(msg)
        except (RuntimeError, KeyError):
            raise RuntimeError("Unknown GRIB projection information")

    def _get_area_def(self, msg):
        """
        Get the area definition of the datasets in the file.
        """

        proj_param = msg.projparams.copy()
        area_extent = 0

        Rx = 2 * np.arcsin(1. / msg.NrInRadiusOfEarth) / msg.dx
        Ry = 2 * np.arcsin(1. / msg.NrInRadiusOfEarth) / msg.dy

        max_x = ((msg.Nx / 2.0) * Rx) * proj_param['h']
        min_x = ((msg.Nx / 2.0) * Rx * -1.) * proj_param['h']
        max_y = ((msg.Ny / 2.0) * Ry) * proj_param['h']
        min_y = ((msg.Ny / 2.0) * Ry * -1.) * proj_param['h']

        area_extent = (min_x, min_y, max_x, max_y)

        area = geometry.AreaDefinition('hsaf_region',
                                       'A region from H-SAF',
                                       'geos',
                                       proj_param,
                                       msg.Nx,
                                       msg.Ny,
                                       area_extent)

        return area

    def _get_message(self, idx):
        with pygrib.open(self.filename) as grib_file:
            msg = grib_file[idx]
            return msg

    def get_dataset(self, ds_id, ds_info):
        """Read a GRIB message into an xarray DataArray."""
        if (ds_id.name not in self.filename):
            raise IOError("File does not contain {} data".format(ds_id.name))

        msg = self._get_message(1)

        ds_info = self.get_metadata(msg)
        fill = msg.missingValue
        data = msg.values.astype(np.float32)
        if msg.valid_key('jScansPositively') and msg['jScansPositively'] == 1:
            data = data[::-1]

        if isinstance(data, np.ma.MaskedArray):
            data = data.filled(np.nan)
            data = da.from_array(data, chunks=CHUNK_SIZE)
        else:
            data[data == fill] = np.nan
            data = da.from_array(data, chunks=CHUNK_SIZE)

        return xr.DataArray(data, attrs=ds_info, dims=('y', 'x'))
