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
"""Generic Reader for GRIB2 files.

Currently this reader depends on the `pygrib` python package. The `eccodes`
package from ECMWF is preferred, but does not support python 3 at the time
of writing.

"""
import logging
import numpy as np
import xarray as xr
import dask.array as da
from pyproj import Proj
from pyresample import geometry
from datetime import datetime

from satpy import DatasetID, CHUNK_SIZE
from satpy.readers.file_handlers import BaseFileHandler
import pygrib

LOG = logging.getLogger(__name__)


CF_UNITS = {
    'none': '1',
}


class GRIBFileHandler(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(GRIBFileHandler, self).__init__(filename, filename_info, filetype_info)

        self._msg_datasets = {}
        self._start_time = None
        self._end_time = None
        try:
            with pygrib.open(filename) as grib_file:
                for idx, msg in enumerate(grib_file):
                    msg_id = DatasetID(name=msg['shortName'],
                                       level=msg['level'])
                    start_time = datetime.strptime(
                        msg['dataDate'] + msg['dataTime'],
                        '%Y%m%d%H%M')
                    ds_info = {
                        'message': idx,
                        'filename': self.filename,
                        'name': msg['shortName'],
                        'long_name': msg['name'],
                        'level': msg['level'],
                        'pressureUnits': msg['pressureUnits'],
                        'standard_name': msg['cfName'],
                        'units': msg['units'],
                        'start_time': start_time,
                        'end_time': start_time,
                        'file_type': self.filetype_info['file_type'],
                    }
                    self._msg_datasets[msg_id] = ds_info

                    if self._start_time is None:
                        self._start_time = start_time
                    if idx == grib_file.messages - 1:
                        self._end_time = start_time
        except (RuntimeError, KeyError):
            raise IOError("Unknown GRIB file format: {}".format(filename))

    @property
    def start_time(self):
        """Get start time of this entire file.

        Assumes the first message is the earliest message.

        """
        return self._start_time

    @property
    def end_time(self):
        """Get end time of this entire file.

        Assumes the last message is the latest message.

        """
        return self._end_time

    def available_datasets(self):
        """Automatically determine datasets provided by this file"""
        return self._msg_datasets.items()

    def _area_def_from_msg(self, msg):
        proj_params = msg.projparams.copy()
        if proj_params['proj'] == 'cyl':
            proj_params['proj'] = 'eqc'
            proj = Proj(**proj_params)
            lons = msg['distinctLongitudes']
            lats = msg['distinctLatitudes']
            min_lon = lons[0]
            max_lon = lons[1]
            min_lat = lats[0]
            max_lat = lats[1]
            shape = (lats.shape[0], lons.shape[0])
            min_x, min_y = proj(min_lon, min_lat)
            max_x, max_y = proj(max_lon, max_lat)
            pixel_size_x = (max_x - min_x) / (shape[1] - 1)
            pixel_size_y = (max_y - min_y) / (shape[0] - 1)
            extents = (
                min_x - pixel_size_x / 2.,
                min_y - pixel_size_y / 2.,
                max_x + pixel_size_x / 2.,
                max_y + pixel_size_y / 2.,
            )
        else:
            lats, lons = msg.latlons()
            shape = lats.shape
            proj = Proj(**proj_params)
            min_x, min_y = proj(lons[-1, 0], lats[-1, 0])
            max_x, max_y = proj(lons[0, -1], lats[0, -1])
            extents = (min_x, min_y, max_x, max_y)

        return geometry.AreaDefinition(
            'on-the-fly grib area',
            'on-the-fly grib area',
            'on-the-fly grib area',
            proj_params,
            shape[1],
            shape[0],
            extents,
        )

    def get_area_def(self, dsid):
        """Get area definition for message.

        If latlong grid then convert to valid eqc grid.

        """
        msg_num = self._msg_datasets[dsid]['message']
        with pygrib.open(self.filename) as grib_file:
            msg = grib_file.message(msg_num)
            try:
                return self._area_def_from_msg(msg)
            except (RuntimeError, KeyError):
                raise RuntimeError("Unknown GRIB projection information")

    def get_dataset(self, dataset_id, ds_info):
        """Read a GRIB message into an xarray DataArray."""
        msg_num = self._msg_datasets[dataset_id]['message']
        with pygrib.open(self.filename) as grib_file:
            msg = grib_file.message(msg_num)
            fill = msg['missingValue']
            data = msg.values

        if isinstance(data, np.ma.MaskedArray):
            data = data.filled(np.nan)
            data = da.from_array(data, chunks=CHUNK_SIZE)
        else:
            data[data == fill] = np.nan
            data = da.from_array(data, chunks=CHUNK_SIZE)

        return xr.DataArray(data, attrs=ds_info, dims=('y', 'x'))
