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
            with pygrib.open(self.filename) as grib_file:
                first_msg = grib_file.message(1)
                last_msg = grib_file.message(grib_file.messages)
                start_time = self._convert_datetime(
                    first_msg, 'validityDate', 'validityTime')
                end_time = self._convert_datetime(
                    last_msg, 'validityDate', 'validityTime')
                self._start_time = start_time
                self._end_time = end_time
                if 'keys' not in filetype_info:
                    self._analyze_messages(grib_file)
                    self._idx = None
                else:
                    self._create_dataset_ids(filetype_info['keys'])
                    self._idx = pygrib.index(self.filename,
                                             *filetype_info['keys'].keys())
        except (RuntimeError, KeyError):
            raise IOError("Unknown GRIB file format: {}".format(self.filename))

    def _analyze_messages(self, grib_file):
        grib_file.seek(0)
        for idx, msg in enumerate(grib_file):
            msg_id = DatasetID(name=msg['shortName'],
                               level=msg['level'])
            ds_info = {
                'message': idx + 1,
                'name': msg['shortName'],
                'level': msg['level'],
                'file_type': self.filetype_info['file_type'],
            }
            self._msg_datasets[msg_id] = ds_info

    def _create_dataset_ids(self, keys):
        from itertools import product
        ordered_keys = [k for k in keys.keys() if 'id_key' in keys[k]]
        for id_vals in product(*[keys[k]['values'] for k in ordered_keys]):
            id_keys = [keys[k]['id_key'] for k in ordered_keys]
            msg_info = dict(zip(ordered_keys, id_vals))
            ds_info = dict(zip(id_keys, id_vals))
            msg_id = DatasetID(**ds_info)
            ds_info = msg_id.to_dict()
            ds_info.update(msg_info)
            ds_info['file_type'] = self.filetype_info['file_type']
            self._msg_datasets[msg_id] = ds_info

    @staticmethod
    def _convert_datetime(msg, date_key, time_key, format="%Y%m%d%H%M"):
        date_str = "{:d}{:04d}".format(msg[date_key], msg[time_key])
        return datetime.strptime(date_str, format)

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

    def _get_message(self, ds_info):
        with pygrib.open(self.filename) as grib_file:
            if 'message' in ds_info:
                msg_num = ds_info['message']
                msg = grib_file.message(msg_num)
            else:
                msg_keys = self.filetype_info['keys'].keys()
                msg = self._idx(**{k: ds_info[k] for k in msg_keys})[0]
            return msg

    def _area_def_from_msg(self, msg):
        proj_params = msg.projparams.copy()
        # correct for longitudes over 180
        for lon_param in ['lon_0', 'lon_1', 'lon_2']:
            if proj_params.get(lon_param, 0) > 180:
                proj_params[lon_param] -= 360

        if proj_params['proj'] == 'cyl':
            proj_params['proj'] = 'eqc'
            proj = Proj(**proj_params)
            lons = msg['distinctLongitudes']
            lats = msg['distinctLatitudes']
            min_lon = lons[0]
            max_lon = lons[-1]
            min_lat = lats[0]
            max_lat = lats[-1]
            if min_lat > max_lat:
                # lats aren't in the order we thought they were, flip them
                # we also need to flip the data in the data loading section
                min_lat, max_lat = max_lat, min_lat
            shape = (lats.shape[0], lons.shape[0])
            min_x, min_y = proj(min_lon, min_lat)
            max_x, max_y = proj(max_lon, max_lat)
            if max_x < min_x and 'over' not in proj_params:
                # wrap around
                proj_params['over'] = True
                proj = Proj(**proj_params)
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
            # take the corner points only
            lons = lons[([0, 0, -1, -1], [0, -1, 0, -1])]
            lats = lats[([0, 0, -1, -1], [0, -1, 0, -1])]
            # correct for longitudes over 180
            lons[lons > 180] -= 360

            proj = Proj(**proj_params)
            x, y = proj(lons, lats)
            if msg.valid_key('jScansPositively') and msg['jScansPositively'] == 1:
                min_x, min_y = x[0], y[0]
                max_x, max_y = x[3], y[3]
            else:
                min_x, min_y = x[2], y[2]
                max_x, max_y = x[1], y[1]
            half_x = abs((max_x - min_x) / (shape[1] - 1)) / 2.
            half_y = abs((max_y - min_y) / (shape[0] - 1)) / 2.
            extents = (min_x - half_x, min_y - half_y, max_x + half_x, max_y + half_y)

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
        msg = self._get_message(self._msg_datasets[dsid])
        try:
            return self._area_def_from_msg(msg)
        except (RuntimeError, KeyError):
            raise RuntimeError("Unknown GRIB projection information")

    def get_metadata(self, msg, ds_info):
        model_time = self._convert_datetime(msg, 'dataDate',
                                            'dataTime')
        start_time = self._convert_datetime(msg, 'validityDate',
                                            'validityTime')
        end_time = start_time
        try:
            center_description = msg['centreDescription']
        except (RuntimeError, KeyError):
            center_description = None
        ds_info.update({
            'filename': self.filename,
            'shortName': msg['shortName'],
            'long_name': msg['name'],
            'pressureUnits': msg['pressureUnits'],
            'typeOfLevel': msg['typeOfLevel'],
            'standard_name': msg['cfName'],
            'units': msg['units'],
            'modelName': msg['modelName'],
            'model_time': model_time,
            'centreDescription': center_description,
            'valid_min': msg['minimum'],
            'valid_max': msg['maximum'],
            'start_time': start_time,
            'end_time': end_time,
            'sensor': msg['modelName'],
            # National Weather Prediction
            'platform_name': 'unknown',
        })
        return ds_info

    def get_dataset(self, dataset_id, ds_info):
        """Read a GRIB message into an xarray DataArray."""
        msg = self._get_message(ds_info)
        ds_info = self.get_metadata(msg, ds_info)
        fill = msg['missingValue']
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
