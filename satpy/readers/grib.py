#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 Satpy developers
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
from datetime import datetime

import dask.array as da
import numpy as np
import pygrib
import xarray as xr
from pyproj import Proj
from pyresample import geometry

from satpy import CHUNK_SIZE
from satpy.dataset import DataQuery
from satpy.readers.file_handlers import BaseFileHandler

LOG = logging.getLogger(__name__)


CF_UNITS = {
    'none': '1',
}


class GRIBFileHandler(BaseFileHandler):
    """Generic GRIB file handler."""

    def __init__(self, filename, filename_info, filetype_info):
        """Open grib file and do initial message parsing."""
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
            msg_id = DataQuery(name=msg['shortName'],
                               level=msg['level'],
                               modifiers=tuple())
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
            msg_id = DataQuery(**ds_info)
            ds_info = msg_id.to_dict()
            ds_info.update(msg_info)
            ds_info['file_type'] = self.filetype_info['file_type']
            self._msg_datasets[msg_id] = ds_info

    @staticmethod
    def _convert_datetime(msg, date_key, time_key, date_format="%Y%m%d%H%M"):
        date_str = "{:d}{:04d}".format(msg[date_key], msg[time_key])
        return datetime.strptime(date_str, date_format)

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

    def available_datasets(self, configured_datasets=None):
        """Automatically determine datasets provided by this file."""
        # previously configured or provided datasets
        # we can't provide any additional information
        for is_avail, ds_info in (configured_datasets or []):
            yield is_avail, ds_info
        # new datasets
        for ds_info in self._msg_datasets.values():
            yield True, ds_info

    def _get_message(self, ds_info):
        with pygrib.open(self.filename) as grib_file:
            if 'message' in ds_info:
                msg_num = ds_info['message']
                msg = grib_file.message(msg_num)
            else:
                msg_keys = self.filetype_info['keys'].keys()
                msg = self._idx(**{k: ds_info[k] for k in msg_keys})[0]
            return msg

    @staticmethod
    def _correct_cyl_minmax_xy(proj_params, min_lon, min_lat, max_lon, max_lat):
        proj = Proj(**proj_params)
        min_x, min_y = proj(min_lon, min_lat)
        max_x, max_y = proj(max_lon, max_lat)
        if max_x <= min_x:
            # wrap around
            # make 180 longitude the prime meridian
            # assuming we are going from 0 to 360 longitude
            proj_params['pm'] = 180
            proj = Proj(**proj_params)
            # recompute x/y extents with this new projection
            min_x, min_y = proj(min_lon, min_lat)
            max_x, max_y = proj(max_lon, max_lat)
        return proj_params, (min_x, min_y, max_x, max_y)

    @staticmethod
    def _get_cyl_minmax_lonlat(lons, lats):
        min_lon = lons[0]
        max_lon = lons[-1]
        min_lat = lats[0]
        max_lat = lats[-1]
        if min_lat > max_lat:
            # lats aren't in the order we thought they were, flip them
            min_lat, max_lat = max_lat, min_lat
        return min_lon, min_lat, max_lon, max_lat

    def _get_cyl_area_info(self, msg, proj_params):
        proj_params['proj'] = 'eqc'
        lons = msg['distinctLongitudes']
        lats = msg['distinctLatitudes']
        shape = (lats.shape[0], lons.shape[0])
        minmax_lonlat = self._get_cyl_minmax_lonlat(lons, lats)
        proj_params, minmax_xy = self._correct_cyl_minmax_xy(proj_params, *minmax_lonlat)
        extents = self._get_extents(*minmax_xy, shape)
        return proj_params, shape, extents

    @staticmethod
    def _get_extents(min_x, min_y, max_x, max_y, shape):
        half_x = abs((max_x - min_x) / (shape[1] - 1)) / 2.
        half_y = abs((max_y - min_y) / (shape[0] - 1)) / 2.
        return min_x - half_x, min_y - half_y, max_x + half_x, max_y + half_y

    @staticmethod
    def _get_corner_xy(proj_params, lons, lats, scans_positively):
        proj = Proj(**proj_params)
        x, y = proj(lons, lats)
        if scans_positively:
            min_x, min_y = x[0], y[0]
            max_x, max_y = x[3], y[3]
        else:
            min_x, min_y = x[2], y[2]
            max_x, max_y = x[1], y[1]
        return min_x, min_y, max_x, max_y

    @staticmethod
    def _get_corner_lonlat(proj_params, lons, lats):
        # take the corner points only
        lons = lons[([0, 0, -1, -1], [0, -1, 0, -1])]
        lats = lats[([0, 0, -1, -1], [0, -1, 0, -1])]
        # if we have longitudes over 180, assume 0-360
        if (lons > 180).any():
            # make 180 longitude the prime meridian
            proj_params['pm'] = 180
        return proj_params, lons, lats

    def _get_area_info(self, msg, proj_params):
        lats, lons = msg.latlons()
        shape = lats.shape
        scans_positively = (msg.valid_key('jScansPositively') and
                            msg['jScansPositively'] == 1)
        proj_params, lons, lats = self._get_corner_lonlat(
            proj_params, lons, lats)
        minmax_xy = self._get_corner_xy(proj_params, lons, lats, scans_positively)
        extents = self._get_extents(*minmax_xy, shape)
        return proj_params, shape, extents

    @staticmethod
    def _correct_proj_params_over_prime_meridian(proj_params):
        # correct for longitudes over 180
        for lon_param in ['lon_0', 'lon_1', 'lon_2']:
            if proj_params.get(lon_param, 0) > 180:
                proj_params[lon_param] -= 360
        return proj_params

    def _area_def_from_msg(self, msg):
        proj_params = msg.projparams.copy()
        proj_params = self._correct_proj_params_over_prime_meridian(proj_params)

        if proj_params['proj'] in ('cyl', 'eqc'):
            # eqc projection that goes from 0 to 360
            proj_params, shape, extents = self._get_cyl_area_info(msg, proj_params)
        else:
            proj_params, shape, extents = self._get_area_info(msg, proj_params)

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
        """Get metadata."""
        model_time = self._convert_datetime(msg, 'dataDate',
                                            'dataTime')
        start_time = self._convert_datetime(msg, 'validityDate',
                                            'validityTime')
        end_time = start_time
        try:
            center_description = msg['centreDescription']
        except (RuntimeError, KeyError):
            center_description = None

        key_dicts = {
            'shortName': 'shortName',
            'long_name': 'name',
            'pressureUnits': 'pressureUnits',
            'typeOfLevel': 'typeOfLevel',
            'standard_name': 'cfName',
            'units': 'units',
            'modelName': 'modelName',
            'valid_min': 'minimum',
            'valid_max': 'maximum',
            'sensor': 'modelName'}

        ds_info.update({
            'filename': self.filename,
            'model_time': model_time,
            'centreDescription': center_description,
            'start_time': start_time,
            'end_time': end_time,
            'platform_name': 'unknown'})

        for key in key_dicts:
            if key_dicts[key] in msg.keys():
                ds_info[key] = msg[key_dicts[key]]
            else:
                ds_info[key] = 'unknown'

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
