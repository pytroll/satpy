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
# type: ignore
"""An OSISAF SST reader for the netCDF GHRSST format."""

import logging
from datetime import datetime

import numpy as np

from satpy.dataset import Dataset
from satpy.readers.netcdf_utils import NetCDF4FileHandler

logger = logging.getLogger(__name__)

PLATFORM_NAME = {'NPP': 'Suomi-NPP', }
SENSOR_NAME = {'VIIRS': 'viirs',
               'AVHRR': 'avhrr/3'}


class GHRSST_OSISAFL2(NetCDF4FileHandler):
    """Reader for the OSISAF SST GHRSST format."""

    def _parse_datetime(self, datestr):
        return datetime.strptime(datestr, '%Y%m%dT%H%M%SZ')

    def get_area_def(self, area_id, area_info):
        """Override abstract baseclass method."""
        raise NotImplementedError

    def get_dataset(self, dataset_id, ds_info, out=None):
        """Load a dataset."""
        var_path = ds_info.get('file_key', '{}'.format(dataset_id['name']))
        dtype = ds_info.get('dtype', np.float32)
        if var_path + '/shape' not in self:
            # loading a scalar value
            shape = 1
        else:
            shape = self[var_path + '/shape']
            if shape[0] == 1:
                # Remove the time dimenstion from dataset
                shape = shape[1], shape[2]

        file_units = ds_info.get('file_units')
        if file_units is None:
            try:
                file_units = self[var_path + '/attr/units']
                # they were almost completely CF compliant...
                if file_units == "none":
                    file_units = "1"
            except KeyError:
                # no file units specified
                file_units = None

        if out is None:
            out = np.ma.empty(shape, dtype=dtype)
            out.mask = np.zeros(shape, dtype=bool)

        out.data[:] = np.require(self[var_path][0][::-1], dtype=dtype)
        self._scale_and_mask_data(out, var_path)

        ds_info.update({
            "units": ds_info.get("units", file_units),
            "platform_name": PLATFORM_NAME.get(self['/attr/platform'], self['/attr/platform']),
            "sensor": SENSOR_NAME.get(self['/attr/sensor'], self['/attr/sensor']),
        })
        ds_info.update(dataset_id.to_dict())
        cls = ds_info.pop("container", Dataset)
        return cls(out, **ds_info)

    def _scale_and_mask_data(self, out, var_path):
        valid_min = self[var_path + '/attr/valid_min']
        valid_max = self[var_path + '/attr/valid_max']
        try:
            scale_factor = self[var_path + '/attr/scale_factor']
            scale_offset = self[var_path + '/attr/add_offset']
        except KeyError:
            scale_factor = scale_offset = None
        if valid_min is not None and valid_max is not None:
            out.mask[:] |= (out.data < valid_min) | (out.data > valid_max)
        factors = (scale_factor, scale_offset)
        if factors[0] != 1 or factors[1] != 0:
            out.data[:] *= factors[0]
            out.data[:] += factors[1]

    def get_lonlats(self, navid, nav_info, lon_out=None, lat_out=None):
        """Load an area."""
        lon_key = 'lon'
        valid_min = self[lon_key + '/attr/valid_min']
        valid_max = self[lon_key + '/attr/valid_max']

        lon_out.data[:] = self[lon_key][::-1]
        lon_out.mask[:] = (lon_out < valid_min) | (lon_out > valid_max)

        lat_key = 'lat'
        valid_min = self[lat_key + '/attr/valid_min']
        valid_max = self[lat_key + '/attr/valid_max']
        lat_out.data[:] = self[lat_key][::-1]
        lat_out.mask[:] = (lat_out < valid_min) | (lat_out > valid_max)

        return {}

    @property
    def start_time(self):
        """Get start time."""
        # return self.filename_info['start_time']
        return self._parse_datetime(self['/attr/start_time'])

    @property
    def end_time(self):
        """Get end time."""
        return self._parse_datetime(self['/attr/stop_time'])
