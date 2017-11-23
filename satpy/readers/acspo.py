#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017.
#
# Author(s):
#
#
#   David Hoese <david.hoese@ssec.wisc.edu>
#
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
"""ACSPO SST Reader

See the following page for more information:

https://podaac.jpl.nasa.gov/dataset/VIIRS_NPP-OSPO-L2P-v2.3

"""
import logging
from datetime import datetime, timedelta

import numpy as np

from satpy.readers.netcdf_utils import NetCDF4FileHandler
from satpy.dataset import Dataset

LOG = logging.getLogger(__name__)


ROWS_PER_SCAN = {
    'MODIS': 10,
    'VIIRS': 16,
    'AVHRR': None,
}


class ACSPOFileHandler(NetCDF4FileHandler):
    """ACSPO L2P SST File Reader"""
    @property
    def platform_name(self):
        res = self['/attr/platform']
        if isinstance(res, np.ndarray):
            return str(res.astype(str))
        else:
            return res

    @property
    def sensor_name(self):
        res = self['/attr/sensor']
        if isinstance(res, np.ndarray):
            return str(res.astype(str))
        else:
            return res

    def get_shape(self, ds_id, ds_info):
        """Get numpy array shape for the specified dataset.
        
        Args:
            ds_id (DatasetID): ID of dataset that will be loaded
            ds_info (dict): Dictionary of dataset information from config file
            
        Returns:
            tuple: (rows, cols)
            
        """
        var_path = ds_info.get('file_key', '{}'.format(ds_id.name))
        s = self[var_path + "/shape"]
        if len(s) == 3:
            # time is first dimension of 1
            s = s[1:]
        return s

    @staticmethod
    def _parse_datetime(datestr):
        return datetime.strptime(datestr, "%Y%m%dT%H%M%SZ")

    @property
    def start_time(self):
        return self._parse_datetime(self['/attr/time_coverage_start'])

    @property
    def end_time(self):
        return self._parse_datetime(self['/attr/time_coverage_end'])

    def get_dataset(self, dataset_id, ds_info, out=None, xslice=slice(None), yslice=slice(None)):
        """Load data array and metadata from file on disk."""
        var_path = ds_info.get('file_key', '{}'.format(dataset_id.name))
        dtype = ds_info.get('dtype', np.float32)
        cls = ds_info.pop("container", Dataset)
        if var_path + '/shape' not in self:
            # loading a scalar value
            shape = 1
        else:
            file_shape = shape = self[var_path + '/shape']
            if len(shape) == 3:
                if shape[0] != 1:
                    raise ValueError("Not sure how to load 3D Dataset with more than 1 time")
                else:
                    shape = shape[1:]

        if isinstance(shape, tuple) and len(shape) == 2:
            # 2D array
            if xslice.start is not None:
                shape = (shape[0], xslice.stop - xslice.start)
            if yslice.start is not None:
                shape = (yslice.stop - yslice.start, shape[1])
        elif isinstance(shape, tuple) and len(shape) == 1 and yslice.start is not None:
            shape = ((yslice.stop - yslice.start) / yslice.step,)

        if out is None:
            out = cls(np.ma.empty(shape, dtype=dtype),
                      mask=np.zeros(shape, dtype=np.bool),
                      copy=False)

        valid_min = self[var_path + '/attr/valid_min']
        valid_max = self[var_path + '/attr/valid_max']
        # no need to check fill value since we are using valid min/max
        scale_factor = self.get(var_path + '/attr/scale_factor')
        add_offset = self.get(var_path + '/attr/add_offset')

        if isinstance(file_shape, tuple) and len(file_shape) == 3:
            out.data[:] = np.require(self[var_path][0, yslice, xslice], dtype=dtype)
        elif isinstance(file_shape, tuple) and len(file_shape) == 2:
            out.data[:] = np.require(self[var_path][yslice, xslice], dtype=dtype)
        elif isinstance(file_shape, tuple) and len(file_shape) == 1:
            out.data[:] = np.require(self[var_path][yslice], dtype=dtype)
        else:
            out.data[:] = np.require(self[var_path][:], dtype=dtype)
        out.mask[:] = (out.data < valid_min) | (out.data > valid_max)
        if scale_factor is not None:
            out.data[:] *= scale_factor
            out.data[:] += add_offset

        if ds_info.get('cloud_clear', False):
            # clear-sky if bit 15-16 are 00
            clear_sky_mask = (self['l2p_flags'][0, :, :] & 0b1100000000000000) != 0
            out.mask[:] |= clear_sky_mask

        units = self[var_path + '/attr/units']
        standard_name = self[var_path + '/attr/standard_name']
        resolution = float(self['/attr/spatial_resolution'].split(' ')[0])
        rows_per_scan = ROWS_PER_SCAN.get(self.sensor_name) or 0
        out.info.update(dataset_id.to_dict())
        out.info.update({
            'units': units,
            'platform': self.platform_name,
            'sensor': self.sensor_name,
            'standard_name': standard_name,
            'resolution': resolution,
            'rows_per_scan': rows_per_scan,
            'long_name': self.get(var_path + '/attr/long_name'),
            'comment': self.get(var_path + '/attr/comment'),
        })
        return out
