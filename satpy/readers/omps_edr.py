#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2011-2015 Satpy developers
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
"""Interface to OMPS EDR format

"""
from datetime import datetime, timedelta
import numpy as np
import logging

from satpy.readers.hdf5_utils import HDF5FileHandler

NO_DATE = datetime(1958, 1, 1)
EPSILON_TIME = timedelta(days=2)
LOG = logging.getLogger(__name__)


class EDRFileHandler(HDF5FileHandler):
    _fill_name = "_FillValue"

    @property
    def start_orbit_number(self):
        return self.filename_info['orbit']

    @property
    def end_orbit_number(self):
        return self.filename_info['orbit']

    @property
    def platform_name(self):
        return self.filename_info['platform_shortname']

    @property
    def sensor_name(self):
        return self.filename_info['instrument_shortname']

    def get_shape(self, ds_id, ds_info):
        return self[ds_info['file_key'] + '/shape']

    def adjust_scaling_factors(self, factors, file_units, output_units):
        if factors is None or factors[0] is None:
            factors = [1, 0]
        if file_units == output_units:
            LOG.debug("File units and output units are the same (%s)", file_units)
            return factors
        return np.array(factors)

    def get_metadata(self, dataset_id, ds_info):
        var_path = ds_info.get('file_key', '{}'.format(dataset_id.name))
        info = getattr(self[var_path], 'attrs', {})
        info.update(ds_info)

        file_units = ds_info.get('file_units')
        if file_units is None:
            file_units = self.get(var_path + '/attr/units', self.get(var_path + '/attr/Units'))
        if file_units is None:
            raise KeyError("File variable '{}' has no units attribute".format(var_path))
        elif file_units == 'deg':
            file_units = 'degrees'
        elif file_units == 'Unitless':
            file_units = '1'

        info.update({
            "shape": self.get_shape(dataset_id, ds_info),
            "file_units": file_units,
            "units": ds_info.get("units", file_units),
            "platform_name": self.platform_name,
            "sensor": self.sensor_name,
            "start_orbit": self.start_orbit_number,
            "end_orbit": self.end_orbit_number,
        })
        info.update(dataset_id.to_dict())
        if 'standard_name' not in ds_info:
            info['standard_name'] = self.get(var_path + '/attr/Title', dataset_id.name)
        return info

    def get_dataset(self, dataset_id, ds_info):
        var_path = ds_info.get('file_key', '{}'.format(dataset_id.name))
        metadata = self.get_metadata(dataset_id, ds_info)
        valid_min, valid_max = self.get(var_path + '/attr/valid_range',
                                        self.get(var_path + '/attr/ValidRange', (None, None)))
        if valid_min is None or valid_max is None:
            raise KeyError("File variable '{}' has no valid range attribute".format(var_path))
        fill_name = var_path + '/attr/{}'.format(self._fill_name)
        if fill_name in self:
            fill_value = self[fill_name]
        else:
            fill_value = None

        data = self[var_path]
        scale_factor_path = var_path + '/attr/ScaleFactor'
        if scale_factor_path in self:
            scale_factor = self[scale_factor_path]
            scale_offset = self[var_path + '/attr/Offset']
        else:
            scale_factor = None
            scale_offset = None

        if valid_min is not None and valid_max is not None:
            # the original .cfg/INI based reader only checked valid_max
            data = data.where((data <= valid_max) & (data >= valid_min))
        if fill_value is not None:
            data = data.where(data != fill_value)

        factors = (scale_factor, scale_offset)
        factors = self.adjust_scaling_factors(factors, metadata['file_units'], ds_info.get("units"))
        if factors[0] != 1 or factors[1] != 0:
            data = data * factors[0] + factors[1]

        data.attrs.update(metadata)
        return data


class EDREOSFileHandler(EDRFileHandler):
    _fill_name = "MissingValue"
