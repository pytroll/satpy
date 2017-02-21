#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2011, 2012, 2013, 2014, 2015.

# Author(s):

#
#   Adam Dybbroe <adam.dybbroe@smhi.se>
#   Kristian Rune Larsen <krl@dmi.dk>
#   Lars Ørum Rasmussen <ras@dmi.dk>
#   Martin Raspaud <martin.raspaud@smhi.se>
#   David Hoese <david.hoese@ssec.wisc.edu>
#

# This file is part of satpy.

# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.

"""Interface to VIIRS SDR format

Format documentation:
http://npp.gsfc.nasa.gov/science/sciencedocuments/082012/474-00001-03_CDFCBVolIII_RevC.pdf

"""
from datetime import datetime, timedelta
import numpy as np
import logging

from satpy.readers.hdf5_utils import HDF5FileHandler
from satpy.dataset import Dataset

NO_DATE = datetime(1958, 1, 1)
EPSILON_TIME = timedelta(days=2)
LOG = logging.getLogger(__name__)


class EDRFileHandler(HDF5FileHandler):
    _fill_name = "_FillValue"

    @property
    def start_time(self):
        return self.filename_info['start_time']

    @property
    def end_time(self):
        return self.filename_info['end_time']

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

    def get_file_units(self, item):
        # What units should we expect from the file
        unit_attr_name = self.file_keys[item].kwargs.get("units_attr")
        if unit_attr_name is not None:
            return self[item + "/attr/{}".format(unit_attr_name)]
        return None

        return file_units

    def get_shape(self, ds_id, ds_info):
        return self[ds_info['file_key'] + '/shape']

    def scale_swath_data(self, data, mask, scaling_factors):
        """Scale swath data using scaling factors and offsets.

        Multi-granule (a.k.a. aggregated) files will have more than the usual two values.
        """
        num_grans = len(scaling_factors)//2
        gran_size = data.shape[0]//num_grans
        for i in range(num_grans):
            start_idx = i * gran_size
            end_idx = start_idx + gran_size
            m = scaling_factors[i*2]
            b = scaling_factors[i*2 + 1]
            # in rare cases the scaling factors are actually fill values
            if m <= -999 or b <= -999:
                mask[start_idx:end_idx] = 1
            else:
                data[start_idx:end_idx] *= m
                data[start_idx:end_idx] += b

    def adjust_scaling_factors(self, factors, file_units, output_units):
        if factors is None or factors[0] is None:
            factors = [1, 0]
        if file_units == output_units:
            LOG.debug("File units and output units are the same (%s)", file_units)
            return factors
        factors = np.array(factors)

        if file_units == "W cm-2 sr-1" and output_units == "W m-2 sr-1":
            LOG.debug("Adjusting scaling factors to convert '%s' to '%s'", file_units, output_units)
            factors[::2] = np.where(factors[::2] != -999, factors[::2] * 10000.0, -999)
            factors[1::2] = np.where(factors[1::2] != -999, factors[1::2] * 10000.0, -999)
            return factors
        elif file_units == "1" and output_units == "%":
            LOG.debug("Adjusting scaling factors to convert '%s' to '%s'", file_units, output_units)
            factors[::2] = np.where(factors[::2] != -999, factors[::2] * 100.0, -999)
            factors[1::2] = np.where(factors[1::2] != -999, factors[1::2] * 100.0, -999)
            return factors
        else:
            return factors

    def get_lonlats(self, navid, nav_info, lon_out, lat_out):
        lon_key = nav_info["longitude_key"]
        valid_min, valid_max = self[lon_key + '/attr/ValidRange']
        fill_value = self[lon_key + '/attr/{}'.format(self._fill_name)]
        lon_out.data[:] = self[lon_key][:]
        lon_out.mask[:] = (lon_out < valid_min) | (lon_out > valid_max) | (lon_out == fill_value)

        lat_key = nav_info["latitude_key"]
        valid_min, valid_max = self[lat_key + '/attr/ValidRange']
        fill_value = self[lat_key + '/attr/{}'.format(self._fill_name)]
        lat_out.data[:] = self[lat_key][:]
        lat_out.mask[:] = (lat_out < valid_min) | (lat_out > valid_max) | (lat_out == fill_value)

        return {}

    def get_dataset(self, dataset_id, ds_info, out=None):
        var_path = ds_info.get('file_key', '{}'.format(dataset_id.name))
        dtype = ds_info.get('dtype', np.float32)
        if var_path + '/shape' not in self:
            # loading a scalar value
            shape = 1
        else:
            shape = self.get_shape(dataset_id, ds_info)
        file_units = ds_info.get('file_units')
        if file_units is None:
            try:
                file_units = self[var_path + '/attr/Units']
                # they were almost completely CF compliant...
                if file_units == "none":
                    file_units = "1"
            except KeyError:
                # no file units specified
                file_units = None

        if out is None:
            out = np.ma.empty(shape, dtype=dtype)
            out.mask = np.zeros(shape, dtype=np.bool)

        try:
            valid_min, valid_max = self[var_path + '/attr/ValidRange']
        except KeyError:
            try:
                valid_min = self[var_path + '/attr/ValidMin']
                valid_max = self[var_path + '/attr/ValidMax']
            except KeyError:
                valid_min = valid_max = None
        fill_name = '/attr/{}'.format(self._fill_name)
        if fill_name in self:
            fill_value = self[fill_name]
        else:
            fill_value = None

        out.data[:] = np.require(self[var_path][:], dtype=dtype)
        scale_factor_path = var_path + '/attr/scale_factor'
        if scale_factor_path in self:
            scale_factor = self[scale_factor_path]
            scale_offset = self[var_path + '/attr/add_offset']
        else:
            scale_factor = None
            scale_offset = None

        if valid_min is not None and valid_max is not None:
            # the original .cfg/INI based reader only checked valid_max
            out.mask[:] |= (out.data > valid_max) # | (out < valid_min)
        if fill_value is not None:
            out.mask[:] |= out.data == fill_value

        factors = (scale_factor, scale_offset)
        factors = self.adjust_scaling_factors(factors, file_units, ds_info.get("units"))
        if factors[0] != 1 or factors[1] != 0:
            out.data[:] *= factors[0]
            out.data[:] += factors[1]

        ds_info.update({
            "units": ds_info.get("units", file_units),
            "platform": self.platform_name,
            "sensor": self.sensor_name,
            "start_orbit": self.start_orbit_number,
            "end_orbit": self.end_orbit_number,
        })
        ds_info.update(dataset_id.to_dict())
        if 'standard_name' not in ds_info:
            ds_info['standard_name'] = self[var_path + '/attr/Title']

        cls = ds_info.pop("container", Dataset)
        return cls(out, **ds_info)


class EDREOSFileHandler(EDRFileHandler):
    _fill_name = "MissingValue"
