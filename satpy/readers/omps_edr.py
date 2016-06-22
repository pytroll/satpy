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
import os.path
from datetime import datetime, timedelta
import numpy as np
import logging

from satpy.readers import ConfigBasedReader, MultiFileReader, FileKey, GenericFileReader
from satpy.readers.hdf5_utils import HDF5MetaData
from trollsift.parser import parse as filename_parse

NO_DATE = datetime(1958, 1, 1)
EPSILON_TIME = timedelta(days=2)
LOG = logging.getLogger(__name__)


class EDRFileReader(GenericFileReader):
    def create_file_handle(self, filename, **kwargs):
        handle = HDF5MetaData(filename, **kwargs)
        return handle.filename, handle

    def __getitem__(self, item):
        base_item = item
        suffix = ""
        if "/attr/" in item:
            parts = item.split("/")
            base_item = "/".join(parts[:-2])
            suffix = "/" + "/".join(parts[-2:])
        elif item.endswith("/shape"):
            base_item = item[:-6]
            suffix = "/shape"

        replace_aggr = self.file_keys[base_item].kwargs.get("replace_aggr", None)
        if base_item in self.file_keys:
            var_info = self.file_keys[base_item]
            item = var_info.variable_name.format(**self.file_info)
            item += suffix

        if replace_aggr:
            # this is an aggregated field that can't easily be loaded, need to join things together
            idx = 0
            base_item = item
            item = base_item.replace(replace_aggr, str(idx))
            result = []
            while True:
                try:
                    res = self.file_handle[item]
                    result.append(res)
                except KeyError:
                    # no more granule keys
                    LOG.debug("Aggregated granule stopping on '%s'", item)
                    break

                idx += 1
                item = base_item.replace(replace_aggr, str(idx))
            return result
        else:
            return self.file_handle[item]

    def _get_start_time(self):
        return filename_parse(self.file_info["file_patterns"][0], self.filename)["start_time"]

    def _get_end_time(self):
        return filename_parse(self.file_info["file_patterns"][0], self.filename)["end_time"]

    # @property
    # def ring_lonlats(self):
    #     return self["gring_longitude"], self["gring_latitude"]

    @property
    def begin_orbit_number(self):
        return filename_parse(self.file_info["file_patterns"][0], self.filename)["orbit"]

    @property
    def end_orbit_number(self):
        return filename_parse(self.file_info["file_patterns"][0], self.filename)["orbit"]

    @property
    def platform_name(self):
        return filename_parse(self.file_info["file_patterns"][0], self.filename)["platform_shortname"]

    @property
    def sensor_name(self):
        return filename_parse(self.file_info["file_patterns"][0], self.filename)["instrument_shortname"]

    def get_file_units(self, item):
        # What units should we expect from the file
        unit_attr_name = self.file_keys[item].kwargs.get("units_attr")
        if unit_attr_name is not None:
            return self[item + "/attr/{}".format(unit_attr_name)]
        return None

        return file_units

    def get_shape(self, item):
        return self[item + "/shape"]

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
        if file_units == output_units:
            LOG.debug("File units and output units are the same (%s)", file_units)
            return factors
        if factors is None:
            factors = [1, 0]
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

    def get_swath_data(self, item, data_out=None, mask_out=None):
        """Get swath data, apply proper scalings, and apply proper masks.
        """
        # Can't guarantee proper file info until we get the data first
        var_info = self.file_keys[item]
        data = self[item]
        if data_out is not None:
            # This assumes that we are promoting the dtypes (ex. float file data -> int array)
            # and that it happens automatically when assigning to the existing out array
            data_out[:] = data
        else:
            data_out = data[:].astype(var_info.dtype)
            mask_out = np.zeros_like(data_out, dtype=np.bool)

        factor_attr_name = var_info.scaling_factors
        offset_attr_name = var_info.offset
        if factor_attr_name and offset_attr_name:
            try:
                factor = self[item + "/attr/{}".format(factor_attr_name)]
                offset = self[item + "/attr/{}".format(offset_attr_name)]
            except KeyError:
                LOG.debug("No scaling factors found for %s", item)
                factor = None
                offset = None
        else:
            factor = None
            offset = None

        fill_attr_name = var_info.kwargs.get("missing_attr")
        if fill_attr_name:
            fill_value = self[item + "/attr/{}".format(fill_attr_name)]
            mask_out[:] |= data_out == fill_value

        # Check if we need to do some unit conversion
        file_units = self.get_file_units(item)
        output_units = getattr(var_info, "units", file_units)
        factors = self.adjust_scaling_factors([factor, offset], file_units, output_units)

        if factors is not None and factors[0] is not None:
            self.scale_swath_data(data_out, mask_out, factors)

        return data_out, mask_out


class OMPSEDRReader(ConfigBasedReader):
    def __init__(self, default_file_reader=EDRFileReader, default_config_filename="readers/omps_edr.cfg", **kwargs):
        super(OMPSEDRReader, self).__init__(default_file_reader=default_file_reader,
                                            default_config_filename=default_config_filename,
                                            **kwargs
                                            )
