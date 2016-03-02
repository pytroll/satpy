#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2011, 2012, 2013, 2014, 2015.

# Author(s):

#
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

"""Interface to VIIRS L1B format

"""
import os.path
from datetime import datetime, timedelta
import numpy as np
import logging

from satpy.readers import ConfigBasedReader, MultiFileReader, FileKey, GenericFileReader
import netCDF4
import six

NO_DATE = datetime(1958, 1, 1)
EPSILON_TIME = timedelta(days=2)
LOG = logging.getLogger(__name__)


class NetCDF4MetaData(object):
    """Small class for inspecting a NetCDF4 file and retrieve its metadata/header data.
    """
    def __init__(self, filename, auto_maskandscale=False, **kwargs):
        self.metadata = {}
        self.filename = filename
        if not os.path.exists(filename):
            raise IOError("File %s does not exist!" % filename)
        file_handle = netCDF4.Dataset(self.filename, 'r')

        self.auto_maskandscale= auto_maskandscale
        if hasattr(file_handle, "set_auto_maskandscale"):
            file_handle.set_auto_maskandscale(auto_maskandscale)

        self.collect_metadata("", file_handle)
        file_handle.close()

    def _collect_attrs(self, name, obj):
        for key in obj.ncattrs():
            value = getattr(obj, key)
            value = np.squeeze(value)
            if issubclass(value.dtype.type, str) or np.issubdtype(value.dtype, np.character):
                self.metadata["%s/attr/%s" % (name, key)] = str(value)
            else:
                self.metadata["%s/attr/%s" % (name, key)] = value

    def collect_metadata(self, name, obj):
        # Look through each subgroup
        base_name = name + "/" if name else ""
        for group_name, group_obj in obj.groups.items():
            self.collect_metadata(base_name + group_name, group_obj)
        for var_name, var_obj in obj.variables.items():
            var_name = base_name + var_name
            self.metadata[var_name] = var_obj
            self.metadata[var_name + "/shape"] = var_obj.shape
            self._collect_attrs(var_name, var_obj)
        self._collect_attrs(name, obj)

    def __getitem__(self, key):
        val = self.metadata[key]
        if isinstance(val, netCDF4.Variable):
            # these datasets are closed and inaccessible when the file is closed, need to reopen
            v = netCDF4.Dataset(self.filename, 'r')
            v.set_auto_maskandscale(self.auto_maskandscale)
            val = v[key]
        return val


class L1BFileReader(GenericFileReader):
    """VIIRS HDF5 File Reader
    """
    def create_file_handle(self, filename, **kwargs):
        handle = NetCDF4MetaData(filename, **kwargs)
        return handle.filename, handle

    def __getitem__(self, item):
        if item.endswith("/shape") and item[:-6] in self.file_keys:
            item = self.file_keys[item[:-6]].variable_name.format(**self.file_info) + "/shape"
        elif item in self.file_keys:
            item = self.file_keys[item].variable_name.format(**self.file_info)

        return self.file_handle[item]

    def _parse_l1b_datetime(self, datestr):
        return datetime.strptime(datestr, "%Y-%m-%dT%H:%M:%S.000Z")

    @property
    def ring_lonlats(self):
        return self["gring_longitude"], self["gring_latitude"]

    def _get_start_time(self):
        return self._parse_l1b_datetime(self['coverage_start'])

    def _get_end_time(self):
        return self._parse_l1b_datetime(self['coverage_end'])

    @property
    def begin_orbit_number(self):
        return int(self['beginning_orbit_number'])

    @property
    def end_orbit_number(self):
        return int(self['ending_orbit_number'])

    @property
    def platform_name(self):
        # FIXME: If an attribute is added to the file, for now hardcode
        # res = self['platform_short_name']
        res = "NPP"
        if isinstance(res, np.ndarray):
            return str(res.astype(str))
        else:
            return res

    @property
    def sensor_name(self):
        res = self['instrument_short_name']
        if isinstance(res, np.ndarray):
            return str(res.astype(str))
        else:
            return res

    @property
    def geofilename(self):
        res = self['geo_file_reference']
        if isinstance(res, np.ndarray):
            return str(res.astype(str))
        else:
            return res

    def get_file_units(self, item):
        # What units should we expect from the file
        file_units = self.file_keys[item].file_units

        # Guess the file units if we need to (normally we would get this from the file)
        if file_units is None:
            # see if there is an attribute
            try:
                var_path = self.file_keys[item].variable_name
                file_units = self[var_path + "/attr/units"]
                # some file units don't follow the satpy standards
                if file_units == "none":
                    file_units = "1"
                return file_units
            except (AttributeError, KeyError):
                LOG.debug("No units attribute found for '%s'", item)

            if "radiance" in item:
                # we are getting some sort of radiance, probably DNB
                file_units = "W cm-2 sr-1"
            elif "reflectance" in item:
                # CF compliant unit for dimensionless
                file_units = "1"
            elif "temperature" in item:
                file_units = "K"
            elif "longitude" in item or "latitude" in item:
                file_units = "degrees"
            else:
                LOG.debug("Unknown units for file key '%s'", item)

        return file_units

    def get_shape(self, item):
        return self[item + "/shape"]

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

    def get_swath_data(self, item, data_out=None, mask_out=None, dataset_id=None):
        """Get swath data, apply proper scalings, and apply proper masks.
        """
        # Can't guarantee proper file info until we get the data first
        var_info = self.file_keys[item]
        # NetCDF4 files for L1B have proper attributes so the NetCDF4 library
        # can auto scale and auto mask the data
        data = self[item]
        valid_max = data.valid_max
        data = data[:]
        if data_out is not None:
            # This assumes that we are promoting the dtypes (ex. float file data -> int array)
            # and that it happens automatically when assigning to the existing out array
            data_out[:] = data
        else:
            data_out = data[:].astype(var_info.dtype)
            mask_out = np.zeros_like(data_out, dtype=np.bool)

        # Check if we need to do some unit conversion
        # file_units = self.get_file_units(item)
        # output_units = getattr(var_info, "units", file_units

        if var_info.scaling_factors:
            # L1B has 2 separate factors
            factors_name, offset_name = var_info.scaling_factors.split(",")
            try:
                factors = (self[factors_name], self[offset_name])
            except KeyError:
                LOG.debug("No scaling factors found for %s", item)
                factors = None
        else:
            factors = None

        mask_out[:] |= data_out > valid_max

        # Check if we need to do some unit conversion
        file_units = self.get_file_units(item)
        output_units = getattr(var_info, "units", file_units)
        factors = self.adjust_scaling_factors(factors, file_units, output_units)

        if factors is not None:
            data_out *= factors[0]
            data_out += factors[1]

        return data_out, mask_out


class VIIRSL1BReader(ConfigBasedReader):
    def __init__(self, default_file_reader=L1BFileReader, default_config_filename="readers/viirs_l1b.cfg", **kwargs):
        super(VIIRSL1BReader, self).__init__(default_file_reader=default_file_reader,
                                             default_config_filename=default_config_filename,
                                             **kwargs
                                             )

    def _load_navigation(self, nav_name, extra_mask=None, dep_file_type=None):
        """Load the `nav_name` navigation.

        For VIIRS, if we haven't loaded the geolocation file read the `dep_file_type` header
        to figure out where it is.
        """
        nav_info = self.navigations[nav_name]
        file_type = nav_info["file_type"]

        # FUTURE: L1B files don't currently have references to their associated geolocation files so this won't work
        # if file_type not in self.file_readers:
        #     LOG.debug("Geolocation files were not provided, will search band file header...")
        #     if dep_file_type is None:
        #         raise RuntimeError("Could not find geolocation files because the main dataset was not provided")
        #     dataset_file_reader = self.file_readers[dep_file_type]
        #     base_dirs = [os.path.dirname(fn) for fn in dataset_file_reader.filenames]
        #     geo_filenames = dataset_file_reader.geofilenames
        #     geo_filepaths = [os.path.join(bd, gf) for bd, gf in zip(base_dirs, geo_filenames)]
        #
        #     file_types = self.identify_file_types(geo_filepaths)
        #     if file_type not in file_types:
        #         raise RuntimeError("The geolocation files from the header (ex. %s)"
        #                            " do not match the configured geolocation (%s)" % (geo_filepaths[0], file_type))
        #     self.file_readers[file_type] = MultiFileReader(file_type, file_types[file_type], self.file_keys)

        return super(VIIRSL1BReader, self)._load_navigation(nav_name, extra_mask=extra_mask)

