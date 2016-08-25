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
import logging
from datetime import datetime, timedelta

import numpy as np

from satpy.readers import ConfigBasedReader, GenericFileReader
from satpy.readers.netcdf_utils import NetCDF4FileHandler

NO_DATE = datetime(1958, 1, 1)
EPSILON_TIME = timedelta(days=2)
LOG = logging.getLogger(__name__)


class L1BFileReader(GenericFileReader):
    """VIIRS L1B File Reader
    """
    def create_file_handle(self, filename, **kwargs):
        handle = NetCDF4FileHandler(filename, **kwargs)
        return handle.filename, handle

    def __getitem__(self, item):
        if item.endswith("/shape") and item[:-6] in self.file_keys:
            item = self.file_keys[item[:-6]].variable_name.format(**self.file_info) + "/shape"
        elif item in self.file_keys:
            item = self.file_keys[item].variable_name.format(**self.file_info)

        return self.file_handle[item]

    def _parse_datetime(self, datestr):
        return datetime.strptime(datestr, "%Y-%m-%dT%H:%M:%S.000Z")

    @property
    def ring_lonlats(self):
        return self["gring_longitude"], self["gring_latitude"]

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

    def get_swath_data(self, item, data_out=None, mask_out=None):
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
        mask_out[:] |= data_out > valid_max

        if "lut" in var_info.kwargs:
            factors = None
            lut = self[var_info.kwargs["lut"]][:]
            # Note: Need to use the original data as `data_out` might be a non-integer data type
            data_out[:] = lut[data.ravel()].reshape(data.shape)
        elif var_info.factor:
            # L1B has 2 separate factors
            factors_name, offset_name = var_info.factor.split(",")
            try:
                factors = (self[factors_name], self[offset_name])
            except KeyError:
                LOG.debug("No scaling factors found for %s", item)
                factors = None
        else:
            factors = None

        # Check if we need to do some unit conversion
        file_units = self.get_file_units(item)
        output_units = getattr(var_info, "units", file_units)
        factors = self.adjust_scaling_factors(factors, file_units, output_units)

        if factors is not None:
            data_out *= factors[0]
            data_out += factors[1]

        return data_out, mask_out


class VIIRSL1BReader(ConfigBasedReader):
    """Reader for NASA VIIRS L1B NetCDF4 files.
    """
    def __init__(self, default_file_reader=L1BFileReader, default_config_filename="readers/viirs_l1b.cfg", **kwargs):
        super(VIIRSL1BReader, self).__init__(default_file_reader=default_file_reader,
                                             default_config_filename=default_config_filename,
                                             **kwargs
                                             )

    def load_navigation(self, nav_name, extra_mask=None, dep_file_type=None):
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

        return super(VIIRSL1BReader, self).load_navigation(nav_name, extra_mask=extra_mask)

