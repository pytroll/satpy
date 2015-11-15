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

# This file is part of mpop.

# mpop is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# mpop is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# mpop.  If not, see <http://www.gnu.org/licenses/>.

"""Interface to VIIRS SDR format

Format documentation:
http://npp.gsfc.nasa.gov/science/sciencedocuments/082012/474-00001-03_CDFCBVolIII_RevC.pdf

"""
import os.path
from datetime import datetime, timedelta
from trollsift.parser import globify
import numpy as np
import h5py
import logging
from collections import namedtuple

from mpop.projectable import Projectable
from mpop.readers import Reader, DatasetDict, ConfigBasedReader, MultiFileReader
from fnmatch import fnmatch
import six

NO_DATE = datetime(1958, 1, 1)
EPSILON_TIME = timedelta(days=2)
LOG = logging.getLogger(__name__)


def _get_invalid_info(granule_data):
    """Get a detailed report of the missing data.
        N/A: not applicable
        MISS: required value missing at time of processing
        OBPT: onboard pixel trim (overlapping/bow-tie pixel removed during
            SDR processing)
        OGPT: on-ground pixel trim (overlapping/bow-tie pixel removed
            during EDR processing)
        ERR: error occurred during processing / non-convergence
        ELINT: ellipsoid intersect failed / instrument line-of-sight does
            not intersect the Earth’s surface
        VDNE: value does not exist / processing algorithm did not execute
        SOUB: scaled out-of-bounds / solution not within allowed range
    """
    if issubclass(granule_data.dtype.type, np.integer):
        msg = ("na:" + str((granule_data == 65535).sum()) +
               " miss:" + str((granule_data == 65534).sum()) +
               " obpt:" + str((granule_data == 65533).sum()) +
               " ogpt:" + str((granule_data == 65532).sum()) +
               " err:" + str((granule_data == 65531).sum()) +
               " elint:" + str((granule_data == 65530).sum()) +
               " vdne:" + str((granule_data == 65529).sum()) +
               " soub:" + str((granule_data == 65528).sum()))
    elif issubclass(granule_data.dtype.type, np.floating):
        msg = ("na:" + str((granule_data == -999.9).sum()) +
               " miss:" + str((granule_data == -999.8).sum()) +
               " obpt:" + str((granule_data == -999.7).sum()) +
               " ogpt:" + str((granule_data == -999.6).sum()) +
               " err:" + str((granule_data == -999.5).sum()) +
               " elint:" + str((granule_data == -999.4).sum()) +
               " vdne:" + str((granule_data == -999.3).sum()) +
               " soub:" + str((granule_data == -999.2).sum()))
    return msg


class FileKey(namedtuple("FileKey", ["name", "variable_name", "scaling_factors", "dtype", "standard_name", "units", "file_units", "kwargs"])):
    def __new__(cls, name, variable_name,
                scaling_factors=None, dtype=np.float32, standard_name=None, units=None, file_units=None, **kwargs):
        if isinstance(dtype, (str, six.text_type)):
            # get the data type from numpy
            dtype = getattr(np, dtype)
        return super(FileKey, cls).__new__(cls, name, variable_name, scaling_factors, dtype, standard_name, units, file_units, kwargs)


class HDF5MetaData(object):
    """Small class for inspecting a HDF5 file and retrieve its metadata/header data.
    """
    def __init__(self, filename, **kwargs):
        self.metadata = {}
        self.filename = filename
        if not os.path.exists(filename):
            raise IOError("File %s does not exist!" % filename)
        file_handle = h5py.File(self.filename, 'r')
        file_handle.visititems(self.collect_metadata)
        self._collect_attrs('', file_handle.attrs)
        file_handle.close()

    def _collect_attrs(self, name, attrs):
        for key, value in six.iteritems(attrs):
            value = np.squeeze(value)
            if issubclass(value.dtype.type, str):
                self.metadata["%s/attr/%s" % (name, key)] = str(value)
            else:
                self.metadata["%s/attr/%s" % (name, key)] = value

    def collect_metadata(self, name, obj):
        if isinstance(obj, h5py.Dataset):
            self.metadata[name] = obj
            self.metadata[name + "/shape"] = obj.shape
        self._collect_attrs(name, obj.attrs)

    def __getitem__(self, key):
        val = self.metadata[key]
        if isinstance(val, h5py.Dataset):
            # these datasets are closed and inaccessible when the file is closed, need to reopen
            return h5py.File(self.filename, 'r')[key].value
        return val


class SDRFileReader(HDF5MetaData):
    """VIIRS HDF5 File Reader
    """
    def __init__(self, file_type, filename, file_keys, **kwargs):
        super(SDRFileReader, self).__init__(filename, **kwargs)
        self.file_type = file_type
        self.file_keys = file_keys
        self.file_info = kwargs

        self.start_time = self.get_begin_time()
        self.end_time = self.get_end_time()

    def __getitem__(self, item):
        if item.endswith("/shape") and item[:-6] in self.file_keys:
            item = self.file_keys[item[:-6]].variable_name.format(**self.file_info) + "/shape"
        elif item in self.file_keys:
            item = self.file_keys[item].variable_name.format(**self.file_info)

        return super(SDRFileReader, self).__getitem__(item)

    def _parse_npp_datetime(self, datestr, timestr):
        try:
            datetime_str = datestr + timestr
        except TypeError:
            datetime_str = str(datestr.astype(str)) + str(timestr.astype(str))
        time_val = datetime.strptime(datetime_str, '%Y%m%d%H%M%S.%fZ')
        if abs(time_val - NO_DATE) < EPSILON_TIME:
            # catch rare case when SDR files have incorrect date
            raise ValueError("Datetime invalid %s " % time_val)
        return time_val

    def get_ring_lonlats(self):
        return self["gring_longitude"], self["gring_latitude"]

    def get_begin_time(self):
        return self._parse_npp_datetime(self['beginning_date'], self['beginning_time'])

    def get_end_time(self):
        return self._parse_npp_datetime(self['ending_date'], self['ending_time'])

    def get_begin_orbit_number(self):
        return int(self['beginning_orbit_number'])

    def get_end_orbit_number(self):
        return int(self['ending_orbit_number'])

    def get_platform_name(self):
        return self['platform_short_name']

    def get_sensor_name(self):
        return self['instrument_short_name']

    def get_geofilename(self):
        return self['geo_file_reference']

    def get_file_units(self, item):
        # What units should we expect from the file
        file_units = self.file_keys[item].file_units

        # Guess the file units if we need to (normally we would get this from the file)
        if file_units is None:
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

    def get_units(self, item):
        units = self.file_keys[item].units
        file_units = self.get_file_units(item)
        # What units does the user want
        if units is None:
            # if the units in the file information
            return file_units
        return units

        # if calibrate == 2 and band not in VIIRS_DNB_BANDS:
        #     return "W m-2 um-1 sr-1"
        #
        # if band in VIIRS_IR_BANDS:
        #     return "K"
        # elif band in VIIRS_VIS_BANDS:
        #     return '%'
        # elif band in VIIRS_DNB_BANDS:
        #     return 'W m-2 sr-1'
        #
        # return None

    def get_shape(self, item):
        return self[item + "/shape"]

    def scale_swath_data(self, data, mask, scaling_factors):
        """Scale swath data using scaling factors and offsets.

        Multi-granule (a.k.a. aggregated) files will have more than the usual two values.
        """
        num_grans = len(scaling_factors)/2
        gran_size = data.shape[0]/num_grans
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

    def get_swath_data(self, item, data_out=None, mask_out=None, dataset_name=None):
        """Get swath data, apply proper scalings, and apply proper masks.
        """
        # Can't guarantee proper file info until we get the data first
        var_info = self.file_keys[item]
        data = self[item]
        is_floating = np.issubdtype(data.dtype, np.floating)
        if data_out is not None:
            # This assumes that we are promoting the dtypes (ex. float file data -> int array)
            # and that it happens automatically when assigning to the existing out array
            data_out[:] = data
        else:
            data_out = data[:].astype(var_info.dtype)
            mask_out = np.zeros_like(data_out, dtype=np.bool)

        if var_info.scaling_factors:
            try:
                factors = self[var_info.scaling_factors]
            except KeyError:
                LOG.debug("No scaling factors found for %s", item)
                factors = None
        else:
            factors = None

        if is_floating:
            # If the data is a float then we mask everything <= -999.0
            fill_max = float(var_info.kwargs.get("fill_max_float", -999.0))
            mask_out[:] |= data_out <= fill_max
        else:
            # If the data is an integer then we mask everything >= fill_min_int
            fill_min = int(var_info.kwargs.get("fill_min_int", 65528))
            mask_out[:] |= data_out >= fill_min

        # Check if we need to do some unit conversion
        file_units = self.get_file_units(item)
        output_units = getattr(var_info, "units", file_units)
        factors = self.adjust_scaling_factors(factors, file_units, output_units)

        if factors is not None:
            self.scale_swath_data(data_out, mask_out, factors)

        return data_out, mask_out


class VIIRSSDRReader(ConfigBasedReader):
    def __init__(self, default_file_reader=SDRFileReader, default_config_filename="readers/viirs_sdr.cfg", **kwargs):
        super(VIIRSSDRReader, self).__init__(default_file_reader=default_file_reader,
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

        if file_type not in self.file_readers:
            LOG.debug("Geolocation files were not provided, will search band file header...")
            if dep_file_type is None:
                raise RuntimeError("Could not find geolocation files because the main dataset was not provided")
            dataset_file_reader = self.file_readers[dep_file_type]
            base_dirs = [os.path.dirname(fn) for fn in dataset_file_reader.filenames]
            geo_filenames = dataset_file_reader.geo_filenames
            geo_filepaths = [os.path.join(bd, gf) for bd, gf in zip(base_dirs, geo_filenames)]

            file_types = self.identify_file_types(geo_filepaths)
            if file_type not in file_types:
                raise RuntimeError("The geolocation files from the header (ex. %s)"
                                   " do not match the configured geolocation (%s)" % (geo_filepaths[0], file_type))
            self.file_readers[file_type] = MultiFileReader(file_type, file_types[file_type], self.file_keys)

        return super(VIIRSSDRReader, self)._load_navigation(nav_name, extra_mask=extra_mask)

