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
import logging
import os.path
from datetime import datetime, timedelta
from glob import glob

import numpy as np

from satpy.dataset import Dataset
from satpy.readers.hdf5_utils import HDF5FileHandler
from satpy.readers.yaml_reader import FileYAMLReader

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


class VIIRSSDRFileHandler(HDF5FileHandler):
    """VIIRS HDF5 File Reader
    """

    def __getitem__(self, item):
        if '*' in item:
            # this is an aggregated field that can't easily be loaded, need to
            # join things together
            idx = 0
            base_item = item
            item = base_item.replace('*', str(idx))
            result = []
            while True:
                try:
                    res = super(VIIRSSDRFileHandler, self).__getitem__(item)
                    result.append(res)
                except KeyError:
                    # no more granule keys
                    LOG.debug("Aggregated granule stopping on '%s'", item)
                    break

                idx += 1
                item = base_item.replace('*', str(idx))
            return result
        else:
            return super(VIIRSSDRFileHandler, self).__getitem__(item)

    def _parse_datetime(self, datestr, timestr):
        try:
            datetime_str = datestr + timestr
        except TypeError:
            datetime_str = str(datestr.astype(str)) + str(timestr.astype(str))
        time_val = datetime.strptime(datetime_str, '%Y%m%d%H%M%S.%fZ')
        if abs(time_val - NO_DATE) < EPSILON_TIME:
            # catch rare case when SDR files have incorrect date
            raise ValueError("Datetime invalid {}".format(time_val))
        return time_val

    @property
    def start_time(self):
        default_start_date = 'Data_Products/{file_group}/{file_group}_Aggr/attr/AggregateBeginningDate'
        default_start_time = 'Data_Products/{file_group}/{file_group}_Aggr/attr/AggregateBeginningTime'
        date_var_path = self.filetype_info.get('start_date', default_start_date).format(**self.filetype_info)
        time_var_path = self.filetype_info.get('start_time', default_start_time).format(**self.filetype_info)
        return self._parse_datetime(self[date_var_path], self[time_var_path])

    @property
    def end_time(self):
        default_end_date = 'Data_Products/{file_group}/{file_group}_Aggr/attr/AggregateEndingDate'
        default_end_time = 'Data_Products/{file_group}/{file_group}_Aggr/attr/AggregateEndingTime'
        date_var_path = self.filetype_info.get('end_date', default_end_date).format(**self.filetype_info)
        time_var_path = self.filetype_info.get('end_time', default_end_time).format(**self.filetype_info)
        return self._parse_datetime(self[date_var_path], self[time_var_path])

    @property
    def start_orbit_number(self):
        default = 'Data_Products/{file_group}/{file_group}_Aggr/attr/AggregateBeginningOrbitNumber'
        start_orbit_path = self.filetype_info.get('start_orbit', default).format(**self.filetype_info)
        return int(self[start_orbit_path])

    @property
    def end_orbit_number(self):
        default = 'Data_Products/{file_group}/{file_group}_Aggr/attr/AggregateEndingOrbitNumber'
        end_orbit_path = self.filetype_info.get('end_orbit', default).format(**self.filetype_info)
        return int(self[end_orbit_path])

    @property
    def platform_name(self):
        default = '/attr/Platform_Short_Name'
        platform_path = self.filetype_info.get(
            'platform_name', default).format(**self.filetype_info)
        platform_dict = {'NPP': 'Suomi-NPP'}
        return platform_dict.get(self[platform_path], self[platform_path])

    @property
    def sensor_name(self):
        default = 'Data_Products/{file_group}/attr/Instrument_Short_Name'
        sensor_path = self.filetype_info.get(
            'sensor_name', default).format(**self.filetype_info)
        return self[sensor_path].lower()

    def get_file_units(self, dataset_id, ds_info):
        file_units = ds_info.get("file_units")

        # Guess the file units if we need to (normally we would get this from
        # the file)
        if file_units is None:
            if dataset_id.calibration == 'radiance':
                if "dnb" in dataset_id.name.lower():
                    return 'W m-2 sr-1'
                else:
                    return 'W cm-2 sr-1'
            elif dataset_id.calibration == 'reflectance':
                # CF compliant unit for dimensionless
                file_units = "1"
            elif dataset_id.calibration == 'brightness_temperature':
                file_units = "K"
            else:
                LOG.debug("Unknown units for file key '%s'", dataset_id)

        return file_units

    def scale_swath_data(self, data, mask, scaling_factors):
        """Scale swath data using scaling factors and offsets.

        Multi-granule (a.k.a. aggregated) files will have more than the usual two values.
        """
        num_grans = len(scaling_factors) // 2
        gran_size = data.shape[0] // num_grans
        for i in range(num_grans):
            start_idx = i * gran_size
            end_idx = start_idx + gran_size
            m = scaling_factors[i * 2]
            b = scaling_factors[i * 2 + 1]
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

    def _generate_file_key(self, ds_id, ds_info, factors=False):
        var_path = ds_info.get('file_key', 'All_Data/{file_group}_All/{calibration}')
        calibration = {
            'radiance': 'Radiance',
            'reflectance': 'Reflectance',
            'brightness_temperature': 'BrightnessTemperature',
        }.get(ds_id.calibration)
        var_path = var_path.format(calibration=calibration, **self.filetype_info)
        return var_path

    def get_shape(self, ds_id, ds_info):
        var_path = self._generate_file_key(ds_id, ds_info)
        return self[var_path + "/shape"]

    def get_dataset(self, dataset_id, ds_info, out=None):
        var_path = self._generate_file_key(dataset_id, ds_info)
        factor_var_path = ds_info.get("factors_key", var_path + "Factors")
        data = self[var_path]
        dtype = ds_info.get("dtype", np.float32)
        is_floating = np.issubdtype(data.dtype, np.floating)
        if out is not None:
            # This assumes that we are promoting the dtypes (ex. float file data -> int array)
            # and that it happens automatically when assigning to the existing
            # out array
            out.data[:] = data
        else:
            shape = self.get_shape(dataset_id, ds_info)
            out = np.ma.empty(shape, dtype=dtype)
            out.mask = np.zeros(shape, dtype=np.bool)

        if is_floating:
            # If the data is a float then we mask everything <= -999.0
            fill_max = float(ds_info.pop("fill_max_float", -999.0))
            out.mask[:] |= out.data <= fill_max
        else:
            # If the data is an integer then we mask everything >= fill_min_int
            fill_min = int(ds_info.pop("fill_min_int", 65528))
            out.mask[:] |= out.data >= fill_min

        factors = self.get(factor_var_path)
        if factors is None:
            LOG.debug("No scaling factors found for %s", dataset_id)

        file_units = self.get_file_units(dataset_id, ds_info)
        output_units = ds_info.get("units", file_units)
        factors = self.adjust_scaling_factors(factors, file_units, output_units)

        if factors is not None:
            self.scale_swath_data(out.data, out.mask, factors)

        i = getattr(out, 'info', {})
        i.update(ds_info)
        i.update({
            "units": ds_info.get("units", file_units),
            "platform_name": self.platform_name,
            "sensor": self.sensor_name,
            "start_orbit": self.start_orbit_number,
            "end_orbit": self.end_orbit_number,
        })
        i.update(dataset_id.to_dict())
        cls = ds_info.pop("container", Dataset)
        return cls(out.data, mask=out.mask, copy=False, **i)


class VIIRSSDRReader(FileYAMLReader):
    """Custom file reader for finding VIIRS SDR geolocation at runtime."""
    def __init__(self, config_files, use_tc=True, **kwargs):
        """Initialize file reader and adjust geolocation preferences.
        
        Args:
            config_files (iterable): yaml config files passed to base class
            use_tc (boolean): If `True` (default) use the terrain corrected
                              file types specified in the config files. If
                              `False`, switch all terrain corrected file types
                              to non-TC file types. If `None` 
                                    
        """
        super(VIIRSSDRReader, self).__init__(config_files, **kwargs)
        for ds_info in self.ids.values():
            ft = ds_info.get('file_type')
            if ft == 'gmtco':
                nontc = 'gmodo'
            elif ft == 'gitco':
                nontc = 'gimgo'
            else:
                continue

            if use_tc is None:
                # we want both TC and non-TC
                ds_info['file_type'] = [ds_info['file_type'], nontc]
            elif not use_tc:
                # we want only non-TC
                ds_info['file_type'] = nontc

    def _load_from_geo_ref(self, dsid):
        """Load filenames from the N_GEO_Ref attribute of a dataset's file"""
        file_handlers = self._get_file_handlers(dsid)
        if not file_handlers:
            return None

        fns = []
        for fh in file_handlers:
            base_dir = os.path.dirname(fh.filename)
            try:
                # get the filename and remove the creation time
                # which is often wrong
                fn = fh['/attr/N_GEO_Ref'][:46] + '*.h5'
                fns.extend(glob(os.path.join(base_dir, fn)))

                # usually is non-terrain corrected file, add the terrain
                # corrected file too
                if fn[:5] == 'GIMGO':
                    fn = 'GITCO' + fn[5:]
                elif fn[:5] == 'GMODO':
                    fn = 'GMTCO' + fn[5:]
                else:
                    continue
                fns.extend(glob(os.path.join(base_dir, fn)))
            except KeyError:
                LOG.debug("Could not load geo-reference information from {}".format(fh.filename))

        return fns

    def _get_coordinates_for_dataset_key(self, dsid):
        """Get the coordinate dataset keys for `dsid`.
        
        Wraps the base class method in order to load geolocation files
        from the geo reference attribute in the datasets file.
        """
        coords = super(VIIRSSDRReader, self)._get_coordinates_for_dataset_key(dsid)
        for c_id in coords:
            c_file_type = self.ids[c_id]['file_type']
            if self._preferred_filetype(c_file_type):
                # coordinate has its file type loaded already
                continue

            # check the dataset file for the geolocation filename
            geo_filenames = self._load_from_geo_ref(dsid)
            if not geo_filenames:
                continue

            self.create_filehandlers(geo_filenames)
        return coords
