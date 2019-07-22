#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 Satpy developers

# Author(s):

#
#   Barry Baker @bbakernoaa GitHub
#   David Hoese <david.hoese@ssec.wisc.edu>
#   Daniel Hueholt <daniel.hueholt@noaa.gov>
#   Tommy Jasmin <tommy.jasmin@ssec.wisc.edu>
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

"""
Interface to JPSS_GRAN (JPSS VIIRS Products (Granule)) format
"""
from datetime import datetime
import numpy as np
from satpy.readers.netcdf_utils import NetCDF4FileHandler, netCDF4


class VIIRSEDRGRANFileHandler(NetCDF4FileHandler):
    """ VIIRS EDR GRAN reader """

    def _parse_datetime(self, datestr):
        """ Parses datetimes in file """
        return datetime.strptime(datestr, "%Y-%m-%dT%H:%M:%SZ")

    @property
    def start_orbit_number(self):
        """ Retrieves the starting orbit number from the file """
        try:
            return int(self['/attr/start_orbit_number'])
        except KeyError:
            return int(self['/attr/OrbitNumber'])

    @property
    def end_orbit_number(self):
        """ Retrieves the ending orbit number from the file """
        try:
            return int(self['/attr/end_orbit_number'])
        except KeyError:
            return int(self['/attr/OrbitNumber'])

    @property
    def platform_name(self):
        """ Retrieves the satellite name from the file """
        try:
            res = self.get('/attr/satellite_name',
                           self.filename_info['platform_shortname'])
        except KeyError:
            res = 'Suomi-NPP'

        return {
            'Suomi-NPP': 'NPP',
            'JPSS-1': 'J01',
            'NP': 'NPP',
            'J1': 'J01',
        }.get(res, res)

    @property
    def sensor_name(self):
        """ Retrieves the sensor name from the file """
        res = self['/attr/instrument_name']
        if isinstance(res, np.ndarray):
            return str(res.astype(str))
        return res

    def adjust_scaling_factors(self, factors, file_units, output_units):
        """ Adjusts factors to make sure that units always match between
        the data and the output
        """
        if factors is None or factors[0] is None:
            factors = [1, 0]
        if file_units == output_units:
            return factors
        factors = np.array(factors)

        if file_units == "W cm-2 sr-1" and output_units == "W m-2 sr-1":
            factors[::2] = np.where(factors[::2] != -999,
                                    factors[::2] * 10000.0, -999)
            factors[1::2] = np.where(factors[1::2] != -999,
                                     factors[1::2] * 10000.0, -999)
            return factors
        if file_units == "1" and output_units == "%":
            factors[::2] = np.where(factors[::2] != -999, factors[::2] * 100.0,
                                    -999)
            factors[1::2] = np.where(factors[1::2] != -999,
                                     factors[1::2] * 100.0, -999)
            return factors
        return factors

    def get_shape(self, ds_id, ds_info):
        """ Retrieves shape of the dataset """
        var_path = ds_info.get('file_key',
                               'observation_data/{}'.format(ds_id.name))
        return self.get(var_path + '/shape', 1)

    @property
    def start_time(self):
        return self._parse_datetime(self['/attr/time_coverage_start'])

    @property
    def end_time(self):
        return self._parse_datetime(self['/attr/time_coverage_end'])

    def _get_dataset_file_units(self, dataset_id, ds_info, var_path):
        file_units = ds_info.get('file_units')
        if file_units is None:
            file_units = self.get(var_path + '/attr/units')
            # They were almost completely CF compliant...
            if file_units == "none":
                file_units = "1"

        if (ds_info['units'] == 'W m-2 um-1 sr-1'):
            rad_units_path = var_path + '/attr/radiance_units'
            if rad_units_path in self:
                if file_units is None:
                    file_units = self[var_path + '/attr/radiance_units']
                if file_units == 'Watts/meter^2/steradian/micrometer':
                    file_units = 'W m-2 um-1 sr-1'
        elif ds_info.get('units') == '%' and file_units is None:
            file_units = "1"

        return file_units

    def _get_dataset_valid_range(self, dataset_id, ds_info, var_path):
        valid_min = self[var_path].valid_range[0]
        valid_max = self[var_path].valid_range[1]
        scale_factor = None
        scale_offset = None
        return valid_min, valid_max, scale_factor, scale_offset

    def get_metadata(self, dataset_id, ds_info):
        var_path = ds_info.get('file_key',
                               'observation_data/{}'.format(dataset_id.name))
        shape = self.get_shape(dataset_id, ds_info)
        file_units = self._get_dataset_file_units(dataset_id, ds_info,
                                                  var_path)

        # Get extra metadata
        if '/dimension/number_of_scans' in self:
            rows_per_scan = int(shape[0] / self['/dimension/number_of_scans'])
            ds_info.setdefault('rows_per_scan', rows_per_scan)

        i = getattr(self[var_path], 'attrs', {})
        i.update(ds_info)
        i.update(dataset_id.to_dict())
        i.update({
            "shape": shape,
            "units": ds_info.get("units", file_units),
            "file_units": file_units,
            "platform_name": self.platform_name,
            "sensor": self.sensor_name,
            "start_orbit": self.start_orbit_number,
            "end_orbit": self.end_orbit_number,
        })
        i.update(dataset_id.to_dict())
        return i

    def available_datasets(self, configured_datasets=None):
        """Automatically determine datasets provided by this file"""
        # Determine shape of the geolocation data (lat/lon)
        lat_shape = None
        for var_name, val in self.file_content.items():
            # Could probably avoid this hardcoding, will think on it
            if var_name == 'Latitude':
                lat_shape = self[var_name + "/shape"]
                break
        handled_variables = set()

        # Update previously configured datasets
        # Only geolocation variables, others generated dynamically
        for is_avail, ds_info in (configured_datasets or []):
            if is_avail is not None:
                yield is_avail, ds_info
            var_name = ds_info.get('file_key', ds_info['name'])
            matches = self.file_type_matches(ds_info['file_type'])
            # Can provide this dataset and more info
            if matches and var_name in self:
                handled_variables.add(var_name)
                new_info = ds_info.copy()
                yield True, new_info
            elif is_avail is None:
                yield is_avail, ds_info

        # Sift through groups and variables for data matching lat/lon shape
        for var_name, val in self.file_content.items():
            # Only evaluate variables
            if isinstance(val, netCDF4.Variable):
                var_shape = self[var_name + "/shape"]
                if var_shape == lat_shape:
                    # Skip anything we have already configured
                    if var_name in handled_variables:
                        continue
                    handled_variables.add(var_name)
                    new_info = {
                        'name': var_name.lower(),
                        'resolution': 742,
                        'units': self[var_name].units,
                        'long_name': var_name,
                        'file_key': var_name,
                        'file_type': self.filetype_info['file_type'],
                        'coordinates': ['longitude', 'latitude'],
                    }
                    yield True, new_info

    def get_dataset(self, dataset_id, ds_info):
        var_path = ds_info.get('file_key', dataset_id.name)
        metadata = self.get_metadata(dataset_id, ds_info)

        valid_min, valid_max, scale_factor, scale_offset = \
            self._get_dataset_valid_range(dataset_id, ds_info, var_path)
        data = self[var_path]
        data.attrs.update(metadata)

        if valid_min is not None and valid_max is not None:
            data = data.where((data >= valid_min) & (data <= valid_max))
        if data.attrs.get('units') in ['%', 'K', '1', 'W m-2 um-1 sr-1'] and \
                'flag_meanings' in data.attrs:
            # Flag meanings don't mean anything anymore for these variables
            # these aren't category products
            data.attrs.pop('flag_meanings', None)
            data.attrs.pop('flag_values', None)

        factors = (scale_factor, scale_offset)
        factors = self.adjust_scaling_factors(factors, metadata['file_units'],
                                              ds_info.get("units"))
        if factors[0] != 1 or factors[1] != 0:
            data *= factors[0]
            data += factors[1]

        if 'Rows' in data.dims:
            data = data.rename({'Rows': 'y', 'Columns': 'x'})
        return data
