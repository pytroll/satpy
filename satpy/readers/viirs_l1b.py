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
from datetime import datetime
import numpy as np
from satpy.readers.netcdf_utils import NetCDF4FileHandler

LOG = logging.getLogger(__name__)


class VIIRSL1BFileHandler(NetCDF4FileHandler):
    """VIIRS L1B File Reader
    """
    def _parse_datetime(self, datestr):
        return datetime.strptime(datestr, "%Y-%m-%dT%H:%M:%S.000Z")

    @property
    def start_orbit_number(self):
        try:
            return int(self['/attr/orbit_number'])
        except KeyError:
            return int(self['/attr/OrbitNumber'])

    @property
    def end_orbit_number(self):
        try:
            return int(self['/attr/orbit_number'])
        except KeyError:
            return int(self['/attr/OrbitNumber'])

    @property
    def platform_name(self):
        try:
            res = self.get('/attr/platform',
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
        res = self['/attr/instrument']
        if isinstance(res, np.ndarray):
            return str(res.astype(str))
        else:
            return res

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

    def get_shape(self, ds_id, ds_info):
        var_path = ds_info.get('file_key', 'observation_data/{}'.format(ds_id.name))
        return self.get(var_path + '/shape', 1)

    @property
    def start_time(self):
        return self._parse_datetime(self['/attr/time_coverage_start'])

    @property
    def end_time(self):
        return self._parse_datetime(self['/attr/time_coverage_end'])

    def _load_and_slice(self, var_path, shape, xslice, yslice):
        if isinstance(shape, tuple) and len(shape) == 2:
            return self[var_path][yslice, xslice]
        elif isinstance(shape, tuple) and len(shape) == 1:
            return self[var_path][yslice]
        else:
            return self[var_path]

    def _get_dataset_file_units(self, dataset_id, ds_info, var_path):
        file_units = ds_info.get('file_units')
        if file_units is None:
            file_units = self.get(var_path + '/attr/units')
            # they were almost completely CF compliant...
            if file_units == "none":
                file_units = "1"

        if dataset_id.calibration == 'radiance' and ds_info['units'] == 'W m-2 um-1 sr-1':
            rad_units_path = var_path + '/attr/radiance_units'
            if rad_units_path in self:
                if file_units is None:
                    file_units = self[var_path + '/attr/radiance_units']
                if file_units == 'Watts/meter^2/steradian/micrometer':
                    file_units = 'W m-2 um-1 sr-1'
        elif ds_info.get('units') == '%' and file_units is None:
            # v1.1 and above of level 1 processing removed 'units' attribute
            # for all reflectance channels
            file_units = "1"

        return file_units

    def _get_dataset_valid_range(self, dataset_id, ds_info, var_path):
        if dataset_id.calibration == 'radiance' and ds_info['units'] == 'W m-2 um-1 sr-1':
            rad_units_path = var_path + '/attr/radiance_units'
            if rad_units_path in self:
                # we are getting a reflectance band but we want the radiance values
                # special scaling parameters
                scale_factor = self[var_path + '/attr/radiance_scale_factor']
                scale_offset = self[var_path + '/attr/radiance_add_offset']
            else:
                # we are getting a btemp band but we want the radiance values
                # these are stored directly in the primary variable
                scale_factor = self[var_path + '/attr/scale_factor']
                scale_offset = self[var_path + '/attr/add_offset']
            valid_min = self[var_path + '/attr/valid_min']
            valid_max = self[var_path + '/attr/valid_max']
        elif ds_info.get('units') == '%':
            # normal reflectance
            valid_min = self[var_path + '/attr/valid_min']
            valid_max = self[var_path + '/attr/valid_max']
            scale_factor = self[var_path + '/attr/scale_factor']
            scale_offset = self[var_path + '/attr/add_offset']
        elif ds_info.get('units') == 'K':
            # normal brightness temperature
            # use a special LUT to get the actual values
            lut_var_path = ds_info.get('lut', var_path + '_brightness_temperature_lut')
            # we get the BT values from a look up table using the scaled radiance integers
            valid_min = self[lut_var_path + '/attr/valid_min']
            valid_max = self[lut_var_path + '/attr/valid_max']
            scale_factor = scale_offset = None
        else:
            valid_min = self.get(var_path + '/attr/valid_min')
            valid_max = self.get(var_path + '/attr/valid_max')
            scale_factor = self.get(var_path + '/attr/scale_factor')
            scale_offset = self.get(var_path + '/attr/add_offset')

        return valid_min, valid_max, scale_factor, scale_offset

    def get_metadata(self, dataset_id, ds_info):
        var_path = ds_info.get('file_key', 'observation_data/{}'.format(dataset_id.name))
        shape = self.get_shape(dataset_id, ds_info)
        file_units = self._get_dataset_file_units(dataset_id, ds_info, var_path)

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

    def get_dataset(self, dataset_id, ds_info, xslice=slice(None), yslice=slice(None)):
        var_path = ds_info.get('file_key', 'observation_data/{}'.format(dataset_id.name))
        metadata = self.get_metadata(dataset_id, ds_info)
        shape = metadata['shape']
        if isinstance(shape, tuple) and len(shape) == 2:
            # 2D array
            if xslice.start is not None:
                shape = (shape[0], xslice.stop - xslice.start)
            if yslice.start is not None:
                shape = (yslice.stop - yslice.start, shape[1])
        elif isinstance(shape, tuple) and len(shape) == 1 and yslice.start is not None:
            shape = ((yslice.stop - yslice.start) / yslice.step,)
        metadata['shape'] = shape

        valid_min, valid_max, scale_factor, scale_offset = self._get_dataset_valid_range(dataset_id, ds_info, var_path)
        if dataset_id.calibration == 'radiance' and ds_info['units'] == 'W m-2 um-1 sr-1':
            data = self._load_and_slice(var_path, shape, xslice, yslice)
        elif ds_info.get('units') == '%':
            data = self._load_and_slice(var_path, shape, xslice, yslice)
        elif ds_info.get('units') == 'K':
            # normal brightness temperature
            # use a special LUT to get the actual values
            lut_var_path = ds_info.get('lut', var_path + '_brightness_temperature_lut')
            # we get the BT values from a look up table using the scaled radiance integers
            # .flatten() currently not supported, workaround: https://github.com/pydata/xarray/issues/1029
            data = self[var_path][yslice, xslice]
            data = data.stack(name=data.dims).astype(np.int)
            coords = data.coords
            data = self[lut_var_path][data]
            if 'dim_0' in data:
                # seems like older versions of xarray take the dims from
                # 'lut_var_path'. newer versions take 'data' dims
                data = data.rename({'dim_0': 'name'})
            data = data.assign_coords(**coords).unstack('name')
        elif shape == 1:
            data = self[var_path]
        else:
            data = self._load_and_slice(var_path, shape, xslice, yslice)

        if valid_min is not None and valid_max is not None:
            data = data.where((data >= valid_min) & (data <= valid_max))

        factors = (scale_factor, scale_offset)
        factors = self.adjust_scaling_factors(factors, metadata['file_units'], ds_info.get("units"))
        if factors[0] != 1 or factors[1] != 0:
            data = data * factors[0] + factors[1]

        data.attrs.update(metadata)
        return data
