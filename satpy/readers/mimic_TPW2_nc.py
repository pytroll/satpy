#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 Satpy developers
#
# Author(s):
#     Joleen Feltz <joleen.feltz@ssec.wisc.edu>
#     David Hoese <david.hoese@ssec.wisc.edu>
#     Daniel Hueholt <daniel.hueholt@noaa.gov>
#     Tommy Jasmin <tommy.jasmin@ssec.wisc.edu>
#
# This file is part of Satpy
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#
"""MIMIC_TPW2FileReader
*************************

This module implements readers for MIMIC_TPW2
netcdf files.
"""

from satpy.readers.netcdf_utils import NetCDF4FileHandler, netCDF4


class MIMIC_TPW2FileHandler(NetCDF4FileHandler):
    """NetCDF4 reader for MIMC TPW 2.0
    """
    def __init__(self, filename, filename_info, filetype_info,
                 auto_maskandscale=False, xarray_kwargs=None):
        super(MIMIC_TPW2FileHandler, self).__init__(
            filename, filename_info, filetype_info)

    def get_dataset(self, dataset_id, ds_info):
        var_path = ds_info.get('file_key', dataset_id.name)
        metadata = self.get_metadata(dataset_id, ds_info)

        data = self[var_path]
        data.attrs.update(metadata)

        if 'latArr' in data.dims:
            data = data.rename({'latArr': 'y', 'lonArr': 'x'})
        return data

    def available_datasets(self, configured_datasets=None):
        """Automatically determine datasets provided by this file"""
        # Determine shape of the geolocation data (lat/lon)
        lat_shape = self["latArr"].shape
        lon_shape = self["lonArr"].shape
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
                if var_shape == (lat_shape[0], lon_shape[0]):
                    # Skip anything already configured
                    if var_name in handled_variables:
                        continue
                    handled_variables.add(var_name)
                    # Create new ds_info object. Copy over some attributes,
                    # tune others or set from values in file.
                    new_info = ds_info.copy()
                    new_info.update({
                        'name': var_name.lower(),
                        'resolution': None,
                        'units': self[var_name].units,
                        'long_name': var_name,
                        'standard_name':  None,
                        'file_key': var_name,
                        'coordinates': ['latArr', 'lonArr'],
                    })
                    yield True, new_info

    def get_metadata(self, data, ds_info):
        metadata = {}

        return metadata

    @property
    def start_time(self):
        return self.filename_info['start_time']

    @property
    def end_time(self):
        return self.filename_info.get('end_time', self.start_time)

    @property
    def sensor_name(self):
        return self["sensor"]