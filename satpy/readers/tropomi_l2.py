#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Satpy developers
#
# This file is part of Satpy.
#
# Satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# Satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# Satpy.  If not, see <http://www.gnu.org/licenses/>.

"""Interface to TROPOMI L2 Reader

The TROPOspheric Monitoring Instrument (TROPOMI) is the satellite instrument
on board the Copernicus Sentinel-5 Precursor satellite. It measures key
atmospheric trace gasses, such as ozone, nitrogen oxides, sulfur dioxide,
carbon monoxide, methane, and formaldehyde.

Level 2 data products are available via the Copernicus Open Access Hub.
For more information visit the following URL:
http://www.tropomi.eu/data-products/level-2-products

"""
from satpy.readers.netcdf_utils import NetCDF4FileHandler, netCDF4
import logging
import numpy as np

logger = logging.getLogger(__name__)


class TROPOMIL2FileHandler(NetCDF4FileHandler):
    @property
    def start_time(self):
        return self.filename_info['start_time']

    @property
    def end_time(self):
        return self.filename_info.get('end_time', self.start_time)

    @property
    def platform_shortname(self):
        return self.filename_info['platform_shortname']

    @property
    def sensor(self):
        """ Retrieves the sensor name from the file """
        res = self['/attr/sensor']
        if isinstance(res, np.ndarray):
            return str(res.astype(str))
        return res

    def available_datasets(self, configured_datasets=None):
        """Automatically determine datasets provided by this file"""
        logger.debug("Available_datasets begin...")

        # Determine shape of the geolocation data (lat/lon)
        lat_shape = None
        for var_name, val in self.file_content.items():
            # Could probably avoid this hardcoding, will think on it
            if (var_name == 'PRODUCT/latitude'):
                lat_shape = self[var_name + "/shape"]
                break

        handled_variables = set()

        # update previously configured datasets
        logger.debug("Starting previously configured variables loop...")
        for is_avail, ds_info in (configured_datasets or []):

            # some other file handler knows how to load this
            if is_avail is not None:
                yield is_avail, ds_info

            var_name = ds_info.get('file_key', ds_info['name'])
            # logger.debug("Evaluating previously configured variable: %s", var_name)
            matches = self.file_type_matches(ds_info['file_type'])
            # we can confidently say that we can provide this dataset and can
            # provide more info
            if matches and var_name in self:
                logger.debug("Handling previously configured variable: %s", var_name)
                handled_variables.add(var_name)
                new_info = ds_info.copy()  # don't mess up the above yielded
                yield True, new_info
            elif is_avail is None:
                # if we didn't know how to handle this dataset and no one else did
                # then we should keep it going down the chain
                yield is_avail, ds_info

        # This is where we dynamically add new datasets
        # We will sift through all groups and variables, looking for data matching
        # the geolocation bounds

        # Iterate over dataset contents
        for var_name, val in self.file_content.items():
            # Only evaluate variables
            if isinstance(val, netCDF4.Variable):
                logger.debug("Evaluating new variable: %s", var_name)
                var_shape = self[var_name + "/shape"]
                logger.debug("Dims:{}".format(var_shape))
                if (var_shape == lat_shape):
                    logger.debug("Found valid additional dataset: %s", var_name)
                    # Skip anything we have already configured
                    if (var_name in handled_variables):
                        logger.debug("Already handled, skipping: %s", var_name)
                        continue
                    handled_variables.add(var_name)
                    last_index_separator = var_name.rindex('/')
                    last_index_separator = last_index_separator + 1
                    var_name_no_path = var_name[last_index_separator:]
                    logger.debug("Using short name of: %s", var_name_no_path)
                    # Create new ds_info object
                    new_info = {
                        'name': var_name_no_path,
                        'file_key': var_name,
                        'coordinates': ['longitude', 'latitude'],
                        'file_type': self.filetype_info['file_type'],
                        'resolution': None,
                    }
                    yield True, new_info

    def get_metadata(self, data, ds_info):
        metadata = {}
        metadata.update(data.attrs)
        metadata.update(ds_info)
        metadata.update({
            'platform_shortname': self.platform_shortname,
            'sensor': self.sensor,
            'start_time': self.start_time,
            'end_time': self.end_time,
        })

        return metadata

    def get_dataset(self, ds_id, ds_info):
        logger.debug("Getting data for: %s", ds_id.name)
        file_key = ds_info.get('file_key', ds_id.name)
        data = self[file_key]
        data.attrs = self.get_metadata(data, ds_info)
        fill = data.attrs.pop('_FillValue')
        data = data.squeeze()
        data = data.where(data != fill)
        return data
