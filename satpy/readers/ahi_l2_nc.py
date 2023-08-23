#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.

"""Advanced Himawari Imager (AHI) reader for the Level 2 NC format

Ref:
    ftp://ftp.ptree.jaxa.jp/pub/README_HimawariGeo_en.txt
"""

from satpy.readers.netcdf_utils import NetCDF4FileHandler, netCDF4
import logging
import numpy as np

logger = logging.getLogger(__name__)


class AHIL2FileHandler(NetCDF4FileHandler):
    """File handler for AHI L2 netCDF files."""
    def available_datasets(self, configured_datasets=None):
        """Automatically determine datasets provided by this file."""
        logger.debug('Available_datasets begin...')
        handled_variables = set()

        for is_avail, ds_info in (configured_datasets or []):
            # some other file handler knows how to load this
            if is_avail is not None:
                yield is_avail, ds_info

            var_name = ds_info.get('file_key', ds_info['name'])
            logger.debug("Evaluating previously configured variable: %s", var_name)
            matches = self.file_type_matches(ds_info['file_type'])

            if (matches and var_name in self):
                logger.debug('Handling previously configured variable: %s', var_name)
                handled_variables.add(var_name)
                new_info = ds_info.copy()  # don't mess up the above yielded
                yield True, new_info
            elif is_avail is None:
                # if we didn't know how to handle this dataset and no one else did
                # then we should keep it going down the chain
                yield is_avail, ds_info

        # Iterate over dataset contents
        for var_name, val in self.file_content.items():
            # Only evaluate variables
            if isinstance(val, netCDF4.Variable):
                logger.debug('Found valid additional dataset: %s', var_name)

                # Skip anything we have already configured
                if (var_name in handled_variables):
                    logger.debug('Already handled, skipping: %s', var_name)
                    continue
                handled_variables.add(var_name)

                # Create new ds_info object
                new_info = {
                    'name': var_name,
                    'file_key': var_name,
                    'file_type': self.filetype_info['file_type'],
                    'resolution': None,
                }
                yield True, new_info

    def get_metadata(self, data, ds_info):
        """Get metadata."""
        metadata = {}
        metadata.update(data.attrs)
        metadata.update(ds_info)
        metadata.update({
            'platform_shortname': self.platform_shortname,
            'sensor': 'ahi',
            'start_time': self.start_time,
        })

        return metadata

    def get_dataset(self, ds_id, ds_info):
        """Get dataset."""
        logger.debug('Getting data for: %s', ds_id.name)
        file_key = ds_info.get('file_key', ds_id.name)

        # read data and attributes
        data = self[file_key]
        data.attrs = self.get_metadata(data, ds_info)

        scale_factor = data.attrs.get('scale_factor')
        add_offset = data.attrs.get('add_offset')
        valid_min = data.attrs.get('valid_min')
        valid_max = data.attrs.get('valid_max')
        fill_value = data.attrs.get('missing_value', np.float32(np.nan))
        data = data.squeeze()

        # preserve integer data types if possible
        if np.issubdtype(data.dtype, np.integer):
            new_fill = fill_value
        else:
            new_fill = np.float32(np.nan)
            data.attrs.pop('missing_value', None)
        good_mask = data != fill_value

        # scale data
        if scale_factor is not None:
            data.data = data.data * scale_factor + add_offset

        # mask data
        data = data.where(good_mask, new_fill)
        data = data.where((data >= valid_min) &
                          (data <= valid_max))

        return data

    @property
    def start_time(self):
        """Get start time."""
        return self.filename_info['start_time']

    @property
    def platform_shortname(self):
        """Get platform shortname."""
        return self.filename_info['platform_shortname']

    @property
    def sensor(self):
        """Assign sensor."""
        return 'ahi'

    @property
    def sensor_names(self):
        """Assign sensor set."""
        return 'ahi'
