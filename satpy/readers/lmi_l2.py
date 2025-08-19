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
"""Lightning Mapping Imager Dataset reader

The files read by this reader are described in:

    http://fy4.nsmc.org.cn/data/en/data/realtime.html

"""

import numpy as np
from satpy.readers.netcdf_utils import NetCDF4FileHandler, netCDF4
import logging

logger = logging.getLogger(__name__)


class LMIL2FileHandler(NetCDF4FileHandler):
    """File handler for LMI L2 netCDF files."""

    @property
    def start_time(self):
        return self.filename_info['start_time']

    @property
    def end_time(self):
        return self.filename_info['end_time']

    @property
    def platform_shortname(self):
        return self.filename_info['platform_id']

    @property
    def sensor(self):
        return self.filename_info['instrument'].lower()

    def available_datasets(self, configured_datasets=None):
        """Automatically determine datasets provided by this file."""
        logger.debug("Available_datasets begin...")

        handled_variables = set()

        # update previously configured datasets
        logger.debug("Starting previously configured variables loop...")
        for is_avail, ds_info in (configured_datasets or []):
            if ds_info['name'] == 'LAT':
                lat_shape = self[ds_info['name']+'/shape']

            # some other file handler knows how to load this
            if is_avail is not None:
                yield is_avail, ds_info

            var_name = ds_info.get('file_key', ds_info['name'])
            matches = self.file_type_matches(ds_info['file_type'])
            # we can confidently say that we can provide this dataset and can
            # provide more info
            if matches and var_name in self:
                logger.debug("Handling previously configured variable: %s",
                             var_name)
                # Because assembled variables and bounds use the same file_key,
                # we need to omit file_key once.
                handled_variables.add(var_name)
                new_info = ds_info.copy()  # don't mess up the above yielded
                yield True, new_info
            elif is_avail is None:
                # if we didn't know how to handle this dataset
                # and no one else did,
                # then we should keep it going down the chain.
                yield is_avail, ds_info

        # Iterate over dataset contents
        for var_name, val in self.file_content.items():
            # Only evaluate variables
            if isinstance(val, netCDF4.Variable):
                logger.debug("Evaluating new variable: %s", var_name)
                var_shape = self[var_name + "/shape"]
                logger.debug("Dims:{}".format(var_shape))
                if (lat_shape == var_shape):
                    logger.debug("Found valid additional dataset: %s",
                                 var_name)

                    # Skip anything we have already configured
                    if (var_name in handled_variables):
                        logger.debug("Already handled, skipping: %s", var_name)
                        continue
                    handled_variables.add(var_name)
                    logger.debug("Using short name of: %s", var_name)

                    # Create new ds_info object
                    new_info = {
                        'name': var_name,
                        'file_key': var_name,
                        'coordinates': ['LON', 'LAT'],
                        'file_type': self.filetype_info['file_type'],
                        'resolution': self.filename_info['resolution'].lower()
                    }
                    yield True, new_info

    def get_metadata(self, data, ds_info):
        """Get metadata."""
        metadata = {}
        metadata.update(data.attrs)
        units = data.attrs['units']
        # fix the wrong unit "uJ/m*m/ster"
        if not units.isascii():
            metadata['units'] = b'\xc2\xb5J/m*m/ster'
        metadata.update(ds_info)
        metadata.update({
            'platform_shortname': self.filename_info['platform_id'],
            'sensor': self.filename_info['instrument'].lower(),
            'start_time': self.start_time,
            'end_time': self.end_time,
        })

        return metadata

    def get_dataset(self, ds_id, ds_info):
        """Load a dataset."""
        logger.debug("Getting data for: %s", ds_id.name)
        file_key = ds_info.get('file_key', ds_id.name)
        data = self[file_key]
        data.attrs = self.get_metadata(data, ds_info)
        # rename coords
        data = data.rename({'x': 'y'})
        # assign 'y' coords which is useful for multiscene,
        # although the units isn't meters
        len_data = data.coords['y'].shape[0]
        data.coords['y'] = np.arange(len_data)
        # check fill value
        fill = data.attrs.pop('FillValue')
        data = data.where(data != fill)
        # remove attributes that could be confusing later
        data.attrs.pop('ancillary_variables', None)
        data.attrs.pop('Description', None)
        # select valid data
        data = data.where((data >= min(data.attrs['valid_range'])) &
                          (data <= max(data.attrs['valid_range'])))

        return data
