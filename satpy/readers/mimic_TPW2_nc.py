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
#
#

"""Reader for Mimic TPW data in netCDF format from SSEC.

This module implements reader for MIMIC_TPW2 netcdf files.
MIMIC-TPW2 is an experimental global product of total precipitable water (TPW),
using morphological compositing of the MIRS retrieval from several available
operational microwave-frequency sensors. Originally described in a 2010 paper by
Wimmers and Velden. This Version 2 is developed from an older method that uses simpler,
but more limited TPW retrievals and advection calculations.

More information, data and credits at
http://tropic.ssec.wisc.edu/real-time/mtpw2/credits.html
"""

import logging

import numpy as np
import xarray as xr
from pyresample.geometry import AreaDefinition

from satpy.readers.netcdf_utils import NetCDF4FileHandler, netCDF4

logger = logging.getLogger(__name__)


class MimicTPW2FileHandler(NetCDF4FileHandler):
    """NetCDF4 reader for MIMC TPW."""

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the reader."""
        super(MimicTPW2FileHandler, self).__init__(filename, filename_info,
                                                   filetype_info,
                                                   xarray_kwargs={"decode_times": False})

    def available_datasets(self, configured_datasets=None):
        """Get datasets in file matching gelocation shape (lat/lon)."""
        lat_shape = self.file_content.get('/dimension/lat')
        lon_shape = self.file_content.get('/dimension/lon')

        # Read the lat/lon variables?
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

        # Iterate over dataset contents
        for var_name, val in self.file_content.items():
            # Only evaluate variables
            if isinstance(val, netCDF4.Variable):
                logger.debug("Evaluating new variable: %s", var_name)
                var_shape = self[var_name + "/shape"]
                logger.debug("Dims:{}".format(var_shape))
                if var_shape == (lat_shape, lon_shape):
                    logger.debug("Found valid additional dataset: %s", var_name)
                    # Skip anything we have already configured
                    if var_name in handled_variables:
                        logger.debug("Already handled, skipping: %s", var_name)
                        continue
                    handled_variables.add(var_name)
                    # Create new ds_info object
                    new_info = {
                        'name': var_name,
                        'file_key': var_name,
                        'file_type': self.filetype_info['file_type'],
                    }
                    logger.debug(var_name)
                    yield True, new_info

    def get_dataset(self, ds_id, info):
        """Load dataset designated by the given key from file."""
        logger.debug("Getting data for: %s", ds_id['name'])
        file_key = info.get('file_key', ds_id['name'])
        data = np.flipud(self[file_key])
        data = xr.DataArray(data, dims=['y', 'x'])
        data.attrs = self.get_metadata(data, info)

        if 'lon' in data.dims:
            data.rename({'lon': 'x'})
        if 'lat' in data.dims:
            data.rename({'lat': 'y'})

        return data

    def get_area_def(self, dsid):
        """Flip data up/down and define equirectangular AreaDefintion."""
        flip_lat = np.flipud(self['latArr'])
        latlon = np.meshgrid(self['lonArr'], flip_lat)

        width = self['lonArr/shape'][0]
        height = self['latArr/shape'][0]

        lower_left_x = latlon[0][height-1][0]
        lower_left_y = latlon[1][height-1][0]

        upper_right_y = latlon[1][0][width-1]
        upper_right_x = latlon[0][0][width-1]

        area_extent = (lower_left_x, lower_left_y, upper_right_x, upper_right_y)
        description = "MIMIC TPW WGS84"
        area_id = 'mimic'
        proj_id = 'World Geodetic System 1984'
        projection = 'EPSG:4326'
        area_def = AreaDefinition(area_id, description, proj_id, projection, width, height, area_extent, )
        return area_def

    def get_metadata(self, data, info):
        """Get general metadata for file."""
        metadata = {}
        metadata.update(data.attrs)
        metadata.update(info)
        metadata.update({
            'platform_shortname': 'aggregated microwave',
            'sensor': 'mimic',
            'start_time': self.start_time,
            'end_time': self.end_time,
        })
        metadata.update(self[info.get('file_key')].variable.attrs)

        return metadata

    @property
    def start_time(self):
        """Start timestamp of the dataset determined from yaml."""
        return self.filename_info['start_time']

    @property
    def end_time(self):
        """End timestamp of the dataset same as start_time."""
        return self.filename_info.get('end_time', self.start_time)

    @property
    def sensor_name(self):
        """Sensor name."""
        return self["sensor"]
