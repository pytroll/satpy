#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 Satpy developers
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

"""SMOS L2 wind Reader.

Data can be found here after register:
https://www.smosstorm.org/Data2/SMOS-NRT-wind-Products-access
Format documentation at the same site after register:
SMOS_WIND_DS_PDD_20191107_signed.pdf
"""

import logging
import numpy as np
from datetime import datetime
from pyresample.geometry import AreaDefinition
from satpy.readers.netcdf_utils import NetCDF4FileHandler, netCDF4

logger = logging.getLogger(__name__)


class SMOSL2WINDFileHandler(NetCDF4FileHandler):
    """File handler for SMOS L2 wind netCDF files."""

    @property
    def start_time(self):
        """Get start time."""
        return datetime.strptime(self['/attr/time_coverage_start'], "%Y-%m-%dT%H:%M:%S Z")

    @property
    def end_time(self):
        """Get end time."""
        return datetime.strptime(self['/attr/time_coverage_end'], "%Y-%m-%dT%H:%M:%S Z")

    @property
    def platform_shortname(self):
        """Get platform shortname."""
        return self.filename_info['platform_shortname']

    @property
    def platform_name(self):
        """Get platform."""
        return self['/attr/platform']

    def get_metadata(self, data, ds_info):
        """Get metadata."""
        metadata = {}
        metadata.update(data.attrs)
        metadata.update(ds_info)
        metadata.update({
            'platform_shortname': self.platform_shortname,
            'platform_name': self.platform_name,
            'sensor': self['/attr/instrument'],
            'start_time': self.start_time,
            'end_time': self.end_time,
            'level': self['/attr/processing_level'],
        })

        return metadata

    def available_datasets(self, configured_datasets=None):
        """Automatically determine datasets provided by this file."""
        logger.debug("Available_datasets begin...")

        # This is where we dynamically add new datasets
        # We will sift through all groups and variables, looking for data matching
        # the geolocation bounds
        handled_variables = set()

        # Iterate over dataset contents
        for var_name, val in self.file_content.items():
            # Only evaluate variables
            if not isinstance(val, netCDF4.Variable):
                continue
            if (var_name in handled_variables):
                logger.debug("Already handled, skipping: %s", var_name)
                continue
            handled_variables.add(var_name)
            new_info = {
                'name': var_name,
                'file_type': self.filetype_info['file_type'],
            }
            yield True, new_info

    def _mask_dataset(self, data):
        """Mask out fill values ( and remove the _FillValue from attributes)."""
        fill = data.attrs.pop('_FillValue')
        return data.where(data != fill)

    def _rename_coords(self, data):
        """Rename coords."""
        rename_dict = {}
        if 'lon' in data.dims:
            # Want lons range from -180 .. 180 ( not 0 .. 360)
            data = data.assign_coords(lon=(((data.lon + 180) % 360) - 180))
            rename_dict['lon'] = 'x'
        if 'lat' in data.dims:
            rename_dict['lat'] = 'y'
        # Rename the coordinates to x and y
        return data.rename(rename_dict)

    def _remove_time_coordinate(self, data):
        """Remove time coordinate."""
        # Remove dimension where size is 1, eg. time
        data = data.squeeze()
        # Remove if exists time as coordinate
        if 'time' in data.coords:
            data = data.drop_vars('time')
        return data

    def get_dataset(self, ds_id, ds_info):
        """Get dataset."""
        data = self[ds_id.name]
        data.attrs = self.get_metadata(data, ds_info)
        data = self._remove_time_coordinate(data)
        data = self._rename_coords(data)
        # Reorganize the data to have it from -180 to 180
        data = data.roll(x=720, roll_coords=True)
        data = self._mask_dataset(data)
        return data

    def _adjust_lon_interval(self):
        """Adjust lon interval"""
        # Fix coordinates values >= 180 to -180 to 0
        _lon = self['lon']
        _lon = _lon.assign_coords(lon=(((_lon.lon + 180) % 360) - 180))
        # Fix the data
        _lon = _lon.where(_lon < 180., _lon - 360.)
        # Roll the data acordingly
        return _lon.roll(lon=720, roll_coords=True)

    def _get_shape(self):
        width = self['lon/shape'][0]
        height = self['lat/shape'][0]
        return width, height

    def _create_area_extent(self, width, height):
        """Create area extent"""
        # Creating a meshgrid, not needed aqtually, but makes it easy to find extremes
        _lon = self._adjust_lon_interval()
        latlon = np.meshgrid(_lon, self['lat'])

        lower_left_x = latlon[0][height - 1][0]
        lower_left_y = latlon[1][height - 1][0]

        upper_right_y = latlon[1][0][width - 1]
        upper_right_x = latlon[0][0][width - 1]
        return (lower_left_x, lower_left_y, upper_right_x, upper_right_y)

    def get_area_def(self, dsid):
        """Define AreaDefintion."""
        width, height = self._get_shape()
        area_extent = self._create_area_extent(width, height)
        description = "SMOS L2 Wind Equirectangular Projection"
        area_id = 'smos_eqc'
        proj_id = 'equirectangular'
        proj_dict = {'init': self['/attr/geospatial_bounds_vertical_crs']}
        area_def = AreaDefinition(area_id, description, proj_id, proj_dict, width, height, area_extent, )
        return area_def
