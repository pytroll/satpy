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
"""Geostationary Lightning Mapper reader for the Level 2 format from glmtools.

More information about `glmtools` and the files it produces can be found on
the project's GitHub repository:

    https://github.com/deeplycloudy/glmtools

"""
import logging
from datetime import datetime

import numpy as np

from satpy.readers.abi_base import NC_ABI_BASE

logger = logging.getLogger(__name__)

PLATFORM_NAMES = {
    'G16': 'GOES-16',
    'G17': 'GOES-17',
    'G18': 'GOES-18',
}

# class NC_GLM_L2_LCFA(BaseFileHandler): — add this with glmtools


class NCGriddedGLML2(NC_ABI_BASE):
    """File reader for individual GLM L2 NetCDF4 files."""

    @property
    def sensor(self):
        """Get sensor name for current file handler."""
        return 'glm'

    @property
    def start_time(self):
        """Start time of the current file's observations."""
        return datetime.strptime(self.nc.attrs['time_coverage_start'], '%Y-%m-%dT%H:%M:%SZ')

    @property
    def end_time(self):
        """End time of the current file's observations."""
        return datetime.strptime(self.nc.attrs['time_coverage_end'], '%Y-%m-%dT%H:%M:%SZ')

    def _is_category_product(self, data_arr):
        # if after autoscaling we still have an integer
        is_int = np.issubdtype(data_arr.dtype, np.integer)
        # and it has a fill value
        has_fill = '_FillValue' in data_arr.attrs
        # or it has flag_meanings
        has_meanings = 'flag_meanings' in data_arr.attrs
        # then it is likely a category product and we should keep the
        # _FillValue for satpy to use later
        return is_int and (has_fill or has_meanings)

    def get_dataset(self, key, info):
        """Load a dataset."""
        logger.debug('Reading in get_dataset %s.', key['name'])
        res = self[key['name']]
        res.attrs.update({'platform_name': self.platform_name,
                          'sensor': self.sensor})
        res.attrs.update(self.filename_info)

        # Add orbital parameters
        projection = self.nc["goes_imager_projection"]
        res.attrs['orbital_parameters'] = {
            'projection_longitude': float(projection.attrs['longitude_of_projection_origin']),
            'projection_latitude': float(projection.attrs['latitude_of_projection_origin']),
            'projection_altitude': float(projection.attrs['perspective_point_height']),
            'satellite_nominal_latitude': float(self['nominal_satellite_subpoint_lat']),
            'satellite_nominal_longitude': float(self['nominal_satellite_subpoint_lon']),
            # 'satellite_nominal_altitude': float(self['nominal_satellite_height']),
        }

        res.attrs.update(key.to_dict())

        # remove attributes that could be confusing later
        if not self._is_category_product(res):
            res.attrs.pop('_FillValue', None)
        res.attrs.pop('scale_factor', None)
        res.attrs.pop('add_offset', None)
        res.attrs.pop('_Unsigned', None)
        res.attrs.pop('ancillary_variables', None)  # Can't currently load DQF
        # add in information from the filename that may be useful to the user
        # for key in ('observation_type', 'scene_abbr', 'scan_mode', 'platform_shortname'):
        for attr in ('scene_abbr', 'scan_mode', 'platform_shortname'):
            res.attrs[attr] = self.filename_info[attr]
        # copy global attributes to metadata
        for attr in ('scene_id', 'orbital_slot', 'instrument_ID',
                     'production_site', 'timeline_ID', 'spatial_resolution'):
            res.attrs[attr] = self.nc.attrs.get(attr)
        return res

    def _is_2d_xy_var(self, data_arr):
        is_2d = data_arr.ndim == 2
        has_x_dim = 'x' in data_arr.dims
        has_y_dim = 'y' in data_arr.dims
        return is_2d and has_x_dim and has_y_dim

    def available_datasets(self, configured_datasets=None):
        """Discover new datasets and add information from file."""
        # we know the actual resolution
        res = self.spatial_resolution_to_number()

        # update previously configured datasets
        handled_vars = set()
        for is_avail, ds_info in (configured_datasets or []):
            # some other file handler knows how to load this
            # don't override what they've done
            if is_avail is not None:
                yield is_avail, ds_info

            matches = self.file_type_matches(ds_info['file_type'])
            if matches and ds_info.get('resolution') != res:
                # we are meant to handle this dataset (file type matches)
                # and the information we can provide isn't available yet
                new_info = ds_info.copy()
                new_info['resolution'] = res
                exists = ds_info['name'] in self.nc
                handled_vars.add(ds_info['name'])
                yield exists, new_info
            elif is_avail is None:
                # we don't know what to do with this
                # see if another future file handler does
                yield is_avail, ds_info

        for var_name, data_arr in self.nc.data_vars.items():
            if var_name in handled_vars:
                # it was manually configured and handled above
                continue
            if not self._is_2d_xy_var(data_arr):
                # only handle 2d (y, x) vars for now
                continue

            new_info = {
                'name': var_name,
                'resolution': res,
                'file_type': self.filetype_info['file_type']
            }
            handled_vars.add(var_name)
            yield True, new_info
