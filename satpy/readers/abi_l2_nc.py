#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Satpy developers
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
"""Advance Baseline Imager NOAA Level 2+ products reader.

The files read by this reader are described in the official PUG document:
    https://www.goes-r.gov/products/docs/PUG-L2+-vol5.pdf

"""

import logging
import numpy as np

from satpy.readers.abi_base import NC_ABI_BASE

LOG = logging.getLogger(__name__)


class NC_ABI_L2(NC_ABI_BASE):
    """Reader class for NOAA ABI l2+ products in netCDF format."""

    def get_dataset(self, key, info):
        """Load a dataset."""
        var = info['file_key']
        LOG.debug('Reading in get_dataset %s.', var)
        variable = self[var]

        _units = variable.attrs['units'] if 'units' in variable.attrs else None

        variable.attrs.update({'platform_name': self.platform_name,
                               'sensor': self.sensor,
                               'units': _units,
                               'satellite_latitude': float(self.nc['nominal_satellite_subpoint_lat']),
                               'satellite_longitude': float(self.nc['nominal_satellite_subpoint_lon']),
                               'satellite_altitude': float(self.nc['nominal_satellite_height'])})

        variable.attrs.update(key.to_dict())

        # remove attributes that could be confusing later
        if not np.issubdtype(variable.dtype, np.integer):
            # integer fields keep the _FillValue
            variable.attrs.pop('_FillValue', None)
        variable.attrs.pop('scale_factor', None)
        variable.attrs.pop('add_offset', None)
        variable.attrs.pop('valid_range', None)
        variable.attrs.pop('_Unsigned', None)
        variable.attrs.pop('valid_range', None)
        variable.attrs.pop('ancillary_variables', None)  # Can't currently load DQF

        if 'flag_meanings' in variable.attrs:
            variable.attrs['flag_meanings'] = variable.attrs['flag_meanings'].split(' ')

        # add in information from the filename that may be useful to the user
        for attr in ('scan_mode', 'platform_shortname'):
            variable.attrs[attr] = self.filename_info[attr]

        # copy global attributes to metadata
        for attr in ('scene_id', 'orbital_slot', 'instrument_ID', 'production_site', 'timeline_ID'):
            variable.attrs[attr] = self.nc.attrs.get(attr)

        return variable

    def available_datasets(self, configured_datasets=None):
        """Add resolution to configured datasets."""
        for is_avail, ds_info in (configured_datasets or []):
            # some other file handler knows how to load this
            # don't override what they've done
            if is_avail is not None:
                yield is_avail, ds_info
            matches = self.file_type_matches(ds_info['file_type'])
            if matches:
                # we have this dataset
                resolution = self.spatial_resolution_to_number()
                new_info = ds_info.copy()
                new_info.setdefault('resolution', resolution)
                yield True, ds_info
            elif is_avail is None:
                # we don't know what to do with this
                # see if another future file handler does
                yield is_avail, ds_info
