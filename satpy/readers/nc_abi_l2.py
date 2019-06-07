#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016-2019 Satpy developers
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
"""
Advance Baseline Imager NOAA Level 2+ products reader 
The files read by this reader are described in the official PUG document:
    https://www.goes-r.gov/products/docs/PUG-L2+-vol5.pdf
"""

import logging

from pyresample import geometry
from satpy.readers.abi_base import NC_ABI_BASE

LOG = logging.getLogger(__name__)

class NC_ABI_L2(NC_ABI_BASE):

    def __init__(self, filename, filename_info, filetype_info):
        super(NC_ABI_L2, self).__init__(filename, filename_info, filetype_info)


    def get_dataset(self, key, info):
        """Load a dataset.
        """
        var = info['file_key']
        LOG.debug('Reading in get_dataset %s.', var)
        variable = self.nc[var]

        # handle coordinates (and recursive fun)
        new_coords = {}
        # 'time' dimension causes issues in other processing
        if 'time' in variable.coords:
            del variable.coords['time']

        if var in variable.coords:
            self.coords[var] = variable

        for coord_name in variable.coords.keys():
            if coord_name not in self.coords:
                self.coords[coord_name] = self.nc[coord_name]
            new_coords[coord_name] = self.coords[coord_name]

        variable.coords.update(new_coords)

        _units = variable.attrs['units'] if 'units' in variable.attrs else None

        variable.attrs.update({'platform_name': self.platform_name,
                               'sensor': self.sensor,
                               'units': _units,
                               'satellite_latitude': float(self.nc['nominal_satellite_subpoint_lat']),
                               'satellite_longitude': float(self.nc['nominal_satellite_subpoint_lon']),
                               'satellite_altitude': float(self.nc['nominal_satellite_height'])})

        variable.attrs.update(key.to_dict())

        # remove attributes that could be confusing later
        variable.attrs.pop('_FillValue', None)
        variable.attrs.pop('scale_factor', None)
        variable.attrs.pop('add_offset', None)
        variable.attrs.pop('valid_range', None)
        
        # add in information from the filename that may be useful to the user
        for key in ('scan_mode', 'platform_shortname'):
            variable.attrs[key] = self.filename_info[key]

        # copy global attributes to metadata
        for key in ('scene_id', 'orbital_slot', 'instrument_ID', 'production_site', 'timeline_ID'):
            variable.attrs[key] = self.nc.attrs.get(key)

        return variable

