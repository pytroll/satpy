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

"""Reader for Satpy-produced netcdf files (CF writer)."""

from satpy.readers.file_handlers import BaseFileHandler
import logging
import xarray as xr
from satpy import CHUNK_SIZE

logger = logging.getLogger(__name__)


class SatpyCFFileHandler(BaseFileHandler):
    """File handler for Satpy's CF netCDF files."""

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize file handler."""
        super().__init__(filename, filename_info, filetype_info)
        self.engine = None

    @property
    def start_time(self):
        """Get start time."""
        return self.filename_info['start_time']

    @property
    def end_time(self):
        """Get end time."""
        return self.filename_info.get('end_time', self.start_time)

    @property
    def sensor(self):
        """Get sensor."""
        nc = xr.open_dataset(self.filename, engine=self.engine)
        return nc.attrs['instrument'].replace('/', '-').lower()

    @property
    def sensor_names(self):
        """Get sensor set."""
        return {self.sensor}

    def available_datasets(self, configured_datasets=None):
        """Add information to configured datasets."""
        # pass along existing datasets
        for is_avail, ds_info in (configured_datasets or []):
            yield is_avail, ds_info
        nc = xr.open_dataset(self.filename, engine=self.engine)
        # get dynamic variables known to this file (that we created)
        for var_name, val in nc.data_vars.items():
            ds_info = dict(val.attrs)
            ds_info['file_type'] = self.filetype_info['file_type']
            ds_info['name'] = var_name
            try:
                ds_info['wavelength'] = tuple([float(wlength) for wlength in ds_info['wavelength'][0:3]])

            except KeyError:
                pass
            # Empty modifiers are read as [], which causes problems later
            if 'modifiers' in ds_info and len(ds_info['modifiers']) == 0:
                ds_info['modifiers'] = ()
            try:
                try:
                    ds_info['modifiers'] = tuple(ds_info['modifiers'].split(' '))
                except AttributeError:
                    pass
            except KeyError:
                pass
            yield True, ds_info
        for var_name, val in nc.coords.items():
            ds_info = dict(val.attrs)
            ds_info['file_type'] = self.filetype_info['file_type']
            ds_info['name'] = var_name
            yield True, ds_info

    def get_dataset(self, ds_id, ds_info):
        """Get dataset."""
        logger.debug("Getting data for: %s", ds_id['name'])
        nc = xr.open_dataset(self.filename, engine=self.engine,
                             chunks={'y': CHUNK_SIZE, 'x': CHUNK_SIZE})
        file_key = ds_info.get('file_key', ds_id['name'])
        data = nc[file_key]
        if file_key in nc.coords:
            data = data.drop_vars(list(nc.coords.keys()))
        try:
            data.attrs['wavelength'] = tuple([float(wlength) for wlength in ds_info['wavelength'][0:3]])
        except KeyError:
            pass
        # Empty modifiers are read as [], which causes problems later
        if 'modifiers' in ds_info and len(ds_info['modifiers']) == 0:
            ds_info['modifiers'] = ()
        try:
            # FIXME in cf writer: this is not consitent: no modifier is (), modifiers is a string
            try:
                ds_info['modifiers'] = tuple(ds_info['modifiers'].split(' '))
            except AttributeError:
                pass
        except KeyError:
            pass
        data.attrs.update(nc.attrs)  # For now add global attributes to all datasets
        return data
