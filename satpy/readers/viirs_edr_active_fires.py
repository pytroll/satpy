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
"""VIIRS Active Fires reader.

This module implements readers for VIIRS Active Fires NetCDF and
ASCII files.
"""

import dask.dataframe as dd
import xarray as xr

from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.netcdf_utils import NetCDF4FileHandler

# map platform attributes to Oscar standard name
PLATFORM_MAP = {
    "NPP": "Suomi-NPP",
    "J01": "NOAA-20",
    "J02": "NOAA-21"
}


class VIIRSActiveFiresFileHandler(NetCDF4FileHandler):
    """NetCDF4 reader for VIIRS Active Fires."""

    def __init__(self, filename, filename_info, filetype_info,
                 auto_maskandscale=False, xarray_kwargs=None):
        """Open and perform initial investigation of NetCDF file."""
        super(VIIRSActiveFiresFileHandler, self).__init__(
            filename, filename_info, filetype_info,
            auto_maskandscale=auto_maskandscale, xarray_kwargs=xarray_kwargs)
        self.prefix = filetype_info.get('variable_prefix')

    def get_dataset(self, dsid, dsinfo):
        """Get requested data as DataArray.

        Args:
            dsid: Dataset ID
            param2: Dataset Information

        Returns:
            Dask DataArray: Data

        """
        key = dsinfo.get('file_key', dsid['name']).format(variable_prefix=self.prefix)
        data = self[key]
        # rename "phoney dims"
        data = data.rename(dict(zip(data.dims, ['y', 'x'])))

        # handle attributes from YAML
        for key in ('units', 'standard_name', 'flag_meanings', 'flag_values', '_FillValue'):
            # we only want to add information that isn't present already
            if key in dsinfo and key not in data.attrs:
                data.attrs[key] = dsinfo[key]
        if isinstance(data.attrs.get('flag_meanings'), str):
            data.attrs['flag_meanings'] = data.attrs['flag_meanings'].split(' ')

        # use more common CF standard units
        if data.attrs.get('units') == 'kelvins':
            data.attrs['units'] = 'K'

        data.attrs["platform_name"] = PLATFORM_MAP.get(self.filename_info['satellite_name'].upper(), "unknown")
        data.attrs["sensor"] = self.sensor_name

        return data

    @property
    def start_time(self):
        """Get first date/time when observations were recorded."""
        return self.filename_info['start_time']

    @property
    def end_time(self):
        """Get last date/time when observations were recorded."""
        return self.filename_info.get('end_time', self.start_time)

    @property
    def sensor_name(self):
        """Name of sensor for this file."""
        return self["/attr/instrument_name"].lower()

    @property
    def platform_name(self):
        """Name of platform/satellite for this file."""
        return self["/attr/satellite_name"]


class VIIRSActiveFiresTextFileHandler(BaseFileHandler):
    """ASCII reader for VIIRS Active Fires."""

    def __init__(self, filename, filename_info, filetype_info):
        """Make sure filepath is valid and then reads data into a Dask DataFrame.

        Args:
            filename: Filename
            filename_info: Filename information
            filetype_info: Filetype information

        """
        skip_rows = filetype_info.get('skip_rows', 15)
        columns = filetype_info['columns']
        self.file_content = dd.read_csv(filename, skiprows=skip_rows, header=None, names=columns)
        super(VIIRSActiveFiresTextFileHandler, self).__init__(filename, filename_info, filetype_info)
        self.platform_name = PLATFORM_MAP.get(self.filename_info['satellite_name'].upper(), "unknown")

    def get_dataset(self, dsid, dsinfo):
        """Get requested data as DataArray."""
        ds = self[dsid['name']].to_dask_array(lengths=True)
        data = xr.DataArray(ds, dims=("y",), attrs={"platform_name": self.platform_name, "sensor": "VIIRS"})
        for key in ('units', 'standard_name', 'flag_meanings', 'flag_values', '_FillValue'):
            # we only want to add information that isn't present already
            if key in dsinfo and key not in data.attrs:
                data.attrs[key] = dsinfo[key]
        if isinstance(data.attrs.get('flag_meanings'), str):
            data.attrs['flag_meanings'] = data.attrs['flag_meanings'].split(' ')
        return data

    @property
    def start_time(self):
        """Get first date/time when observations were recorded."""
        return self.filename_info['start_time']

    @property
    def end_time(self):
        """Get last date/time when observations were recorded."""
        return self.filename_info.get('end_time', self.start_time)

    def __getitem__(self, key):
        """Get file content for 'key'."""
        return self.file_content[key]

    def __contains__(self, item):
        """Check if variable is in current file."""
        return item in self.file_content
