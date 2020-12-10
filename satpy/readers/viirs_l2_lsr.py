#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 20 Satpy developers
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
"""VIIRS Land Surface Reflectance reader.

This module implements readers for VIIRS Land Surface Reflectance NetCDF
files, with additions for CSPP LEO NDVI & EVI products.
"""

from satpy.readers.netcdf_utils import NetCDF4FileHandler
from satpy.readers.file_handlers import BaseFileHandler
import dask.dataframe as dd
import xarray as xr
import numpy as np

# map platform attributes to Oscar standard name
PLATFORM_MAP = {
    "NPP": "Suomi-NPP",
    "J01": "NOAA-20",
    "J02": "NOAA-21"
}


class VIIRSLandSurfaceReflectanceFileHandler(NetCDF4FileHandler):
    """NetCDF4 reader for VIIRS Land Surface Reflectance."""

    def __init__(self, filename, filename_info, filetype_info,
                 auto_maskandscale=False, xarray_kwargs=None):
        """Open and perform initial investigation of NetCDF file."""
        super(VIIRSLandSurfaceReflectanceFileHandler, self).__init__(
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

        data.attrs["platform_name"] = PLATFORM_MAP.get(self.filename_info['platform_shortname'].upper(), "unknown")
        data.attrs["sensor"] = "VIIRS"

        if dsid['name'] == "ndvi" or dsid['name'] == "evi":
            data = data.where(data <= 1.0, np.float32(np.nan))
            data = data.where(data >= -1.0, np.float32(np.nan))

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
        return self["sensor"]

    @property
    def platform_name(self):
        """Name of platform/satellite for this file."""
        return self["platform_name"]
