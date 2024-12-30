#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Satpy developers
#
# This file is part of Satpy.
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

"""Reader for CAMEL Level 3 emissivity files in netCDF4 format.

For more information about the data, see: <https://lpdaac.usgs.gov/products/cam5k30emv002/>.

NOTE: This reader only supports the global 0.05 degree grid data.
"""


import datetime as dt
import logging

import xarray as xr
from pyresample import geometry

from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)

# Area extent for the CAMEL product (global)
GLOB_AREA_EXT = [-180, -90, 180, 90]


class CAMELL3NCFileHandler(BaseFileHandler):
    """File handler for CAMEL data in netCDF format."""

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the reader."""
        super().__init__(filename, filename_info, filetype_info)
        self.nc = xr.open_dataset(self.filename,
                                  decode_cf=True,
                                  mask_and_scale=True,
                                  chunks={"xc": "auto", "yc": "auto"})

        if "0.05" not in self.nc.attrs["geospatial_lon_resolution"]:
            raise ValueError("Only 0.05 degree grid data is supported.")
        if "0.05" not in self.nc.attrs["geospatial_lat_resolution"]:
            raise ValueError("Only 0.05 degree grid data is supported.")

        self.nlines = self.nc.sizes["latitude"]
        self.ncols = self.nc.sizes["longitude"]
        self.area = None


    @property
    def start_time(self):
        """Start timestamp of the dataset."""
        date_str = self.nc.attrs["time_coverage_start"]
        return dt.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%SZ")

    @property
    def end_time(self):
        """End timestamp of the dataset."""
        date_str = self.nc.attrs["time_coverage_end"]
        return dt.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%SZ")


    def get_dataset(self, key, info):
        """Load a dataset."""
        var = info["file_key"]
        logger.debug("Reading in get_dataset %s.", var)
        variable = self.nc[var]

        # For the emissivity there are multiple bands, so we need to select the correct one
        if var == "camel_emis":
            if info["band_id"] >= variable.shape[2]:
                raise ValueError("Band id requested is larger than dataset.")
            variable = variable[:, :, info["band_id"]]

        # Rename the latitude and longitude dimensions to x and y
        variable = variable.rename({"latitude": "y", "longitude": "x"})

        variable.attrs.update(key.to_dict())
        return variable


    def get_area_def(self, dsid):
        """Get the area definition, a global lat/lon area for this type of dataset."""
        proj_param = "EPSG:4326"

        area = geometry.AreaDefinition("gridded_camel",
                                       "A global gridded area",
                                       "longlat",
                                       proj_param,
                                       self.ncols,
                                       self.nlines,
                                       GLOB_AREA_EXT)
        self.area = area
        return area
