#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023 Satpy developers
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
"""
Reader for GK-2B GOCI-II L2 products from NOSC.
"""

import logging
from datetime import datetime

import xarray as xr

from satpy.readers.netcdf_utils import NetCDF4FileHandler

logger = logging.getLogger(__name__)


class GOCI2L2NCFileHandler(NetCDF4FileHandler):
    """File handler for GOCI-II L2 official data in netCDF format."""

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the reader."""
        super().__init__(filename, filename_info, filetype_info)
        self.slot = filename_info.get("slot", None)

        self.attrs = self["/attrs"]
        self.nc = self._merge_navigation_data(filetype_info)

        self.sensor = self.attrs["instrument"].lower()
        self.nlines = self.nc.sizes["number_of_lines"]
        self.ncols = self.nc.sizes["pixels_per_line"]
        self.platform_shortname = filename_info["platform"]
        self.coverage = filename_info["coverage"]

    def _merge_navigation_data(self, filetype_info):
        """Merge navigation data and geophysical data."""
        navigation = self["navigation_data"]
        if filetype_info["file_type"] == "goci_l2_ac":
            Rhoc = self["geophysical_data/RhoC"]
            Rrs = self["geophysical_data/Rrs"]
            data = xr.merge([Rhoc, Rrs, navigation])
        else:
            data = xr.merge([self["geophysical_data"], navigation])
        return data

    @property
    def start_time(self):
        """Start timestamp of the dataset."""
        dt = self.attrs["observation_start_time"]
        return datetime.strptime(dt, "%Y%m%d_%H%M%S")

    @property
    def end_time(self):
        """End timestamp of the dataset."""
        dt = self.attrs["observation_end_time"]
        return datetime.strptime(dt, "%Y%m%d_%H%M%S")

    def get_dataset(self, key, info):
        """Load a dataset."""
        var = info["file_key"]
        logger.debug("Reading in get_dataset %s.", var)
        variable = self.nc[var]

        # Data has 'Latitude' and 'Longitude' coords, these must be replaced.
        variable = variable.rename({"number_of_lines": "y", "pixels_per_line": "x"})

        variable.attrs.update(key.to_dict())
        return variable
