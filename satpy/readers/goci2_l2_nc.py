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

"""Reader for GK-2B GOCI-II L2 products from NOSC.

For more information about the data, see: <https://www.nosc.go.kr/eng/boardContents/actionBoardContentsCons0028.do>
"""

import datetime as dt
import logging

import xarray as xr

from satpy.readers.netcdf_utils import NetCDF4FileHandler

logger = logging.getLogger(__name__)

GROUPS_MAP = {
    "goci2_l2_ac": ["geophysical_data/RhoC", "geophysical_data/Rrs", "navigation_data"],
    "goci2_l2_iop": [
        "geophysical_data/a_total",
        "geophysical_data/bb_total",
        "navigation_data",
    ],
}


class GOCI2L2NCFileHandler(NetCDF4FileHandler):
    """File handler for GOCI-II L2 official data in netCDF format."""

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the reader."""
        super().__init__(filename, filename_info, filetype_info)

        self.attrs = self["/attrs"]
        self.nc = self._merge_navigation_data(filetype_info["file_type"])

        # Read metadata which are common to all datasets
        self.nlines = self.nc.sizes["number_of_lines"]
        self.ncols = self.nc.sizes["pixels_per_line"]
        self.coverage = filename_info["coverage"]

    def _merge_navigation_data(self, filetype):
        """Merge navigation data and geophysical data."""
        if filetype in GROUPS_MAP.keys():
            groups = GROUPS_MAP[filetype]
        else:
            groups = ["geophysical_data", "navigation_data"]
        return xr.merge([self[group] for group in groups])

    @property
    def start_time(self):
        """Start timestamp of the dataset."""
        date_str = self.attrs["observation_start_time"]
        return dt.datetime.strptime(date_str, "%Y%m%d_%H%M%S")

    @property
    def end_time(self):
        """End timestamp of the dataset."""
        date_str = self.attrs["observation_end_time"]
        return dt.datetime.strptime(date_str, "%Y%m%d_%H%M%S")

    def get_dataset(self, key, info):
        """Load a dataset."""
        var = info["file_key"]
        logger.debug("Reading in get_dataset %s.", var)
        variable = self.nc[var]

        variable = variable.rename({"number_of_lines": "y", "pixels_per_line": "x"})

        # Some products may miss lon/lat standard_name, use name as base name if it is not already present
        if variable.attrs.get("standard_name", None) is None:
            variable.attrs.update({"standard_name": variable.name})

        variable.attrs.update(key.to_dict())
        return variable
