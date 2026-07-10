#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016-2020 Satpy developers
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

"""SLSTR L2 reader."""

import datetime as dt
import logging

import xarray as xr

from satpy.readers.core.file_handlers import BaseFileHandler
from satpy.readers.core.slstr import PLATFORM_NAMES
from satpy.utils import get_legacy_chunk_size

logger = logging.getLogger(__name__)

CHUNK_SIZE = get_legacy_chunk_size()

class NCSLSTRL2LST(BaseFileHandler):
    """Filehandler for L2 SLSTR Land Surface Temperature (LST) data."""

    def __init__(self, filename, filename_info, filetype_info,
                 user_calibration=None):
        """Initialize the SLSTR l2 data filehandler."""
        super(NCSLSTRL2LST, self).__init__(filename, filename_info,
                                        filetype_info)

        self.nc = xr.open_dataset(self.filename,
                                  decode_cf=True,
                                  mask_and_scale=True,
                                  chunks={"columns": CHUNK_SIZE,
                                          "rows": CHUNK_SIZE})
        self.nc = self.nc.rename({"columns": "x", "rows": "y"})
        self.stripe = filename_info["stripe"]
        views = {"n": "nadir", "o": "oblique"}
        self.view = views[filename_info["view"]]

        self.platform_name = PLATFORM_NAMES[filename_info["mission_id"]]
        self.sensor = "slstr"

    def get_dataset(self, key, info):
        """Load a dataset."""
        if (self.stripe != key["stripe"].name or self.view != key["view"].name):
            return
        file_key = info["file_key"]
        logger.debug("Reading %s.", key["name"])
        data = self.nc[f"{file_key}"]

        units = data.attrs["units"]

        info = info.copy()
        info.update(data.attrs)
        info.update(key.to_dict())
        info.update(dict(units=units,
                         platform_name=self.platform_name,
                         sensor=self.sensor,
                         view=self.view))

        data.attrs = info
        return data

    @property
    def start_time(self):
        """Get the start time."""
        return dt.datetime.strptime(self.nc.attrs["start_time"], "%Y-%m-%dT%H:%M:%S.%fZ")

    @property
    def end_time(self):
        """Get the end time."""
        return dt.datetime.strptime(self.nc.attrs["stop_time"], "%Y-%m-%dT%H:%M:%S.%fZ")
