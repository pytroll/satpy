#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009-2023 Satpy developers
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
"""Reading VIIRS VGAC data.

"""

import logging
from datetime import datetime

import numpy as np
import xarray as xr

from satpy import CHUNK_SIZE
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)


class VGACFileHandler(BaseFileHandler):
    """Reader VGAC data."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init the file handler."""

        super(VGACFileHandler, self).__init__(
            filename, filename_info, filetype_info)

        self.engine = "h5netcdf"
        self._start_time = filename_info['start_time']
        self._end_time = None
        self.sensor = 'viirs'
        self.filename_info = filename_info

    def convert_to_bt(self, data, data_lut, scale_factor):
        from scipy import interpolate
        x = np.arange(0, len(data_lut))
        y = data_lut
        func = interpolate.interp1d(x, y)
        brightness_temperatures = func(data.values / scale_factor)
        return brightness_temperatures

    def fix_radiances_not_in_percent(self, data):
        """Scale radiances to percent. This was not done in first version of data."""
        return 100*data

    def get_dataset(self, key, yaml_info):
        """Get dataset."""
        logger.debug("Getting data for: %s", yaml_info['name'])
        nc = xr.open_dataset(self.filename, engine=self.engine, decode_times=False,
                             chunks={'y': CHUNK_SIZE, 'x': CHUNK_SIZE})
        name = yaml_info.get('nc_store_name', yaml_info['name'])
        file_key = yaml_info.get('nc_key', name)
        data = nc[file_key]
        scale_factor = yaml_info.get("scale_factor_nc", 0.0002)
        if file_key + "_LUT" in nc:
            data.values = self.convert_to_bt(data, nc[file_key + "_LUT"], scale_factor)
        if name != yaml_info['name']:
            data = data.rename(yaml_info['name'])
        if data.attrs["units"] == "percent":
            data = self.fix_radiances_not_in_percent(data)
        data.attrs.update(nc.attrs)  # For now add global attributes to all datasets
        data.attrs.update(yaml_info)
        if "StartTime" in data.attrs:
            data.attrs["start_time"] = datetime.strptime(data.attrs["StartTime"], "%Y-%m-%dT%H:%M:%S")
            data.attrs["end_time"] = datetime.strptime(data.attrs["EndTime"], "%Y-%m-%dT%H:%M:%S")
            self._end_time = data.attrs["end_time"]
        return data

    @property
    def start_time(self):
        """Get the start time."""
        return self._start_time

    @property
    def end_time(self):
        """Get the end time."""
        return self._end_time
