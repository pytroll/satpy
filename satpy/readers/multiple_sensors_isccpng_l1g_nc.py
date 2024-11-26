# Copyright (c) 2009-2024 Satpy developers
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
"""Reader for ISCCP-NG L1G data."""

import datetime as dt
import logging

import numpy as np
import xarray as xr

from satpy.readers.file_handlers import BaseFileHandler
from satpy.utils import get_legacy_chunk_size

CHUNK_SIZE = get_legacy_chunk_size()
logger = logging.getLogger(__name__)


class IsccpngL1gFileHandler(BaseFileHandler):
    """Reader L1G ISCCP-NG data."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init the file handler."""
        super(IsccpngL1gFileHandler, self).__init__(
            filename, filename_info, filetype_info)

        self.engine = "h5netcdf"
        self._start_time = filename_info["start_time"]
        self._end_time = None
        self.sensor = "multiple_sensors"
        self.filename_info = filename_info

    def tile_geolocation(self, data, key):
        if key in "latitude":
            return xr.DataArray(np.tile(data.values[:, np.newaxis], (1, 7200)), dims=["y", "x"], attrs=data.attrs)
        if key in "longitude":
            return xr.DataArray(np.tile(data.values, (3600, 1)), dims=["y", "x"], attrs=data.attrs)
        return data
            
    def get_dataset(self, key, yaml_info):
        """Get dataset."""
        logger.debug("Getting data for: %s", yaml_info["name"])
        # res = super(IsccpngL1gFileHandler, self).get_dataset(key, yaml_info)
        nc = xr.open_dataset(self.filename, engine=self.engine,
                             chunks={"y": CHUNK_SIZE, "x": 800})
        name = yaml_info.get("nc_store_name", yaml_info["name"])
        file_key = yaml_info.get("nc_key", name)
        data = nc[file_key]
        if len(data.dims) > 2 :
            # lat = self.tile_geolocation(data.coords["latitude"], "latitude")
            # lon = self.tile_geolocation(data.coords["longitude"], "longitude")
            data = data[0, 0, :, :].squeeze(drop=True)
            data = data.drop_vars('latitude')
            data = data.drop_vars('longitude')
            data = data.drop_vars('start_time')
            data = data.drop_vars('end_time')
            data = data.rename({'longitude': 'x','latitude': 'y'})
            # data = data.assign_coords({"lat": lat, "lon": lon})
        data = self.tile_geolocation(data, file_key)
        return data

    @property
    def start_time(self):
        """Get the start time."""
        return self._start_time

    @property
    def end_time(self):
        """Get the end time."""
        return self._end_time
