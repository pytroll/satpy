#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2021 Satpy developers
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
"""Reader for SEADAS L2 products.

This reader currently only supports MODIS and VIIRS Chlorophyll A from SEADAS.

The reader includes an additional keyword argument ``apply_quality_flags``
which can be used to mask out low-quality pixels based on quality flags
contained in the file (``l2_flags``). This option defaults to ``False``, but
when set to ``True`` the "CHLWARN" pixels of the ``l2_flags`` variable
are masked out. These pixels represent data where the chlorophyll algorithm
warned about the quality of the result.

"""

from datetime import datetime

from .hdf4_utils import HDF4FileHandler

TIME_FORMAT = "%Y%j%H%M%S"


class SEADASL2HDFFileHandler(HDF4FileHandler):
    """Simple handler of SEADAS L2 files."""

    def __init__(self, filename, filename_info, filetype_info, apply_quality_flags=False):
        """Initialize file handler and determine if data quality flags should be applied."""
        super().__init__(filename, filename_info, filetype_info)
        self.apply_quality_flags = apply_quality_flags and "l2_flags" in self

    def _add_satpy_metadata(self, data):
        data.attrs["sensor"] = self.sensor_names
        data.attrs["platform_name"] = self._platform_name()
        data.attrs["rows_per_scan"] = self._rows_per_scan()
        return data

    def _rows_per_scan(self):
        if "modis" in self.sensor_names:
            return 10
        if "viirs" in self.sensor_names:
            return 16
        raise ValueError(f"Don't know how to read data for sensors: {self.sensor_names}")

    def _platform_name(self):
        platform = self["/attr/Mission"]
        platform_dict = {'NPP': 'Suomi-NPP',
                         'JPSS-1': 'NOAA-20',
                         'JPSS-2': 'NOAA-21'}
        return platform_dict.get(platform, platform)

    @property
    def start_time(self):
        """Get the starting observation time of this file's data."""
        start_time = self["/attr/Start Time"]
        return datetime.strptime(start_time[:-3], TIME_FORMAT)

    @property
    def end_time(self):
        """Get the ending observation time of this file's data."""
        end_time = self["/attr/End Time"]
        return datetime.strptime(end_time[:-3], TIME_FORMAT)

    @property
    def sensor_names(self):
        """Get sensor for the current file's data."""
        # Example: MODISA or VIIRSN or VIIRSJ1
        sensor_name = self["/attr/Sensor Name"].lower()
        if sensor_name.startswith("modis"):
            return {"modis"}
        return {"viirs"}

    def get_dataset(self, data_id, dataset_info):
        """Get DataArray for the specified DataID."""
        file_key = dataset_info.get("file_key", data_id["name"])
        data = self[file_key]
        valid_range = data.attrs["valid_range"]
        data = data.where(valid_range[0] <= data)
        data = data.where(data <= valid_range[1])
        if self.apply_quality_flags and not ("lon" in file_key or "lat" in file_key):
            l2_flags = self["l2_flags"]
            mask = (l2_flags & 0b00000000010000000000000000000000) != 0
            data = data.where(~mask)
        for attr_name in ("standard_name", "long_name", "units"):
            val = data.attrs[attr_name]
            if val[-1] == "\x00":
                data.attrs[attr_name] = data.attrs[attr_name][:-1]
        data = self._add_satpy_metadata(data)
        return data
