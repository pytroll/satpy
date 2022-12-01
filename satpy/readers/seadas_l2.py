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
from .netcdf_utils import NetCDF4FileHandler


class _SEADASL2Base:
    """Simple handler of SEADAS L2 files."""

    def __init__(self, filename, filename_info, filetype_info, apply_quality_flags=False):
        """Initialize file handler and determine if data quality flags should be applied."""
        super().__init__(filename, filename_info, filetype_info)
        self.apply_quality_flags = apply_quality_flags and self.l2_flags_var_name in self

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
        platform = self[self.platform_attr_name]
        platform_dict = {'NPP': 'Suomi-NPP',
                         'JPSS-1': 'NOAA-20',
                         'JPSS-2': 'NOAA-21'}
        return platform_dict.get(platform, platform)

    @property
    def start_time(self):
        """Get the starting observation time of this file's data."""
        start_time = self[self.start_time_attr_name]
        return datetime.strptime(start_time[:-3], self.time_format)

    @property
    def end_time(self):
        """Get the ending observation time of this file's data."""
        end_time = self[self.end_time_attr_name]
        return datetime.strptime(end_time[:-3], self.time_format)

    @property
    def sensor_names(self):
        """Get sensor for the current file's data."""
        # Example: MODISA or VIIRSN or VIIRSJ1
        sensor_name = self[self.sensor_attr_name].lower()
        if sensor_name.startswith("modis"):
            return {"modis"}
        return {"viirs"}

    def get_dataset(self, data_id, dataset_info):
        """Get DataArray for the specified DataID."""
        file_key, data = self._get_file_key_and_variable(data_id, dataset_info)
        data = self._filter_by_valid_min_max(data)
        data = self._rename_2d_dims_if_necessary(data)
        data = self._mask_based_on_l2_flags(data)
        for attr_name in ("standard_name", "long_name", "units"):
            val = data.attrs[attr_name]
            if val[-1] == "\x00":
                data.attrs[attr_name] = data.attrs[attr_name][:-1]
        data = self._add_satpy_metadata(data)
        return data

    def _get_file_key_and_variable(self, data_id, dataset_info):
        file_keys = dataset_info.get("file_key", data_id["name"])
        if not isinstance(file_keys, list):
            file_keys = [file_keys]
        for file_key in file_keys:
            try:
                data = self[file_key]
                return file_key, data
            except KeyError:
                continue
        raise KeyError(f"Unable to find any of the possible keys for {data_id}: {file_keys}")

    def _rename_2d_dims_if_necessary(self, data_arr):
        if data_arr.ndim != 2 or data_arr.dims == ("y", "x"):
            return data_arr
        return data_arr.rename(dict(zip(data_arr.dims, ("y", "x"))))

    def _filter_by_valid_min_max(self, data_arr):
        valid_range = self._valid_min_max(data_arr)
        data_arr = data_arr.where(valid_range[0] <= data_arr)
        data_arr = data_arr.where(data_arr <= valid_range[1])
        return data_arr

    def _valid_min_max(self, data_arr):
        try:
            return data_arr.attrs["valid_range"]
        except KeyError:
            return data_arr.attrs["valid_min"], data_arr.attrs["valid_max"]

    def _mask_based_on_l2_flags(self, data_arr):
        standard_name = data_arr.attrs.get("standard_name", "")
        if self.apply_quality_flags and not ("lon" in standard_name or "lat" in standard_name):
            l2_flags = self[self.l2_flags_var_name]
            l2_flags = self._rename_2d_dims_if_necessary(l2_flags)
            mask = (l2_flags & 0b00000000010000000000000000000000) != 0
            data_arr = data_arr.where(~mask)
        return data_arr


class SEADASL2NetCDFFileHandler(_SEADASL2Base, NetCDF4FileHandler):
    """Simple handler of SEADAS L2 NetCDF4 files."""

    start_time_attr_name = "/attr/time_coverage_start"
    end_time_attr_name = "/attr/time_coverage_end"
    time_format = "%Y-%m-%dT%H:%M:%S.%f"
    platform_attr_name = "/attr/platform"
    sensor_attr_name = "/attr/instrument"
    l2_flags_var_name = "geophysical_data/l2_flags"


class SEADASL2HDFFileHandler(_SEADASL2Base, HDF4FileHandler):
    """Simple handler of SEADAS L2 HDF4 files."""

    start_time_attr_name = "/attr/Start Time"
    end_time_attr_name = "/attr/End Time"
    time_format = "%Y%j%H%M%S"
    platform_attr_name = "/attr/Mission"
    sensor_attr_name = "/attr/Sensor Name"
    l2_flags_var_name = "l2_flags"
