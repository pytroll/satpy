#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 Satpy developers
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

"""ACSPO SST Reader.

See the following page for more information:

https://podaac.jpl.nasa.gov/dataset/VIIRS_NPP-OSPO-L2P-v2.3

Cloud clearing and data filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Cloud clearing can be enabled for supported variables (ex. "sst") by passing
the filter name "cloud_clear" to the file handler "filters" keyword argument::

   scn = Scene(reader="ascpo", filenames=[...],
               reader_kwargs={"filters": ["cloud_clear"]})

.. versionchanged: 0.58.0

   The cloud clearing is no longer enabled by default.

"""

import datetime as dt
import logging

import numpy as np

from satpy.readers.core.netcdf import NetCDF4FileHandler

LOG = logging.getLogger(__name__)


ROWS_PER_SCAN = {
    "modis": 10,
    "viirs": 16,
    "avhrr": None,
}


class ACSPOFileHandler(NetCDF4FileHandler):
    """ACSPO L2P SST File Reader."""

    def __init__(self, filename, filename_info, filetype_info,
                 filters: list[str] | None = None, **kwargs):
        """Initialize file handler and store cloud clear flag."""
        super().__init__(filename, filename_info, filetype_info, **kwargs)
        filters = filters or []
        for filter_name in filters:
            if filter_name not in ("cloud_clear",):
                raise ValueError(f"Unknown filter '{filter_name}'")
        self.cloud_clear = "cloud_clear" in filters

    @property
    def platform_name(self):
        """Get satellite name for this file's data."""
        res = self["/attr/platform"]
        if isinstance(res, np.ndarray):
            return str(res.astype(str))
        return res

    @property
    def sensor_name(self):
        """Get instrument name for this file's data."""
        res = self["/attr/sensor"]
        if isinstance(res, np.ndarray):
            res = str(res.astype(str))
        return res.lower()

    def get_shape(self, ds_id, ds_info):
        """Get numpy array shape for the specified dataset.

        Args:
            ds_id (DataID): ID of dataset that will be loaded
            ds_info (dict): Dictionary of dataset information from config file

        Returns:
            tuple: (rows, cols)

        """
        var_path = ds_info.get("file_key", "{}".format(ds_id["name"]))
        if var_path + "/shape" not in self:
            # loading a scalar value
            shape = 1
        else:
            shape = self[var_path + "/shape"]
            if len(shape) == 3:
                if shape[0] != 1:
                    raise ValueError("Not sure how to load 3D Dataset with more than 1 time")
                shape = shape[1:]
        return shape

    @staticmethod
    def _parse_datetime(datestr):
        return dt.datetime.strptime(datestr, "%Y%m%dT%H%M%SZ")

    @property
    def start_time(self):
        """Get first observation time of data."""
        return self._parse_datetime(self["/attr/time_coverage_start"])

    @property
    def end_time(self):
        """Get final observation time of data."""
        return self._parse_datetime(self["/attr/time_coverage_end"])

    def get_metadata(self, dataset_id, ds_info):
        """Collect various metadata about the specified dataset."""
        var_path = ds_info.get("file_key", "{}".format(dataset_id["name"]))
        shape = self.get_shape(dataset_id, ds_info)
        units = self[var_path + "/attr/units"]
        info = getattr(self[var_path], "attrs", {})
        standard_name = self[var_path + "/attr/standard_name"]
        resolution = float(self["/attr/spatial_resolution"].split(" ")[0])
        rows_per_scan = ROWS_PER_SCAN.get(self.sensor_name) or 0
        info.update(dataset_id.to_dict())
        info.update({
            "shape": shape,
            "units": units,
            "platform_name": self.platform_name,
            "sensor": self.sensor_name,
            "standard_name": standard_name,
            "resolution": resolution,
            "rows_per_scan": rows_per_scan,
            "long_name": self.get(var_path + "/attr/long_name"),
            "comment": self.get(var_path + "/attr/comment"),
        })
        return info

    def get_dataset(self, dataset_id, ds_info):
        """Load data array and metadata from file on disk."""
        var_path = ds_info.get("file_key", "{}".format(dataset_id["name"]))
        metadata = self.get_metadata(dataset_id, ds_info)
        shape = metadata["shape"]
        file_shape = self[var_path + "/shape"]
        metadata["shape"] = shape

        valid_min = self[var_path + "/attr/valid_min"]
        valid_max = self[var_path + "/attr/valid_max"]
        # no need to check fill value since we are using valid min/max
        scale_factor = self.get(var_path + "/attr/scale_factor")
        add_offset = self.get(var_path + "/attr/add_offset")

        data = self[var_path]
        data = data.rename({"ni": "x", "nj": "y"})
        if isinstance(file_shape, tuple) and len(file_shape) == 3:
            # can only read 3D arrays with size 1 in the first dimension
            data = data[0]
        data = data.where((data >= valid_min) & (data <= valid_max))
        if scale_factor is not None:
            data = data * scale_factor + add_offset

        if self.cloud_clear and ds_info.get("cloud_clear", False):
            LOG.info(f"Cloud clearing {dataset_id}")
            # clear-sky if bit 15-16 are 00
            l2p_flags = self._get_unsigned_l2p_flags()
            clear_sky_mask = (l2p_flags & 0b1100000000000000) != 0
            clear_sky_mask = clear_sky_mask.rename({"ni": "x", "nj": "y"})
            data = data.where(~clear_sky_mask)

        data.attrs.update(metadata)
        # Remove these attributes since they are no longer valid and can cause invalid value filling.
        data.attrs.pop("_FillValue", None)
        data.attrs.pop("valid_max", None)
        data.attrs.pop("valid_min", None)
        return data

    def _get_unsigned_l2p_flags(self):
        l2p_flags = self["l2p_flags"][0]
        # l2p_flags is usually signed 16-bit (int16) but we need (uint16) for binary operations
        unsigned_type = l2p_flags.dtype.str.replace("i", "u")
        return l2p_flags.astype(unsigned_type, copy=False)
