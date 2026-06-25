#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Satpy developers

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Reader for the ESA SMAP L2 soil moisture data."""


import h5py
import xarray as xr

from satpy.readers.file_handlers import BaseFileHandler


class SMAPFileHandler(BaseFileHandler):
    """File handler for the SMAP Soil Moisture hdf5 product."""

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the SMAP File handler."""
        super().__init__(filename, filename_info, filetype_info)
        self.h5 = None

    @property
    def start_time(self):
        """Return the start time of data acqusition."""
        return self.filename_info["start_time"]

    def get_dataset(self, dataset_id, dataset_info):
        """Get the SMAP L2 data."""
        if self.h5 is None:
            self.h5 = h5py.File(self.filename)
        dataset = xr.DataArray(self.h5[dataset_info["file_key"]],dims=["y"])
        dataset = dataset.where(dataset>-9.)
        dataset.attrs.update(dataset_info)
        return dataset
