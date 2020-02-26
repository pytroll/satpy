#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009-2019 Satpy developers
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
Reader for FIDUCEO GAC AVHRR L1c Easy FCDR V1.00 NetCDF file.

Authors: Abhay Devasthale and Nina Håkansson, SMHI

"""

import logging
import datetime
import xarray as xr
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)


class FiduceoFileHandler(BaseFileHandler):
    """Fiduceo file class."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init defined."""
        super(FiduceoFileHandler, self).__init__(filename, filename_info, filetype_info)
        """Super defined."""

        self.nc = None
        self._read_file()

    @property
    def start_time(self):
        """Return start time."""
        return self.starttime

    @property
    def end_time(self):
        """Return end time."""
        return self.endtime

    def _read_file(self):
        """Read fiduceo data file."""
        if self.nc is None:
            self.nc = xr.open_dataset(self.filename,
                                      decode_cf=True,
                                      mask_and_scale=True,
                                      chunks={})
            self.satellite_name = self.nc.attrs['platform']
            self.gac_file = self.nc.attrs['source']
            f_time = self.nc['Time'].values[:]
            self.starttime = datetime.datetime.utcfromtimestamp(f_time[0])
            self.endtime = datetime.datetime.utcfromtimestamp(f_time[-1])

    def get_dataset(self, dataset_id, dataset_info):
        """Get dataset for a given key."""
        dataset = self.nc[dataset_info['nc_key']]
        dataset.attrs.update(dataset_info)
        dataset.attrs.update(self.nc.attrs)
        return dataset
