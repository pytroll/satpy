#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 Satpy developers
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
"""Reader for GPM imerg data on half-hourly timesteps.

References:
   - The NASA GPM L2 Data:
     https://gpm.nasa.gov/data-access/downloads/gpm

"""

import logging
import h5py
import numpy as np
from datetime import datetime
from satpy.readers.hdf5_utils import HDF5FileHandler

logger = logging.getLogger(__name__)


class HDF_GPM_L2(HDF5FileHandler):
    """GPM L2 hdf5 reader."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init method."""
        super(HDF_GPM_L2, self).__init__(filename, filename_info,
                                         filetype_info)
        self.finfo = filename_info
        self.cache = {}

    @property
    def start_time(self):
        """Find the start time from filename info."""
        return datetime(self.finfo["date"].year,
                        self.finfo["date"].month,
                        self.finfo["date"].day,
                        self.finfo["start_time"].hour,
                        self.finfo["start_time"].minute,
                        self.finfo["start_time"].second)

    @property
    def end_time(self):
        """Find the end time from filename info."""
        return datetime(self.finfo["date"].year,
                        self.finfo["date"].month,
                        self.finfo["date"].day,
                        self.finfo["end_time"].hour,
                        self.finfo["end_time"].minute,
                        self.finfo["end_time"].second)

    def available_datasets(self, configured_datasets=None):
        """Automatically determine datasets provided by this file."""
        logger.debug("Available_datasets begin...")
        handled_variables = set()

        # update previously configured datasets
        logger.debug("Starting previously configured variables loop...")
        for is_avail, ds_info in (configured_datasets or []):
            # some other file handler knows how to load this
            if is_avail is not None:
                yield is_avail, ds_info
            var_name = ds_info.get("file_key", ds_info["name"])

        # get lon and lat in different groups first
        for var_name, val in self.file_content.items():
            if isinstance(val, h5py._hl.dataset.Dataset):
                if ("Longitude" in var_name) or \
                    ("Latitude" in var_name):
                    logger.debug("Evaluating new variable: %s", var_name)
                    var_shape = self[var_name + "/shape"]
                    logger.debug("Dims:{}".format(var_shape))
                    logger.debug("Found valid additional dataset: %s", var_name)

                    handled_variables.add(var_name)
                    coordinates = [var_name.split('/')[0]+"/Longitude",
                                   var_name.split('/')[0]+"/Latitude"]

                    new_info = {
                        "name": var_name,
                        "standard_name": var_name.split('/')[-1].lower(),
                        "file_key": var_name,
                        "file_type": self.filetype_info["file_type"],
                        "instrument": self.finfo["instrument"],
                        "coordinates": coordinates,
                        "resolution": None,
                    }
                    yield True, new_info

        # Iterate over dataset contents
        for var_name, val in self.file_content.items():
            # Only evaluate variables
            if isinstance(val, h5py._hl.dataset.Dataset):
                logger.debug("Evaluating new variable: %s", var_name)
                var_shape = self[var_name + "/shape"]
                logger.debug("Dims:{}".format(var_shape))
                logger.debug("Found valid additional dataset: %s", var_name)

                # Skip anything we have already configured
                if (var_name in handled_variables):
                    logger.debug("Already handled, skipping: %s", var_name)
                    continue
                handled_variables.add(var_name)

                # Deal with same variables in multiple groups
                if "/" in var_name:
                    first_index_separator = var_name.index("/")
                    last_index_separator = var_name.rindex("/")
                    # We need to keep the group name
                    #   because of same varnames in different groups
                    var_name_no_path = var_name[:first_index_separator] + var_name[last_index_separator:]
                else:
                    var_name_no_path = var_name

                # assign coordinates only for 2D array
                if len(self[var_name + "/shape"]) == 2:
                    coordinates = [var_name.split('/')[0]+"/Longitude",
                                   var_name.split('/')[0]+"/Latitude"]
                else:
                    coordinates = []

                new_info = {
                    "name": var_name_no_path,
                    "file_key": var_name,
                    "file_type": self.filetype_info["file_type"],
                    "instrument": self.finfo["instrument"],
                    "coordinates": coordinates,
                    "resolution": None,
                }
                yield True, new_info

    def get_dataset(self, ds_id, ds_info):
        """Load a dataset."""
        file_key = ds_info.get("file_key", ds_id.name)
        data = self[file_key]

        # mask fill value
        fill = data.attrs["_FillValue"]
        data = data.squeeze()
        data = data.where(data != fill, np.float32(np.nan))

        # get attributes
        data.attrs.update(ds_info)
        # remove attributes that could be confusing later
        data.attrs.pop("Units", None)

        # convert bytes to string
        for attr_key in data.attrs:
            if isinstance(data.attrs[attr_key], bytes):
                data.attrs[attr_key] = data.attrs[attr_key].decode("utf-8")

        return data
