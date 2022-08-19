#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Satpy developers
#
# satpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# satpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Advanced Technology Microwave Sounder (ATMS) Level 1B product reader.

The format is explained in the `ATMS L1B Product User Guide`_

.. _`ATMS L1B Product User Guide`: https://docserver.gesdisc.eosdis.nasa.gov/public/project/Sounder/ATMS_V3_L1B_Product_User_Guide.pdf

"""

import logging
from datetime import datetime

from satpy.readers.netcdf_utils import NetCDF4FileHandler


logger = logging.getLogger(__name__)

DATE_FMT = '%Y-%m-%dT%H:%M:%SZ'
ATMS_CHANNEL_NAME_TO_INDEX = {
    "1": 0,
    "2": 1,
    "3": 2,
    "4": 3,
    "5": 4,
    "6": 5,
    "7": 6,
    "8": 7,
    "9": 8,
    "10": 9,
    "11": 10,
    "12": 11,
    "13": 12,
    "14": 13,
    "15": 14,
    "16": 15,
    "17": 16,
    "18": 17,
    "19": 18,
    "20": 19,
    "21": 20,
    "22": 21,
}


class AtmsL1bNCFileHandler(NetCDF4FileHandler):
    """Reader class for ATMS L1B products in netCDF format."""

    def __init__(self, filename, filename_info, filetype_info, **kwargs):
        """Initialize file handler."""
        super().__init__(
            filename, filename_info, filetype_info, auto_maskandscale=True,
        )
        self.channel_name_to_index = ATMS_CHANNEL_NAME_TO_INDEX

    @property
    def start_time(self):
        """Get observation start time."""
        return datetime.strptime(self['/attr/time_coverage_start'], DATE_FMT)

    @property
    def end_time(self):
        """Get observation end time."""
        return datetime.strptime(self['/attr/time_coverage_end'], DATE_FMT)

    @property
    def platform_name(self):
        """Get platform name."""
        return self["/attr/platform"]

    @property
    def sensor(self):
        """Get sensor."""
        return self["/attr/instrument"]

    @property
    def antenna_temperature(self):
        """Get antenna temperature."""
        file_key = self.filetype_info["antenna_temperature"]
        return self[file_key]

    @property
    def attrs(self):
        """Return attributes."""
        return {
            "filename": self.filename,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "platform_name": self.platform_name,
            "sensor": self.sensor,
        }

    @staticmethod
    def _standardize_dims(dataset):
        """Standardize dims to y, x."""
        if "atrack" in dataset.dims:
            dataset = dataset.rename({"atrack": "y"})
        if "xtrack" in dataset.dims:
            dataset = dataset.rename({"xtrack": "x"})
        if dataset.dims[0] == "x":
            dataset = dataset.transpose("y", "x")
        return dataset

    @staticmethod
    def _drop_coords(dataset):
        """Drop coords that are not in dims."""
        for coord in dataset.coords:
            if coord not in dataset.dims:
                dataset = dataset.drop_vars(coord)
        return dataset

    def _merge_attributes(self, dataset, dataset_info):
        """Merge attributes of the dataset."""
        dataset.attrs.update(self.filename_info)
        dataset.attrs.update(dataset_info)
        dataset.attrs.update(self.attrs)
        return dataset

    def _get_channel_data(self, channel_name):
        """Get channel data."""
        idx = self.channel_name_to_index.get(channel_name)
        return self.antenna_temperature[:, :, idx]

    def get_dataset(self, dataset_id, ds_info):
        """Get dataset using file_key in dataset_info."""
        param = dataset_id['name']
        logger.debug(f'Reading in file to get dataset with key {param}.')
        try:
            dataset = (
                self[param] if param not in self.channel_name_to_index
                else self._get_channel_data(param)
            )
        except KeyError:
            logger.warning(f'Could not find {param} in NetCDF file, no valid Dataset created')  # noqa: E501
            return None
        dataset = self._merge_attributes(dataset, ds_info)
        dataset = self._drop_coords(dataset)
        return self._standardize_dims(dataset)
