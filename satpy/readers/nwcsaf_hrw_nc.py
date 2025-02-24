#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2025- Satpy developers
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
"""NWC SAF GEO v2021 HRW.

This is a reader for the High Resolution Winds (HRW) data produced with NWC SAF GEO.

"""

import datetime as dt
import logging
from contextlib import suppress

import dask.array as da
import h5py
import numpy as np
import xarray as xr

from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.nwcsaf_nc import PLATFORM_NAMES, SENSOR, read_nwcsaf_time
from satpy.utils import get_chunk_size_limit

logger = logging.getLogger(__name__)

CHUNK_SIZE = get_chunk_size_limit()

WIND_CHANNELS = [
    "wind_hrvis",
    "wind_ir108",
    "wind_ir120",
    "wind_vis06",
    "wind_vis08",
    "wind_wv062",
    "wind_wv073",
]

class NWCSAFGEOHRWFileHandler(BaseFileHandler):
    """A file handler class for NWC SAF GEO HRW files."""

    def __init__(self, filename, filename_info, filetype_info, merge_channels=False):
        """Initialize the file handler."""
        super().__init__(filename, filename_info, filetype_info)

        self.h5f = h5py.File(self.filename, "r")
        self.filename_info = filename_info
        self.filetype_info = filetype_info
        self.merge_channels = merge_channels
        self.platform_name = PLATFORM_NAMES.get(self.h5f.attrs["satellite_identifier"].astype(str))
        self.sensor = SENSOR.get(self.platform_name, "seviri")
        self.lons = {}
        self.lats = {}
        # Imaging period, which is set after reading any data, and used to calculate end time
        self.period = None

        # The resolution is given in kilometers, convert to meters
        self.resolution = 1000 * self.h5f.attrs["spatial_resolution"].item()

    def __del__(self):
        """Close file handlers when we are done."""
        with suppress(OSError):
            self.h5f.close()

    def available_datasets(self, configured_datasets=None):
        """Form the names for the available datasets."""
        for channel in WIND_CHANNELS:
            prefix = channel + "_"
            if self.merge_channels:
                prefix = ""
            dset = self.h5f[channel]
            for measurand in dset.dtype.fields.keys():
                if measurand == "trajectory":
                    continue
                ds_info = {
                    "file_type": self.filetype_info["file_type"],
                    "resolution": self.resolution,
                    "name": prefix + measurand,
                }
                if measurand not in ("longitude", "latitude"):
                    ds_info["coordinates"] = (prefix + "longitude", prefix + "latitude")
                if measurand == "longitude":
                    ds_info["standard_name"] = "longitude"
                if measurand == "latitude":
                    ds_info["standard_name"] = "latitude"
                yield True, ds_info
            if self.merge_channels:
                break

    def get_dataset(self, key, info):
        """Load a dataset."""
        logger.debug("Reading %s.", key["name"])
        if self.merge_channels:
            data = self._read_merged_dataset(key, info)
        else:
            data = self._read_dataset(key, info)
        data.attrs.update(info)

        return data

    def _read_merged_dataset(self, dataset_key, info):
        """Read a dataset merged from every channel."""
        dataset_name = dataset_key["name"]

        data = []
        collect_coords = True
        if "merged" in self.lons:
            collect_coords = False
        for channel in WIND_CHANNELS:
            if collect_coords:
                self._read_channel_coordinates(channel)
                self._append_merged_coordinates(channel)
            if self.period is None:
                self.period = self.h5f[channel].attrs["time_period"].item()
            try:
                data.append(self.h5f[channel][dataset_name])
            except ValueError:
                logger.warning("Reading %s is not supported.", dataset_name)

        return self._create_xarray(
            np.concat(data),
            dataset_name,
            np.concat(self.lons["merged"]),
            np.concat(self.lats["merged"]),
            info)

    def _append_merged_coordinates(self, channel):
        if "merged" not in self.lons:
            self.lons["merged"] = []
            self.lats["merged"] = []
        self.lons["merged"].append(self.lons[channel])
        self.lats["merged"].append(self.lats[channel])

    @staticmethod
    def _create_xarray(data, dataset_name, lons, lats, info, prefix=""):
        xr_data = xr.DataArray(da.from_array(data, chunks=CHUNK_SIZE),
                               name=dataset_name,
                               dims=["y"])
        xr_data[prefix + "longitude"] = ("y", lons)
        xr_data[prefix + "latitude"] = ("y", lats)
        xr_data.attrs.update(info)

        return xr_data

    def _read_dataset(self, dataset_key, info):
        """Read a dataset."""
        dataset_name = dataset_key["name"]
        key_parts = dataset_name.split("_")
        channel = "_".join(key_parts[:2])
        self._read_channel_coordinates(channel)
        if self.period is None:
            self.period = self.h5f[channel].attrs["time_period"].item()
        measurand = "_".join(key_parts[2:])
        try:
            data = self.h5f[channel][measurand]
        except ValueError:
            logger.warning("Reading %s is not supported.", dataset_name)

        prefix = channel + "_"

        return self._create_xarray(data, dataset_name, self.lons[channel], self.lats[channel], info, prefix=prefix)


    def _read_channel_coordinates(self, channel):
        if channel not in self.lons:
            self.lons[channel] = self.h5f[channel]["longitude"]
            self.lats[channel] = self.h5f[channel]["latitude"]

    @property
    def start_time(self):
        """Get the start time."""
        return read_nwcsaf_time(self.h5f.attrs["nominal_product_time"])

    @property
    def end_time(self):
        """Get the end time."""
        if self.period is None:
            return self.start_time
        return self.start_time + dt.timedelta(minutes=self.period)
