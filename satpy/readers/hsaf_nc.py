#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019.
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

"""Reader for H SAF blended precipitation products in NetCDF format.

These products provide instantaneous precipitation rate estimates derived from
a blending of geostationary (GEO) infrared and low-Earth-orbit (LEO) microwave
observations using the H SAF "Rapid Update" algorithm.

Currently, this reader supports the following products:
  * **H60B** – Blended GEO/IR and LEO/MW instantaneous precipitation over the
    full Meteosat (0°) disk.
  * **H63** – Blended GEO/IR and LEO/MW instantaneous precipitation over the
    Meteosat IODC (Indian Ocean) disk.
  * **H90** – Accumulated Precipitation at ground by blended MW and IR IODC
    over the Meteosat IODC (Indian Ocean) disk.

Notes:
    - Externally compressed files with the ``.gz`` suffix are automatically
      uncompressed to the same directory and deleted on reader close.
    - The reader relies on area definitions provided in the accompanying YAML
      configuration file.
    - Supports ``rain_rate`` and ``q_ind#`` datasets.
    - Output coverage and resolution correspond to the MSG-SEVIRI grid.
"""

import datetime as dt
import gzip
import logging
import os
import shutil
from pathlib import Path

import xarray as xr
from pyresample import AreaDefinition

from satpy.area import get_area_def
from satpy.readers.core.file_handlers import BaseFileHandler
from satpy.utils import get_legacy_chunk_size

LOG = logging.getLogger(__name__)

def gunzip(source, destination):
    """Unzips an externally compressed HSAF file."""
    with gzip.open(source) as s:
        with open(destination, "wb") as d:
            shutil.copyfileobj(s, d, 10 * 1024 * 1024)

def create_named_empty_file(source):
    """Create an empty file."""
    # Use delete=False for cross-platform safety
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".nc")
    tmp.close()  # we only needed the filename
    LOG.debug(f"Created temporary file {Path(tmp.name)} from {source}")
    return Path(tmp.name)


class HSAFFileWrapper:
    """Wrapper for a H SAF NetCDF file for handling external compression if an external gzip layer exist."""

    def __init__(self, filename):
        """Opens the nc file and stores the nc data."""
        self.filename = filename
        self._tmp_file = None
        self._compressed = Path(self.filename).suffix == ".gz"

        if self._compressed:
            self._tmp_file = create_named_empty_file(self.filename)
            gunzip(self.filename, self._tmp_file)
            self.filename = str(self._tmp_file)

        # Open dataset
        chunks = get_legacy_chunk_size()
        LOG.debug(f"Opening HSAF file {self.filename}")
        self.nc_data = xr.open_dataset(self.filename, decode_cf=True, chunks=chunks)


    def close(self):
        """Close the nc file and clean up temp file if needed."""
        if self.nc_data is not None:
            self.nc_data.close()
            self.nc_data = None
            LOG.debug("Closed NetCDF dataset")

        if self._compressed and self._tmp_file is not None:
            try:
                self._tmp_file.unlink(missing_ok=True)
                LOG.debug(f"Deleted temporary file {self._tmp_file}")
            except Exception as e:
                LOG.warning(f"Failed to delete temp file {self._tmp_file}: {e}")
            finally:
                self._tmp_file = None

def _resolve_area(area_value):
    """Resolve an area definition from corresponding string value."""
    if isinstance(area_value, AreaDefinition):
        resolved_area = area_value
    elif isinstance(area_value, str):
        try:
            resolved_area = get_area_def(area_value)
        except Exception:
            # if resolution fails, keep the original string (fallback)
            LOG.warning(f"Area value {area_value} could not be resolved to an AreaDefinition")
            resolved_area = area_value
    else:
        # None or unexpected type, keep as-is
        LOG.warning(f"Area value {area_value} is neither a str nor an AreaDefinition")
        resolved_area = area_value

    return resolved_area


class HSAFNCFileHandler(BaseFileHandler):
    """Handle H SAF NetCDF files, with optional .gz external compression.

    This file handler handles H SAF Instantaneous Rain Rate products which contain:
    - rr: rain rate in mm/h
    - qind: quality index in percent

    The data is on a geostationary projection grid.
    This file handler is tested with H SAF h60, h63 & h40 data in NetCDF format.
    """

    def __init__(self, filename, filename_info, filetype_info):
        """Create a wrapper to opens the nc file and store the nc data."""
        super().__init__(filename, filename_info, filetype_info)
        self._wrapper = HSAFFileWrapper(filename)

    def close(self):
        """Clean up by closing the dataset."""
        self._wrapper.close()

    def _get_global_attributes(self):
        """Create a dictionary of global attributes to be added to all datasets."""
        attributes = {
            "filename": self.filename,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "spacecraft_name": self._wrapper.nc_data.attrs["satellite_identifier"],
            "platform_name": self._wrapper.nc_data.attrs["satellite_identifier"],
        }

        return attributes

    @staticmethod
    def _standarize_dims(dataset):
        """Standarize dims to y & x (what Satpy/pyresample expect), if it is needed."""
        if "y" not in dataset.dims:
            dataset = dataset.rename({"ny": "y"})
        if "x" not in dataset.dims:
            dataset = dataset.rename({"nx": "x"})

        return dataset

    def get_dataset(self, dataset_id, dataset_info):
        """Get a dataset from the file."""
        # Get the variable
        var_name = dataset_info.get("file_key", dataset_id["name"])
        LOG.debug(f"Getting dataset {var_name} from file {self._wrapper.filename}")
        data = self._wrapper.nc_data[var_name]

        data = self._standarize_dims(data)

        # Handle missing values
        if "missing_value" in data.attrs:
            data = data.where(data != data.attrs["missing_value"])

        # Add metadata from YAML
        for key, value in dataset_info.items():
            if key == "area":
                data.attrs[key] = _resolve_area(value)
            elif key not in ("file_key",):
                data.attrs[key] = value

        # Add global attributes which are shared across datasets
        data.attrs.update(self._get_global_attributes())

        return data

    def combine_info(self, all_infos):
        """Override super class method to prevent AreaDefinitions being replaced with SwathDefinitions."""
        # Default combination (times, orbits, etc.)
        combined_info = super().combine_info(all_infos)

        # If 'area' is already an AreaDefinition, keep it
        if "area" in all_infos[0] and hasattr(all_infos[0]["area"], "area_id"):
            combined_info["area"] = all_infos[0]["area"]

        return combined_info

    @property
    def end_time(self):
        """Get end time."""
        return self.start_time + dt.timedelta(minutes=15)

    @property
    def start_time(self):
        """Get start time."""
        return self.filename_info["start_time"]
