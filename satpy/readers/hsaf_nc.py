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

These products provide precipitation estimates derived from a blending of
geostationary (GEO) infrared and low-Earth-orbit (LEO) microwave observations
using the H SAF "Rapid Update" algorithm.

Currently, this reader supports the following products:
  * **H60/H60B** – Blended GEO/IR and LEO/MW instantaneous precipitation over
    the full Meteosat (0°) disk.
  * **H63** – Blended GEO/IR and LEO/MW instantaneous precipitation over the
    Meteosat IODC (Indian Ocean) disk.
  * **H90** – Accumulated Precipitation at ground by blended MW and IR IODC
    over the Meteosat IODC (Indian Ocean) disk.

Notes:
  * Externally compressed files (.gz) and uncompressed files are handled
    transparently via Satpys ``generic_open``, without creating temporary
    decompressed files on disk.
  * The reader uses area definitions specified in the associated YAML
    configuration file to determine projection, coverage, and resolution.
  * Provides access to the ``rain_rate`` and ``q_ind`` datasets (including
    accumulated variants where available).
"""

import datetime as dt
import logging

import xarray as xr

from satpy.area import get_area_def
from satpy.readers.core.file_handlers import BaseFileHandler
from satpy.readers.core.utils import generic_open
from satpy.utils import get_legacy_chunk_size

LOG = logging.getLogger(__name__)

platform_translate = {"MSG1": "Meteosat-8",
                      "MSG2": "Meteosat-9",
                      "MSG3": "Meteosat-10",
                      "MSG4": "Meteosat-11",
                      }

CHUNK_SIZE = get_legacy_chunk_size()

class HSAFNCFileHandler(BaseFileHandler):
    """Handle H SAF NetCDF files, with optional .gz external compression.

    This file handler handles H SAF Instantaneous Rain Rate products which contain:
    - rr or acc_rr: rain rate in mm/h (instantaneous) or mm (accumulated)
    - qind: quality index in percent

    The data is on a geostationary projection grid.
    This file handler is tested with H SAF h60/h60b, h63 & h90 data in NetCDF
    format.
    """

    def __init__(self, filename, filename_info, filetype_info):
        """Open the nc file and store the nc data."""
        super().__init__(filename, filename_info, filetype_info)
        self._area_name = None
        self._lon_0 = None
        self._cm = generic_open(filename, mode="rb", compression="infer")
        fp = self._cm.__enter__()
        self._nc_data = xr.open_dataset(fp, engine="h5netcdf", chunks="auto")

    def __del__(self):
        """Instruct the context manager to clean up and close the file."""
        try:
            self._cm.__exit__(None, None, None)
        except (AttributeError, RuntimeError, OSError, ValueError):
            LOG.warning(f"An error occurred while cleaning up the file {self.filename}")

    def _get_global_attributes(self):
        """Create a dictionary of global attributes to be added to all datasets."""
        attributes = {
            "filename": self.filename,
            "platform_name": platform_translate.get(
                self._nc_data.attrs["satellite_identifier"], self._nc_data.attrs["satellite_identifier"]
            ),
        }

        return attributes

    @staticmethod
    def _standarize_dims(dataset):
        """Standardize dims to y & x (what Satpy/pyresample expect), if it is needed."""
        if "y" not in dataset.dims:
            dataset = dataset.rename({"ny": "y"})
        if "x" not in dataset.dims:
            dataset = dataset.rename({"nx": "x"})

        return dataset

    def get_dataset(self, dataset_id, dataset_info):
        """Get a dataset from the file."""
        # Get the variable
        var_name = dataset_info.get("file_key", dataset_id["name"])
        LOG.debug(f"Getting dataset {var_name} from file {self.filename}")
        data = self._nc_data[var_name]

        data = self._standarize_dims(data)

        if "missing_value" in data.attrs:
            data = data.where(data != data.attrs["missing_value"])

        # Add metadata from YAML
        for key, value in dataset_info.items():
            if key not in ("file_key",):
                data.attrs[key] = value

        data.attrs["resolution"] = self.filetype_info["resolution"]
        self._area_name = self.filetype_info["area"]
        self._lon_0 = float(self._nc_data.attrs["sub_satellite_longitude"].rstrip("f"))

        data.attrs.update(self._get_global_attributes())

        return data

    @property
    def end_time(self):
        """Get end time."""
        return self.start_time + dt.timedelta(minutes=self.filetype_info["end_time_delta_min"])

    @property
    def start_time(self):
        """Get start time."""
        return self.filename_info["start_time"]

    def get_area_def(self, dsid):
        """Get the area definition of the band."""
        area_def = get_area_def(self._area_name)

        if area_def.proj_dict["lon_0"] != self._lon_0:
            LOG.warning("Subsatellite longitude does not match the `lon_0` value in the AreaDefinition projection"
                        f" '{self._area_name}'. `lon_0` will now be adapted to the value from the loaded data.")

            pdict_new = area_def.proj_dict.copy()
            pdict_new["lon_0"] = self._lon_0

            area_def = area_def.copy(projection=pdict_new)

        return area_def
