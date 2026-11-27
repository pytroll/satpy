#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2022-2023 Satpy developers
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

"""Reader for the EMIT L1B NetCDF data."""

import logging
from datetime import datetime

import numpy as np
from pyresample.geometry import SwathDefinition
from satpy.readers.netcdf_utils import NetCDF4FileHandler
from satpy.utils import get_legacy_chunk_size

logger = logging.getLogger(__name__)
CHUNK_SIZE = get_legacy_chunk_size()

DATE_FMT = "%Y-%m-%dT%H:%M:%S%z"


class EMITL1BFileHandler(NetCDF4FileHandler):
    """File handler for EMIT L1B netCDF files."""

    def __init__(self, filename, filename_info, filetype_info, *req_fhs):
        """Prepare the class for dataset reading."""
        super().__init__(filename, filename_info, filetype_info)
        self.area = None

        self._load_bands()

    def _load_bands(self):
        """Load bands data and attributes."""
        if self.filetype_info["file_type"] == "emit_l1b_rad":
            bands_name = "wavelengths"
        elif self.filetype_info["file_type"] == "emit_l1b_obs":
            bands_name = "observation_bands"

        self.bands = self[f"sensor_band_parameters/{bands_name}"].rename("bands")

        # add other parameters as coords of bands
        if self.filetype_info["file_type"] == "emit_l1b_rad":
            # add wavelength
            self.bands.coords["fwhm"] = self["sensor_band_parameters/fwhm"]

    @property
    def start_time(self):
        """Get observation start time."""
        return datetime.strptime(self["/attr/time_coverage_start"], DATE_FMT).replace(tzinfo=None)

    @property
    def end_time(self):
        """Get observation end time."""
        return datetime.strptime(self["/attr/time_coverage_end"], DATE_FMT).replace(tzinfo=None)

    @property
    def platform_name(self):
        """Get platform name."""
        return self["/attr/platform"]

    @property
    def sensor(self):
        """Get sensor."""
        return self["/attr/instrument"]

    @property
    def spatial_ref(self):
        """Get spatial_ref."""
        return self["/attr/spatial_ref"]

    @property
    def geotransform(self):
        """Get geotransform."""
        return self["/attr/geotransform"]

    @property
    def attrs(self):
        """Return attributes."""
        return {
            "filename": self.filename,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "platform_name": self.platform_name,
            "sensor": self.sensor,
            "geotransform": self.geotransform,
            "spatial_ref": self.spatial_ref,
        }

    def _standardize_dims(self, dataset):
        """Standardize dims to y, x."""
        if "downtrack" in dataset.dims:
            dataset = dataset.rename({"downtrack": "y"})
        if "crosstrack" in dataset.dims:
            dataset = dataset.rename({"crosstrack": "x"})
        if dataset.dims[0] == "x":
            dataset = dataset.transpose("y", "x")
        if "bands" in dataset.dims:
            dataset.coords["bands"] = self.bands
            dataset = dataset.transpose("bands", ...)

        return dataset

    def get_metadata(self, dataset, ds_info):
        """Get metadata."""
        metadata = {}
        metadata.update(dataset.attrs)
        metadata.update(ds_info)
        metadata.update(self.attrs)

        return metadata

    def get_dataset(self, dataset_id, ds_info):
        """Get dataset."""
        name = dataset_id["name"]
        file_type = self.filetype_info["file_type"]

        if ds_info["nc_group"] is None:
            var_path = name
        else:
            var_path = ds_info["nc_group"] + "/" + name

        if file_type == "emit_l1b_obs":
            # because the "obs" DataArray includes the varname as the "bands" dim
            #   we need to reset the var_path to "obs"
            var_path = 'obs'

        logger.debug(f"Reading in file to get dataset with path {var_path}.")
        dataset = self[var_path]

        dataset.attrs = self.get_metadata(dataset, ds_info)
        fill_value = dataset.attrs.get('_FillValue', np.float32(np.nan))

        # preserve integer data types if possible
        if np.issubdtype(dataset.dtype, np.integer):
            new_fill = fill_value
        else:
            new_fill = np.float32(np.nan)
            dataset.attrs.pop('_FillValue', None)

        # mask data by fill_value
        good_mask = dataset != fill_value
        dataset = dataset.where(good_mask, new_fill)

        # standardize dims
        dataset = self._standardize_dims(dataset)

        # only load the variable by selecting bands dim
        if file_type == "emit_l1b_obs":
            dataset = dataset.sel(bands=ds_info['long_name']).rename(name)

        # add area
        self.get_lonlats()
        dataset.attrs['area'] = self.area

        return dataset

    def get_lonlats(self):
        """Get lonlats."""
        if self.area is None:
            lons = self['location/lon']
            lats = self['location/lat']
            self.area = SwathDefinition(lons, lats)
            self.area.name = '_'.join([self.sensor, str(self.start_time)])
