#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2011-2015 Satpy developers
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

"""Interface to the NOAA JPSS OMPS EDR format.

This reader supports the official NOAA JPSS OMPS EDR format which are in
the NetCDF4 file format.

Ozone Error Flag Filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^

The Total Ozone files (V8TOZ) include a variable called "ErrorFlag" with
values marking various conditions. It is possible to filter out any pixels
not matching certain conditions of this ErrorFlag variable. To specify what
values should be used/preserved specify ``filter_by_error_flag`` with a list
of the "ErrorFlag" values.

.. code-block:: python

   scn = Scene(reader="omps_edr", filenames=[...], reader_kwargs={"filter_by_error_flag": [0, 1]})

By default no filtering is done related to "ErrorFlag". Refer to algorithm
documentation for more details on the "ErrorFlag" variable.

"""

import logging
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import xarray as xr

from satpy.readers.core.file_handlers import BaseFileHandler
from satpy.readers.core.remote import open_file_or_filename

LOG = logging.getLogger(__name__)


class EDRFileHandler(BaseFileHandler):
    """File handler for NOAA JPSS OMPS EDR files."""

    y_dim_name = "nTimes"
    x_dim_name = "nIFOV"
    error_flag_var_name = "ErrorFlag"

    def __init__(
        self, filename, filename_info, filetype_info, filter_by_error_flag: Sequence[int] | None = None, **kwargs
    ):
        """Initialize the geo filehandler."""
        super(EDRFileHandler, self).__init__(filename, filename_info, filetype_info)

        self.filter_by_error_flag = filter_by_error_flag

        drop_variables = filetype_info.get("drop_variables", None)
        f_obj = open_file_or_filename(self.filename)
        self.nc = xr.open_dataset(
            f_obj,
            decode_cf=filetype_info.get("decode_cf", True),
            mask_and_scale=True,
            drop_variables=drop_variables,
            chunks=-1,  # do one big chunk given size of the data
        )

    @property
    def start_orbit_number(self):
        """Get the start orbit number."""
        return self.nc.start_orbit_number

    @property
    def end_orbit_number(self):
        """Get the end orbit number."""
        return self.nc.end_orbit_number

    @property
    def platform_name(self):
        """Get the platform name."""
        platform_dict = {
            "NPP": "Suomi-NPP",
            "NOAA20": "NOAA-20",
            "NOAA21": "NOAA-21",
            "NOAA22": "NOAA-22",
            "NOAA23": "NOAA-23",
        }
        return platform_dict[self.nc.platform]

    @property
    def sensor_name(self):
        """Get the sensor name."""
        return self.nc.instrument.lower()

    def get_metadata(self, dataset_id, ds_info):
        """Get the metadata."""
        var_path = ds_info.get("file_key", "{}".format(dataset_id["name"]))
        info = getattr(self.nc[var_path], "attrs", {}).copy()
        info.update(ds_info)

        info.update(
            {
                "platform_name": self.platform_name,
                "sensor": self.sensor_name,
                "orbital_parameters": {
                    "start_orbit": self.start_orbit_number,
                    "end_orbit": self.end_orbit_number,
                },
            }
        )
        info.update(dataset_id.to_dict())
        return info

    def get_dataset(self, dataset_id, ds_info):
        """Get the dataset."""
        var_path = ds_info.get("file_key", "{}".format(dataset_id["name"]))
        metadata = self.get_metadata(dataset_id, ds_info)
        data_arr = self.nc[var_path]
        data_arr = data_arr.rename({self.y_dim_name: "y", self.x_dim_name: "x"})
        data_arr = self._mask_invalid(data_arr, metadata)
        data_arr.attrs.update(metadata)
        return data_arr

    def _mask_invalid(self, data_arr: xr.DataArray, ds_info: dict) -> xr.DataArray:
        if self.filter_by_error_flag is not None:
            ef_mask = np.isin(self.nc[self.error_flag_var_name].data, self.filter_by_error_flag)
            data_arr.data = np.where(ef_mask, data_arr.data, np.nan)
        # xarray auto mask and scale handled any fills from the file
        valid_range = ds_info.get("valid_range", data_arr.attrs.get("valid_range"))
        if "valid_min" in data_arr.attrs and valid_range is None:
            valid_range = (data_arr.attrs["valid_min"], data_arr.attrs["valid_max"])
        if valid_range is not None:
            # NOTE: may modify attrs in place
            fill_value = self._handle_fill_value(data_arr)
            return data_arr.where((valid_range[0] <= data_arr) & (data_arr <= valid_range[1]), fill_value)
        return data_arr

    def _handle_fill_value(self, data_arr: xr.DataArray) -> Any:
        if "_FillValue" in data_arr.attrs and not np.issubdtype(data_arr.dtype, np.floating):
            fill_value = data_arr.attrs["_FillValue"]
            if hasattr(fill_value, "shape"):
                fill_value = fill_value.item()
                data_arr.attrs["_FillValue"] = fill_value
        else:
            # fill value is being overwritten with NaN
            data_arr.attrs.pop("_FillValue", None)
            fill_value = xr.core.dtypes.NA
        return fill_value

    def available_datasets(self, configured_datasets=None):
        """Get information of available datasets in this file.

        Args:
            configured_datasets (list): Series of (bool or None, dict) in the
                same way as is returned by this method (see below). The bool
                is whether the dataset is available from at least one
                of the current file handlers. It can also be ``None`` if
                no file handler before us knows how to handle it.
                The dictionary is existing dataset metadata. The dictionaries
                are typically provided from a YAML configuration file and may
                be modified, updated, or used as a "template" for additional
                available datasets. This argument could be the result of a
                previous file handler's implementation of this method.

        Returns:
            Iterator of (bool or None, dict) pairs where dict is the
            dataset's metadata. If the dataset is available in the current
            file type then the boolean value should be ``True``, ``False``
            if we **know** about the dataset but it is unavailable, or
            ``None`` if this file object is not responsible for it.

        """
        # keep track of what variables the YAML has configured, so we don't
        # duplicate entries for them in the dynamic portion
        handled_var_names = set()
        for is_avail, ds_info in configured_datasets or []:
            file_key = ds_info.get("file_key", ds_info["name"])
            # we must add all variables here even if another file handler has
            # claimed the variable. It could be another instance of this file
            # type and we don't want to add that variable dynamically if the
            # other file handler defined it by the YAML definition.
            handled_var_names.add(file_key)
            if is_avail is not None:
                # some other file handler said it has this dataset
                # we don't know any more information than the previous
                # file handler so let's yield early
                yield is_avail, ds_info
                continue
            if self.file_type_matches(ds_info["file_type"]) is None:
                # this is not the file type for this dataset
                yield None, ds_info
                continue
            yield file_key in self.nc, ds_info

        yield from self._dynamic_variables_from_file(handled_var_names)

    def _dynamic_variables_from_file(self, handled_var_names: set) -> Iterable[tuple[bool, dict]]:
        coords: dict[str, dict] = {}
        for is_avail, ds_info in self._generate_dynamic_metadata(self.nc.variables.keys(), coords):
            var_name = ds_info["file_key"]
            if var_name in handled_var_names and var_name not in ("Longitude", "Latitude"):
                continue
            handled_var_names.add(var_name)
            yield is_avail, ds_info

        for coord_info in coords.values():
            yield True, coord_info

    def _generate_dynamic_metadata(self, variable_names: Iterable[str], coords: dict) -> Iterable[tuple[bool, dict]]:
        ftype = self.filetype_info["file_type"]

        for var_name in variable_names:
            data_arr = self.nc[var_name]
            if data_arr.ndim != 2 or data_arr.dims != (self.y_dim_name, self.x_dim_name):
                # only 2D arrays supported at this time
                continue
            res = 50000  # meters
            coords_names = (f"longitude_{ftype}", f"latitude_{ftype}")
            ds_info = {
                "file_key": var_name,
                "file_type": self.filetype_info["file_type"],
                "name": var_name,
                "resolution": res,
                "coordinates": coords_names,
            }

            is_lon = "longitude" in var_name.lower()
            is_lat = "latitude" in var_name.lower()
            if not (is_lon or is_lat):
                yield True, ds_info
                continue

            ds_info["standard_name"] = "longitude" if is_lon else "latitude"
            ds_info["units"] = "degrees_east" if is_lon else "degrees_north"
            # recursive coordinate/SwathDefinitions are not currently handled well in the base reader
            del ds_info["coordinates"]
            yield True, ds_info

            # "standard" geolocation coordinate (assume shorter variable name is "better")
            new_name = coords_names[int(not is_lon)]
            if new_name not in coords or len(var_name) < len(coords[new_name]["file_key"]):
                ds_info = ds_info.copy()
                ds_info["name"] = new_name
                coords[ds_info["name"]] = ds_info
