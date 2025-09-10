# Copyright (c) 2017-2023 Satpy developers
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

"""IASI-NG L2 reader implementation.

This reader supports reading all the products from the IASI-NG L2 processing
level:

* IASI-L2-TWV
* IASI-L2-CLD
* IASI-L2-GHG
* IASI-L2-SFC
* IASI-L2-O3
* IASI-L2-CO

For more information in the product files content, please refer to the
EPS-SG IASI-NG Level 2 Format specification document provided by EUMETSAT at:
https://www-cdn.eumetsat.int/files/2022-04/EPS-SG%20IASI-NG%20L2%20Product%20Format%20Specification_web_0.pdf
"""

import re

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

from satpy.readers.core.netcdf import NetCDF4FsspecFileHandler


class IASINGL2NCFileHandler(NetCDF4FsspecFileHandler):
    """Reader for IASI-NG L2 products in NetCDF format."""

    def __init__(self, filename, filename_info, filetype_info, **kwargs):
        """Initialize object."""
        super().__init__(
            filename, filename_info, filetype_info, auto_maskandscale=True, **kwargs
        )

        self.sensors = {"iasi_ng"}

        self.dataset_infos = None
        self.variable_desc = {}
        self.dimensions_desc = {}

        patterns = self.filetype_info.get("ignored_patterns", [])
        self.ignored_patterns = [re.compile(pstr) for pstr in patterns]

        aliases = self.filetype_info.get("dataset_aliases", {})
        self.dataset_aliases = {re.compile(key): val for key, val in aliases.items()}

        self.register_available_datasets()

    @property
    def start_time(self):
        """Get the start time."""
        return self.filename_info["sensing_start_time"]

    @property
    def end_time(self):
        """Get the end time."""
        return self.filename_info["sensing_end_time"]

    @property
    def sensor_names(self):
        """List of sensors represented in this file."""
        return self.sensors

    # Note: patching the collect_groups_info method below to
    # also collect dimensions in sub groups.
    def _collect_groups_info(self, base_name, obj):
        for group_name, group_obj in obj.groups.items():
            full_group_name = base_name + group_name
            self.file_content[full_group_name] = group_obj
            self._collect_attrs(full_group_name, group_obj)
            self.collect_metadata(full_group_name, group_obj)
            self.collect_dimensions(full_group_name, group_obj)

    def available_datasets(self, configured_datasets=None):
        """Determine automatically the datasets provided by this file.

        First yield on any element from the provided configured_datasets,
        and then continues with the internally provided datasets.
        """
        for is_avail, ds_info in configured_datasets or []:
            yield is_avail, ds_info

        for _, ds_info in self.dataset_infos.items():
            yield True, ds_info

    def register_dataset(self, ds_name, desc):
        """Register a simple dataset given its name and a desc dict."""
        if ds_name in self.dataset_infos:
            raise KeyError(f"Dataset for {ds_name} already registered.")

        ds_infos = {
            "name": ds_name,
            "sensor": "iasi_ng",
            "file_type": self.filetype_info["file_type"],
        }

        ds_infos.update(desc)

        self.dataset_infos[ds_name] = ds_infos

    def same_dim_names_for_different_groups(self, dim_name, value):
        """Check if we already have this dim_name registered from another group."""
        return (
            dim_name in self.dimensions_desc and self.dimensions_desc[dim_name] != value
        )

    def process_dimension(self, key, value):
        """Process a dimension entry from the file_content."""
        dim_name = key.split("/")[-1]

        if self.same_dim_names_for_different_groups(dim_name, value):
            raise KeyError(f"Detected duplicated dim name: {dim_name}")

        self.dimensions_desc[dim_name] = value

    def has_variable_desc(self, var_path):
        """Check if a given variable path is available."""
        return var_path in self.variable_desc

    def process_attribute(self, key, value):
        """Process a attribute entry from the file_content."""
        var_path, aname = key.split("/attr/")

        if not self.has_variable_desc(var_path):
            return

        self.variable_desc[var_path]["attribs"][aname] = value

    def has_at_most_one_element(self, shape):
        """Check if a shape corresponds to an array with at most 1 element."""
        return np.prod(shape) <= 1

    def is_variable_ignored(self, var_name):
        """Check if a variable should be ignored."""
        return any(p.search(var_name) is not None for p in self.ignored_patterns)

    def prepare_variable_description(self, key, shape):
        """Prepare a description for a given variable."""
        prefix, var_name = key.rsplit("/", 1)
        dims = self.file_content[f"{key}/dimensions"]
        dtype = self.file_content[f"{key}/dtype"]

        return {
            "location": key,
            "prefix": prefix,
            "var_name": var_name,
            "shape": shape,
            "dtype": f"{dtype}",
            "dims": dims,
            "attribs": {},
        }

    def process_variable(self, key):
        """Process a variable entry from the file_content."""
        shape = self.file_content[f"{key}/shape"]

        if self.has_at_most_one_element(shape):
            return

        if self.is_variable_ignored(key):
            return

        self.variable_desc[key] = self.prepare_variable_description(key, shape)

    def parse_file_content(self):
        """Parse the file_content to discover the available datasets and dimensions."""
        for key, val in self.file_content.items():

            if "/dimension/" in key:
                self.process_dimension(key, val)
                continue

            if "/attr/" in key:
                self.process_attribute(key, val)
                continue

            if f"{key}/shape" in self.file_content:
                self.process_variable(key)
                continue

    def check_variable_alias(self, vpath, ds_name):
        """Check if a variable path matches an alias pattern."""
        for pat, sub in self.dataset_aliases.items():
            match = pat.search(vpath)
            if match:
                var_name = match.group(1)
                return sub.replace("${VAR_NAME}", var_name)

        return ds_name

    def register_available_datasets(self):
        """Register the available dataset in the current product file."""
        if self.dataset_infos is not None:
            return

        self.dataset_infos = {}

        self.parse_file_content()

        for vpath, desc in self.variable_desc.items():
            ds_name = desc["var_name"]
            ds_name = self.check_variable_alias(vpath, ds_name)

            unit = desc["attribs"].get("units", None)
            if unit is not None and unit.startswith("seconds since "):
                desc["seconds_since_epoch"] = unit.replace("seconds since ", "")

            self.register_dataset(ds_name, desc)

    def get_dataset_infos(self, ds_name):
        """Retrieve the dataset infos corresponding to one of the registered datasets."""
        if ds_name not in self.dataset_infos:
            raise KeyError(f"No dataset registered for {ds_name}")

        return self.dataset_infos[ds_name]

    def is_attribute_path(self, var_path):
        """Check if a given path is a root attribute path."""
        return var_path.startswith("/attr")

    def is_property_path(self, var_path):
        """Check if a given path is a sub-property path."""
        return var_path.endswith(("/dtype", "/shape", "/dimensions"))

    def is_netcdf_group(self, obj):
        """Check if a given object is a netCDF group."""
        return isinstance(obj, netCDF4.Group)

    def variable_path_exists(self, var_path):
        """Check if a given variable path is available in the underlying netCDF file.

        All we really need to do here is to access the file_content dictionary
        and check if we have a variable under that var_path key.
        """
        if self.is_attribute_path(var_path) or self.is_property_path(var_path):
            return False

        if var_path in self.file_content:
            return not self.is_netcdf_group(self.file_content[var_path])

        return False

    def convert_to_datetime(self, data_array, ds_info):
        """Convert the data to datetime values."""
        epoch = ds_info["seconds_since_epoch"]

        # Note: converting the time values to ns precision to avoid warnings
        # from panda+numpy:
        base_time = np.datetime64(pd.to_datetime(epoch), "ns")
        nanoseconds = data_array.astype("timedelta64[ns]") * 1e9

        data_array = xr.DataArray(
            data=base_time + nanoseconds,
            dims=data_array.dims,
            attrs=data_array.attrs,
        )

        return data_array

    def get_transformed_dataset(self, ds_info):
        """Retrieve a dataset with all transformations applied on it."""
        vname = ds_info["location"]

        if not self.variable_path_exists(vname):
            raise KeyError(f"Invalid variable path: {vname}")

        arr = self[vname]

        if "seconds_since_epoch" in ds_info:
            arr = self.convert_to_datetime(arr, ds_info)

        return arr

    def get_dataset(self, dataset_id, ds_info=None):
        """Get a dataset."""
        ds_name = dataset_id["name"]

        if ds_name not in self.dataset_infos:
            return None

        if ds_info is None:
            ds_info = self.get_dataset_infos(ds_name)

        ds_name = ds_info["name"]

        data_array = self.get_transformed_dataset(ds_info)

        return data_array
