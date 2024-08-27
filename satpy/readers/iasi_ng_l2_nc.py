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
  * IASI-L2-O3_
  * IASI-L2-CO_
"""

import logging
import re

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

from .netcdf_utils import NetCDF4FsspecFileHandler

logger = logging.getLogger(__name__)


class IASINGL2NCFileHandler(NetCDF4FsspecFileHandler):
    """Reader for IASI-NG L2 products in NetCDF format."""

    def __init__(self, filename, filename_info, filetype_info, **kwargs):
        """Initialize object."""
        super().__init__(filename, filename_info, filetype_info, **kwargs)

        self.sensors = {"iasi_ng"}

        # List of datasets provided by this handler:
        self.dataset_infos = None

        # Description of the available variables:
        self.variable_desc = {}

        # Description of the available dimensions:
        self.dimensions_desc = {}

        # Ignored variable patterns:
        patterns = self.filetype_info.get("ignored_patterns", [])
        self.ignored_patterns = [re.compile(pstr) for pstr in patterns]

        # dataset aliases:
        self.dataset_aliases = self.filetype_info.get("dataset_aliases", {})

        # broadcasts timestamps flag:
        self.broadcast_timestamps = self.filetype_info.get("broadcast_timestamps", False)

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

        Uses a per product type dataset registration mechanism.
        """
        # pass along existing datasets
        for is_avail, ds_info in configured_datasets or []:
            yield is_avail, ds_info

        for _, ds_info in self.dataset_infos.items():
            yield True, ds_info

    def register_dataset(self, ds_name, desc):
        """Register a simple dataset given its name and a desc dict."""
        if ds_name in self.dataset_infos:
            raise ValueError(f"Dataset for {ds_name} already registered.")

        ds_infos = {
            "name": ds_name,
            "sensor": "iasi_ng",
            "file_type": self.filetype_info["file_type"],
        }

        ds_infos.update(desc)

        self.dataset_infos[ds_name] = ds_infos

    def process_dimension(self, key, value):
        """Process a dimension entry from the file_content."""
        dim_name = key.split("/")[-1]
        # print(f"Found dimension: {dim_name}={val}")
        if dim_name in self.dimensions_desc and self.dimensions_desc[dim_name] != value:
            # This might happen if we have the same dim name from different groups:
            raise ValueError(f"Detected duplicated dim name: {dim_name}")

        self.dimensions_desc[dim_name] = value

    def process_attribute(self, key, value):
        """Process a attribute entry from the file_content."""
        var_path, aname = key.split("/attr/")
        # print(f"Found attrib for: {var_path}: {aname}")

        if var_path not in self.variable_desc:
            # maybe this variable is ignored, or this is a group attr.
            return

        self.variable_desc[var_path]["attribs"][aname] = value

    def process_variable(self, key):
        """Process a variable entry from the file_content."""
        shape = self.file_content[f"{key}/shape"]
        # print(f"Found variable: {key}")
        if np.prod(shape) <= 1:
            # print(f"Ignoring scalar variable {key}")
            return

        # Check if this variable should be ignored:
        if any(p.search(key) is not None for p in self.ignored_patterns):
            # print(f"Ignoring variable {key}")
            return

        # Prepare a description for this variable:
        prefix, var_name = key.rsplit("/", 1)
        dims = self.file_content[f"{key}/dimensions"]
        dtype = self.file_content[f"{key}/dtype"]

        desc = {
            "location": key,
            "prefix": prefix,
            "var_name": var_name,
            "shape": shape,
            "dtype": f"{dtype}",
            "dims": dims,
            "attribs": {},
        }

        self.variable_desc[key] = desc

    def parse_file_content(self):
        """Parse the file_content to discover the available datasets and dimensions."""
        for key, val in self.file_content.items():
            # print(f"Found key: {key}")

            if "/dimension/" in key:
                self.process_dimension(key, val)
                continue

            if "/attr/" in key:
                self.process_attribute(key, val)
                continue

            # if isinstance(val, netCDF4.Variable):
            if f"{key}/shape" in self.file_content:
                self.process_variable(key)
                continue

        # print(f"Found {len(self.variable_desc)} variables:")
        # for vpath, desc in self.variable_desc.items():
        #     print(f"{vpath}: {desc}")

    def register_available_datasets(self):
        """Register the available dataset in the current product file."""
        if self.dataset_infos is not None:
            # Datasets are already registered.
            return

        # Otherwise, we need to perform the registration:
        self.dataset_infos = {}

        # Parse the file content:
        self.parse_file_content()

        for vpath, desc in self.variable_desc.items():
            # Check if we have an alias for this variable:
            ds_name = desc["var_name"]
            if vpath in self.dataset_aliases:
                ds_name = self.dataset_aliases[vpath]

            unit = desc["attribs"].get("units", None)
            if unit is not None and unit.startswith("seconds since "):
                # request conversion to datetime:
                desc["seconds_since_epoch"] = unit.replace("seconds since ", "")

            if self.broadcast_timestamps and desc["var_name"] == "onboard_utc":
                # Broadcast on the "n_fov" dimension:
                desc["broadcast_on_dim"] = "n_fov"

            self.register_dataset(ds_name, desc)

    def get_dataset_infos(self, ds_name):
        """Retrieve the dataset infos corresponding to one of the registered datasets."""
        if ds_name not in self.dataset_infos:
            raise KeyError(f"No dataset registered for {ds_name}")

        return self.dataset_infos[ds_name]

    def variable_path_exists(self, var_path):
        """Check if a given variable path is available in the underlying netCDF file.

        All we really need to do here is to access the file_content dictionary
        and check if we have a variable under that var_path key.
        """
        # but we ignore attributes: or sub properties:
        if var_path.startswith("/attr") or var_path.endswith(
            ("/dtype", "/shape", "/dimensions")
        ):
            return False

        # Check if the path is found:
        if var_path in self.file_content:
            # This is only a valid variable if it is not a netcdf group:
            return not isinstance(self.file_content[var_path], netCDF4.Group)

        # Var path not in file_content:
        return False

    def convert_data_type(self, data_array, dtype="auto"):
        """Convert the data type if applicable."""
        attribs = data_array.attrs

        cur_dtype = np.dtype(data_array.dtype).name

        if dtype == "auto" and cur_dtype in ["float32", "float64"]:
            dtype = cur_dtype

        to_float = "scale_factor" in attribs or "add_offset" in attribs

        if dtype == "auto":
            dtype = "float64" if to_float else cur_dtype

        if cur_dtype != dtype:
            data_array = data_array.astype(dtype)

        return data_array

    def apply_fill_value(self, data_array):
        """Apply the rescaling transform on a given array."""
        dtype = np.dtype(data_array.dtype).name
        if dtype not in ["float32", "float64"]:
            # We won't be able to use NaN in the other cases:
            return data_array

        nan_val = np.nan if dtype == "float64" else np.float32(np.nan)
        attribs = data_array.attrs

        # Apply the min/max valid range:
        if "valid_min" in attribs:
            vmin = attribs["valid_min"]
            data_array = data_array.where(data_array >= vmin, other=nan_val)

        if "valid_max" in attribs:
            vmax = attribs["valid_max"]
            data_array = data_array.where(data_array <= vmax, other=nan_val)

        if "valid_range" in attribs:
            vrange = attribs["valid_range"]
            data_array = data_array.where(data_array >= vrange[0], other=nan_val)
            data_array = data_array.where(data_array <= vrange[1], other=nan_val)

        # Check the missing value:
        missing_val = attribs.get("missing_value", None)
        missing_val = attribs.get("_FillValue", missing_val)

        if missing_val is None:
            return data_array

        return data_array.where(data_array != missing_val, other=nan_val)

    def apply_rescaling(self, data_array):
        """Apply the rescaling transform on a given array."""
        # Check if we have the scaling elements:
        attribs = data_array.attrs
        if "scale_factor" in attribs or "add_offset" in attribs:
            scale_factor = attribs.setdefault("scale_factor", 1)
            add_offset = attribs.setdefault("add_offset", 0)

            data_array = (data_array * scale_factor) + add_offset

            # rescale the valid range accordingly
            for key in ["valid_range", "valid_min", "valid_max"]:
                if key in attribs:
                    attribs[key] = attribs[key] * scale_factor + add_offset

            data_array.attrs.update(attribs)

        return data_array

    def apply_reshaping(self, data_array):
        """Apply the reshaping transform on a given IASI-NG data array.

        Those arrays may come as 3D array, in which case we collapse the
        last 2 dimensions on a single axis (ie. the number of columns or "y")

        In the process, we also rename the first axis to "x"
        """
        if len(data_array.dims) > 2:
            data_array = data_array.stack(y=(data_array.dims[1:]))

        if data_array.dims[0] != "x":
            data_array = data_array.rename({data_array.dims[0]: "x"})

        if data_array.dims[1] != "y":
            data_array = data_array.rename({data_array.dims[1]: "y"})

        return data_array

    def convert_to_datetime(self, data_array, ds_info):
        """Convert the data to datetime values."""
        epoch = ds_info["seconds_since_epoch"]

        # Note: below could convert the resulting data to another type
        # with .astype("datetime64[us]") for instance
        data_array = xr.DataArray(
            data=pd.to_datetime(epoch) + data_array.astype("timedelta64[s]"),
            dims=data_array.dims,
            attrs=data_array.attrs,
        )

        return data_array

    def apply_broadcast(self, data_array, ds_info):
        """Apply the broadcast of the data array."""
        dim_name = ds_info["broadcast_on_dim"]
        if dim_name not in self.dimensions_desc:
            raise ValueError(f"Invalid dimension name {dim_name}")
        rep_count = self.dimensions_desc[dim_name]

        # Apply "a repeat operation" with the last dimension size:
        data_array = xr.concat([data_array] * rep_count, dim=data_array.dims[-1])

        return data_array

    def get_transformed_dataset(self, ds_info):
        """Retrieve a dataset with all transformations applied on it."""
        # Extract location:
        vname = ds_info["location"]

        if not self.variable_path_exists(vname):
            raise ValueError(f"Invalid variable path: {vname}")

        # Read the raw variable data from file (this is an xr.DataArray):
        arr = self[vname]

        # Apply the transformations:
        arr = self.convert_data_type(arr)
        arr = self.apply_fill_value(arr)
        arr = self.apply_rescaling(arr)
        arr = self.apply_reshaping(arr)

        if "seconds_since_epoch" in ds_info:
            arr = self.convert_to_datetime(arr, ds_info)

        if "broadcast_on_dim" in ds_info:
            arr = self.apply_broadcast(arr, ds_info)

        return arr

    def get_dataset(self, dataset_id, ds_info=None):
        """Get a dataset."""
        ds_name = dataset_id["name"]

        # In case this dataset name is not explicitly provided by this file
        # handler then we should simply return None.
        if ds_name not in self.dataset_infos:
            return None

        # Retrieve default infos if missing:
        if ds_info is None:
            ds_info = self.get_dataset_infos(ds_name)

        ds_name = ds_info["name"]

        # Retrieve the transformed data array:
        data_array = self.get_transformed_dataset(ds_info)

        # Return the resulting array:
        return data_array
