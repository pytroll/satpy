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

r"""IASI-NG L2 reader

This reader supports reading all the products from the IASI-NG L2 processing
level:
  * IASI-L2-TWV
  * IASI-L2-CLD
  * IASI-L2-GHG
  * IASI-L2-SFC
"""

import logging

import dask.array as da
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

        # Description of the available datasets:
        self.ds_desc = self.filetype_info["datasets"]

        logger.info("Creating reader with infos: %s", filename_info)

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
        """Register a simple dataset given its name and a desc dict"""

        if ds_name in self.dataset_infos:
            raise ValueError(f"Dataset for {ds_name} already registered.")

        ds_infos = {
            "name": ds_name,
            "sensor": "iasi_ng",
            "file_type": self.filetype_info["file_type"],
        }

        ds_infos.update(desc)

        self.dataset_infos[ds_name] = ds_infos

    def register_available_datasets(self):
        """Register the available dataset in the current product file"""

        if self.dataset_infos is not None:
            # Datasets are already registered.
            return

        # Otherwise, we need to perform the registration:
        self.dataset_infos = {}

        for grp_desc in self.ds_desc:
            prefix = grp_desc["group"]
            for vname in grp_desc["variables"]:
                # Check if we have an alias for this variable:
                ds_name = vname
                if ":" in vname:
                    vname, ds_name = vname.split(":")

                desc = {"location": f"{prefix}/{vname}"}

                if ds_name == "onboard_utc":
                    # add the seconds_since_epoch:
                    desc["seconds_since_epoch"] = self.filetype_info["onboard_utc_epoch"]

                # print(f"Registering {ds_name} with desc: {desc}")

                self.register_dataset(ds_name, desc)

    def get_dataset_infos(self, ds_name):
        """Retrieve the dataset infos corresponding to one of the registered
        datasets."""
        if ds_name not in self.dataset_infos:
            raise KeyError(f"No dataset registered for {ds_name}")

        return self.dataset_infos[ds_name]

    def variable_path_exists(self, var_path):
        """Check if a given variable path is available in the underlying
        netCDF file.

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

    def apply_fill_value(self, data_array, ds_info):
        """Apply the rescaling transform on a given array."""

        # Check if we should apply:
        if ds_info.get("apply_fill_value", True) is not True:
            return data_array

        dtype = ds_info.get("data_type", "auto")
        convert = False

        if data_array.dtype == np.int32:
            if dtype == "auto":
                dtype = "float64"
            convert = dtype != "int32"
        elif data_array.dtype == np.float64:
            if dtype == "auto":
                dtype = "float64"
            convert = dtype != "float64"
        elif data_array.dtype == np.float32:
            if dtype == "auto":
                dtype = "float32"
            convert = dtype != "float32"
        else:
            raise ValueError(f"Unexpected raw dataarray data type: {data_array.dtype}")

        if convert:
            data_array = data_array.astype(dtype)

        if dtype not in ["float32", "float64"]:
            # We won't be able to use NaN in the other cases:
            return data_array

        attribs = data_array.attrs
        nan_val = np.nan if dtype == "float64" else np.float32(np.nan)

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

        return data_array.where(data_array != missing_val, nan_val)

    def apply_rescaling(self, data_array, ds_info):
        """Apply the rescaling transform on a given array."""

        # Here we should apply the rescaling except if it is explicitly
        # requested not to rescale:
        if ds_info.get("apply_rescaling", True) is not True:
            return data_array

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

    def apply_reshaping(self, data_array, ds_info):
        """Apply the reshaping transform on a given IASI-NG data array

        Those arrays may come as 3D array, in which case we collapse the
        last 2 dimensions on a single axis (ie. the number of columns or "y")

        In the process, we also rename the first axis to "x"
        """

        if ds_info.get("apply_reshaping", True) is not True:
            return data_array

        if len(data_array.dims) > 2:
            data_array = data_array.stack(y=(data_array.dims[1:]))

        if data_array.dims[0] != "x":
            data_array = data_array.rename({data_array.dims[0]: "x"})

        if data_array.dims[1] != "y":
            data_array = data_array.rename({data_array.dims[1]: "y"})

        return data_array

    def apply_to_datetime(self, data_array, ds_info):
        """Convert the data to datetime values."""

        if "seconds_since_epoch" not in ds_info:
            return data_array

        epoch = ds_info["seconds_since_epoch"]
        # Note: below could convert the resulting data to another type
        # with .astype("datetime64[us]") for instance
        data_array = xr.DataArray(
            data=da.from_array(
                pd.to_datetime(epoch) + data_array.astype("timedelta64[s]")
            ),
            dims=data_array.dims,
            attrs=data_array.attrs,
        )

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
        arr = self.apply_fill_value(arr, ds_info)
        arr = self.apply_rescaling(arr, ds_info)
        arr = self.apply_reshaping(arr, ds_info)
        arr = self.apply_to_datetime(arr, ds_info)

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
