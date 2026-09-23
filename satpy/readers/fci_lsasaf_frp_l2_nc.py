# Copyright (c) 2022 Satpy developers
#
# satpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# satpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with satpy.  If not, see <http://www.gnu.org/licenses/>.

"""MTG FCI Fire Radiative Power (FRP) Level-2 (L2) NC reader.

This reader supports reading the frp product from the LSASAF FRPPIXEL based Product.
It can be e.g. used with SingleBandCompositor or MaskingCompositor.

More detailed information about the related product and data see:
https://lsa-saf.eumetsat.int/en/data/products/fire-products/
"""


from contextlib import suppress
from datetime import timedelta

import dask.array as da
import numpy as np
import xarray as xr

from satpy.area import get_area_def
from satpy.readers.core.fci import platform_name_translate
from satpy.readers.core.file_handlers import BaseFileHandler
from satpy.utils import get_chunk_size_limit

# Full-disk 1 km FCI grid
FRP_GRID_SHAPE = (11136, 11136)

# Derived from PYTROLL_CHUNK_SIZE, else defaults to 128MiB.
CHUNK_SIZE = get_chunk_size_limit()

# Map dataset names to equivalent source file header names
VAR_MAP = {
    "frp": "FRP",
    "fire_confidence": "FIRE_CONFIDENCE",
    "latitude": "LATITUDE",
    "longitude": "LONGITUDE",
    "abs_line": "ABS_LINE",
    "abs_samp": "ABS_SAMP",
}
gridded_vars = {"frp", "fire_confidence"}

class FRPFileHandler(BaseFileHandler):
    """ASCII reader for NC files for LSA SAF Fire Radiative Power product."""

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize file handler."""
        super().__init__(filename, filename_info, filetype_info)

        self.filename_info = filename_info
        self.filename = filename
        #self.satellite_name = type(self).satellite_name.fget(self)

        # Use xarray's default netcdf4 engine to open the file. Read content lazy via dask arrays.
        self.root_nc = xr.open_dataset(
            self.filename,
            decode_cf=False,
            mask_and_scale=False,
            chunks=None
        )
        self.nc = xr.open_dataset(
            self.filename,
            group="ListProduct",
            decode_cf=True,
            mask_and_scale=True,
            chunks={
                "sample": CHUNK_SIZE,
                "line": CHUNK_SIZE
            }
        )
        self.global_attrs = {
            **self.root_nc.attrs,
            **self.nc.attrs
        }

    def __del__(self):
        """Close the NetCDF file that may still be open."""
        with suppress(AttributeError, OSError):
            if self.nc is not None:
                self.nc.close()
            if self.root_nc is not None:
                self.root_nc.close()

    @property
    def satellite_name(self):
        """Return spacecraft name."""
        platform = self.global_attrs.get("platform")
        return platform_name_translate.get(platform, platform)

    @property
    def sensor_name(self):
        """Return instrument name."""
        return self.global_attrs.get("sensor").lower() or "fci"

    @property
    def start_time(self):
        """Get  date and time when observations started."""
        return self.filename_info["start_time"]

    @property
    def end_time(self):
        """Calculate end time 10 minutes after start time, if not defined any else."""
        try:
            frequency = float(self.global_attrs.get("product_frequency", "10-min").split("-min")[0])
        except (AttributeError, TypeError, ValueError):
            frequency = 10
        return self.start_time + timedelta(minutes=frequency)

    def __contains__(self, item):
        """Check if variable is available in current file."""
        return item in VAR_MAP and VAR_MAP[item] in self.nc.data_vars

    def __getitem__(self, key):
        """Get file content for dataset key."""
        return self.nc[VAR_MAP[key]]

    def _get_attributes(self):
        """Create a dictionary of global attributes to be added to all datasets.

        Returns:
            dict: A dictionary of global attributes.
                filename: name of the product file
                satellite_name: name of the satellite
                sensor: name of sensor
                platform_name: name of the platform
                start_time: Begin of Scan
                end_time: End of Scan (not nominal)

        """
        attributes = {
            "filename": self.filename,
            "satellite_name": self.satellite_name,
            "sensor": self.sensor_name,
            "platform_name": self.filename_info.get("platform_name"),
            "start_time": self.start_time,
            "end_time": self.end_time

        }
        return attributes

    def get_dataset(self, dsid, dsinfo):
        """Get requested dataset as xarray.DataArray."""
        name = dsid["name"]
        ds = self.nc[VAR_MAP[name]].data
        data = xr.DataArray(
            ds,
            dims=("y",),
            attrs=self._get_attributes() #,
        )

        for key in ("units", "standard_name", "resolution"):
            if key in dsinfo:
                data.attrs[key] = dsinfo[key]

        # Only the required variables should be mapped on the FCI grid.
        if name in gridded_vars:
            data = self.get_array_on_fci_grid(data)

        return data

    def get_area_def(self, dsid):
        """Get area definition for fci native 1 km grid."""
        return get_area_def("mtg_fci_fdss_1km")

    def get_array_on_fci_grid(self, data_array):
        """Place 1D fire detections on sparse 2D FCI grid."""
        rows = self["abs_line"]
        cols = self["abs_samp"]
        attrs = data_array.attrs.copy()

        rows_int = (rows.astype(int)).compute()
        cols_int = (cols.astype(int)).compute()

        # values is a 1-D List of np.ndarray
        values = data_array.data.compute() if hasattr(data_array.data, "compute") else data_array.values

        flattened_result = np.nan * da.zeros((FRP_GRID_SHAPE[0] * FRP_GRID_SHAPE[1]), dtype=data_array.dtype)
        flattened_result[rows_int * FRP_GRID_SHAPE[1] + cols_int] = values

        #reshape to final 2D grid
        data_2d = da.reshape(flattened_result, FRP_GRID_SHAPE)

        xarr = xr.DataArray(
            da.asarray(data_2d, CHUNK_SIZE),
            dims=("y", "x"),
            attrs=attrs,
        )
        return xarr
