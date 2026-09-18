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

"""MTG FCI Fire Radiative Power (FRP) Level-2 (L2) CSV reader.

This reader supports reading the frp product from the LSASAF FRPPIXEL based Product.
It can be used standalone to read the data from the files, or to generate composites
e.g. with SingleBandCompositor or MaskingCompositor.

More detailed information about the related product and data see:
https://lsa-saf.eumetsat.int/en/data/products/fire-products/

Per default, the reader reads and loads a 1-D array
of fire pixels and maps them on a sparse 2-D grid based on the 1 km Full disk

NOTE: The reader currently assumes the product to be full-disc,
with a 10-minute repeat cycle, and coming from Meteosat-12.
"""


from datetime import timedelta

import dask.array as da
import dask.dataframe as dd
import numpy as np
import xarray as xr

from satpy.area import get_area_def
from satpy.readers.core.file_handlers import BaseFileHandler
from satpy.utils import get_chunk_size_limit

# Full-disk 1 km FCI grid
FRP_GRID_SHAPE = (11136, 11136)

#Derived from PYTROLL_CHUNK_SIZE, else defaults to 128MiB.
CHUNK_SIZE = get_chunk_size_limit()

# Map internal/platform short names to OSCAR standard platform names
PLATFORM_MAP = {
    "MTG": "Meteosat-12",
}

# Map dataset names to equivalent source file header names
COLUMN_MAP = {
    "frp": "FRP",
    "latitude": "LATITUDE",
    "longitude": "LONGITUDE",
    "abs_line": "ABS_LINE",
    "abs_samp": "ABS_SAMP",
}

class FRPFileHandler(BaseFileHandler):
    """ASCII reader for CSV files for LSA SAF Fire Radiative Power product."""

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize file handler."""
        super().__init__(filename, filename_info, filetype_info)

        self.filename_info = filename_info
        self.platform_name = filename_info.get("platform_name")
        self.satellite_name = PLATFORM_MAP.get(self.platform_name, self.platform_name)
        self.file_content = dd.read_csv(
            filename,
            usecols=list(COLUMN_MAP.values())
        )

    @property
    def start_time(self):
        """Get  date and time when observations started."""
        return self.filename_info["start_time"]

    @property
    def end_time(self):
        """Calculate end time 10 minutes after start time."""
        return self.start_time + timedelta(minutes=10)

    def __contains__(self, item):
        """Check if variable is available in current file."""
        return item in COLUMN_MAP and COLUMN_MAP[item] in self.file_content.columns

    def __getitem__(self, key):
        """Get file content for dataset key."""
        return self.file_content[COLUMN_MAP[key]]

    def get_dataset(self, dsid, dsinfo):
        """Get requested dataset as xarray.DataArray."""
        name = dsid["name"]
        series = self[name]
        ds = series.to_dask_array(lengths=True)

        data = xr.DataArray(
            ds,
            dims=("y",),
            attrs={
                "satellite_name": self.satellite_name,
                "platform_name": self.platform_name,
                "sensor": "fci",
                "start_time": self.start_time,
                "end_time": self.end_time,
            },
        )

        for key in ("units", "standard_name", "resolution"):
            if key in dsinfo:
                data.attrs[key] = dsinfo[key]

        # Only the required variables should be mapped on the FCI grid.
        if name == "frp":
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

        # Create an empty 1-D nan array for the results
        flattened_result = np.nan * da.zeros((FRP_GRID_SHAPE[0] * FRP_GRID_SHAPE[1]), dtype=data_array.dtype)
        # Insert the data. Dask doesn't support this for more than one dimension at a time, so ...
        flattened_result[rows_int * FRP_GRID_SHAPE[1] + cols_int] = values #data_array
        # ... reshape to final 2D grid
        data_2d = da.reshape(flattened_result, FRP_GRID_SHAPE)

        xarr = xr.DataArray(
            da.asarray(data_2d, CHUNK_SIZE),
            dims=("y", "x"),
            attrs=attrs,
        )
        return xarr
