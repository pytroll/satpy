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
It can be e.g. used with SingleBandCompositor or MaskingCompositor.

More detailed information about the related product and data see:
https://lsa-saf.eumetsat.int/en/data/products/fire-products/


Per default, the reader retds and loads an sparse 1-D array
of fire pixels and maps them on an 2-D grid based on the 1 km Full disk
FCI grid. The resulting 2-D array is Compatible with later resampling.

NOTE: Currently method end_time is designed for 10 minute scan times, not for RSS.
"""


from datetime import timedelta

import dask.dataframe as dd
import numpy as np
import xarray as xr

from satpy.area import get_area_def
from satpy.readers.core.file_handlers import BaseFileHandler

# Full-disk 1 km FCI grid
FRP_GRID_SHAPE = (11136, 11136)

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
        self.satellite_name = filename_info.get("satellite_name")

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
                "platform_name": PLATFORM_MAP.get(self.satellite_name, self.satellite_name),
                "sensor": "fci",
                "start_time": self.start_time,
                "end_time": self.end_time,
            },
        )

        for key in ("units", "standard_name", "resolution"):
            if key in dsinfo:
                data.attrs[key] = dsinfo[key]

        if name == "frp":
            lons = self["longitude"].to_dask_array(lengths=True)
            lats = self["latitude"].to_dask_array(lengths=True)
            data = data.assign_coords({
                "longitude": ("y", lons),
                "latitude": ("y", lats),
            })
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

        rows_int = (rows.astype(int) - 1).compute()
        cols_int = (cols.astype(int) - 1).compute()

        values = data_array.data.compute() if hasattr(data_array.data, "compute") else data_array.values

        valid = (
            (rows_int >= 0) & (rows_int < FRP_GRID_SHAPE[0]) &
            (cols_int >= 0) & (cols_int < FRP_GRID_SHAPE[1])
        )

        grid = np.full(FRP_GRID_SHAPE, np.nan, dtype=np.float32)
        grid[rows_int[valid], cols_int[valid]] = values[valid]

        xarr = xr.DataArray(
            grid,
            dims=("y", "x"),
            attrs=attrs,
        )
        return xarr
