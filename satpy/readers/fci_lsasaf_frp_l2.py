"""MTG FCI Fire Radiative Power (FRP) Level-2 (L2) combined NC and CSV reader.

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

from contextlib import suppress
from datetime import timedelta

import dask.array as da
import dask.dataframe as dd
import numpy as np
import xarray as xr

from satpy.area import get_area_def
from satpy.readers.core.fci import platform_name_translate
from satpy.readers.core.file_handlers import BaseFileHandler
from satpy.utils import get_chunk_size_limit

# Full-disk 1 km FCI grid
FRP_GRID_SHAPE = (11136, 11136)

# Derived from PYTROLL_CHUNK_SIZE, otherwise defaults to 128 MiB
CHUNK_SIZE = get_chunk_size_limit()

AREA_ID = "mtg_fci_fdss_1km"


# Logical Satpy dataset name -> source variable/column name

CSV_COLUMN_MAP = {
    "frp": "FRP",
    "latitude": "LATITUDE",
    "longitude": "LONGITUDE",
    "abs_line": "ABS_LINE",
    "abs_samp": "ABS_SAMP",
}

# The NC files contain additional variables that are not present in the CSV files.
NC_VAR_MAP = {
    **CSV_COLUMN_MAP,
    "fire_confidence": "FIRE_CONFIDENCE",
}

# Only these variables are mapped to the full FCI grid
GRIDDED_VARS = {"frp", "fire_confidence"}

PLATFORM_MAP = {
    "MTG": "Meteosat-12",
}


class _FRPBaseHandler:
    """Common functionality shared by the NetCDF and CSV handlers."""

    @property
    def start_time(self):
        """Return the observation start time."""
        return self.filename_info["start_time"]

    @property
    def end_time(self):
        """Return the observation end time."""
        return self.start_time + timedelta(minutes=self.product_frequency)

    @property
    def product_frequency(self):
        """Return the product frequency in minutes."""
        return 10.0

    def get_area_def(self, dsid):
        """Return the area definition for the native FCI grid."""
        return get_area_def(AREA_ID)

    def _get_attributes(self):
        """Return common attributes added to all datasets."""
        return {
            "filename": self.filename,
            "satellite_name": self.satellite_name,
            "platform_name": self.filename_info.get("platform_name"),
            "sensor": self.sensor_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }

    def _add_dataset_attributes(self, data, dsinfo):
        """Add metadata from the Satpy dataset information."""
        for key in ("units", "standard_name", "resolution"):
            if key in dsinfo:
                data.attrs[key] = dsinfo[key]

        return data

    def get_array_on_fci_grid(self, data_array):
        """Place 1-D fire detections on the sparse 2-D FCI grid."""
        rows = self["abs_line"]
        cols = self["abs_samp"]

        rows_int = rows.astype(int).compute()
        cols_int = cols.astype(int).compute()

        values = data_array.data.compute() if hasattr(data_array.data, "compute") else data_array.values

        # Use a NumPy array here because the indices and values are
        # computed immediately anyway.
        flattened_result = np.full(
            FRP_GRID_SHAPE[0] * FRP_GRID_SHAPE[1],
            np.nan,
            dtype=data_array.dtype,
        )

        flat_indices = rows_int * FRP_GRID_SHAPE[1] + cols_int
        flattened_result[flat_indices] = values

        data_2d = da.from_array(
            flattened_result.reshape(FRP_GRID_SHAPE),
            chunks=CHUNK_SIZE,
        )

        return xr.DataArray(
            data_2d,
            dims=("y", "x"),
            attrs=data_array.attrs.copy(),
        )

    def _create_dataset(self, name, values, dsinfo):
        """Create a one-dimensional Satpy dataset."""
        data = xr.DataArray(
            values,
            dims=("y",),
            attrs=self._get_attributes(),
        )

        data = self._add_dataset_attributes(data, dsinfo)

        if name in GRIDDED_VARS:
            data = self.get_array_on_fci_grid(data)

        return data


class NCFileHandler(_FRPBaseHandler, BaseFileHandler):
    """Reader for NetCDF LSA SAF Fire Radiative Power products."""

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize file handler."""
        super().__init__(filename, filename_info, filetype_info)

        self.filename = filename
        self.filename_info = filename_info

        # Read root-level attributes without decoding/scaling.
        self.root_nc = xr.open_dataset(
            self.filename,
            decode_cf=False,
            mask_and_scale=False,
            chunks=None,
        )

        # Read the actual product data from the ListProduct group.
        self.nc = xr.open_dataset(
            self.filename,
            group="ListProduct",
            decode_cf=True,
            mask_and_scale=True,
            chunks={
                "sample": CHUNK_SIZE,
                "line": CHUNK_SIZE,
            },
        )

        self.global_attrs = {
            **self.root_nc.attrs,
            **self.nc.attrs,
        }

    def __del__(self):
        """Close open NetCDF datasets."""
        with suppress(AttributeError, OSError):
            if getattr(self, "nc", None) is not None:
                self.nc.close()

            if getattr(self, "root_nc", None) is not None:
                self.root_nc.close()

    @property
    def satellite_name(self):
        """Return the translated spacecraft name."""
        platform = self.global_attrs.get("platform")
        return platform_name_translate.get(platform, platform)

    @property
    def sensor_name(self):
        """Return the instrument name."""
        sensor = self.global_attrs.get("sensor")
        return sensor.lower() if sensor else "fci"

    @property
    def product_frequency(self):
        """Return the product frequency in minutes."""
        product_frequency = self.global_attrs.get("product_frequency", "10-min")

        try:
            return float(product_frequency.split("-min")[0])
        except (AttributeError, TypeError, ValueError):
            return 10.0

    def __contains__(self, item):
        """Check whether a variable is available."""
        return item in NC_VAR_MAP and NC_VAR_MAP[item] in self.nc.data_vars

    def __getitem__(self, key):
        """Return a variable from the NetCDF product."""
        return self.nc[NC_VAR_MAP[key]]

    def get_dataset(self, dsid, dsinfo):
        """Return the requested dataset."""
        name = dsid["name"]
        values = self.nc[NC_VAR_MAP[name]].data
        return self._create_dataset(name, values, dsinfo)


class CSVFileHandler(_FRPBaseHandler, BaseFileHandler):
    """Reader for CSV LSA SAF Fire Radiative Power products."""

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize file handler."""
        super().__init__(filename, filename_info, filetype_info)

        self.filename = filename
        self.filename_info = filename_info

        self.platform_name = filename_info.get("platform_name")
        self._satellite_name = PLATFORM_MAP.get(
            self.platform_name,
            self.platform_name,
        )

        self.file_content = dd.read_csv(
            filename,
            usecols=list(CSV_COLUMN_MAP.values()),
        )

    @property
    def satellite_name(self):
        """Return the satellite name."""
        return self._satellite_name

    @property
    def sensor_name(self):
        """Return the instrument name."""
        return "fci"

    def __contains__(self, item):
        """Check whether a variable is available."""
        return item in CSV_COLUMN_MAP and CSV_COLUMN_MAP[item] in self.file_content.columns

    def __getitem__(self, key):
        """Return a column from the CSV product."""
        return self.file_content[CSV_COLUMN_MAP[key]]

    def get_dataset(self, dsid, dsinfo):
        """Return the requested dataset."""
        name = dsid["name"]

        series = self[name]
        values = series.to_dask_array(lengths=True)

        return self._create_dataset(name, values, dsinfo)
