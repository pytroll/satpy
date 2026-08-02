
"""EUMETSAT EPS-SG METimage (VII) readers base class."""


import datetime as dt
import logging
import os

import numpy as np
import xarray as xr
from geotiepoints.viiinterpolator import tie_points_geo_interpolation, tie_points_interpolation

from satpy.readers.core.metimage import (
    PLATFORM_NAME_TRANSLATE,
    ROWS_PER_SCAN,
    SCAN_ALT_TIE_POINTS,
    TIE_POINTS_FACTOR,
)
from satpy.readers.core.netcdf import NetCDF4FileHandler
from satpy.readers.core.utils import unzip_file
from satpy.utils import normalize_low_res_chunks

logger = logging.getLogger(__name__)

# Row/column dimension name pairs used by the various METimage products. The
# equivalent renaming is done by ``METimageNCBaseFileHandler._standardize_dims``.
PIXEL_DIMS = (("num_lines", "num_pixels"), ("num_points_alt", "num_points_act"))
TIE_POINT_DIMS = ("num_tie_points_alt", "num_tie_points_act")


class METimageNCBaseFileHandler(NetCDF4FileHandler):
    """Base reader class for METimage (VII) products in netCDF format.

    Args:
        filename (str): File to read
        filename_info (dict): Dictionary with filename information
        filetype_info (dict): Dictionary with filetype information
        orthorect (bool): activates the orthorectification correction where available

    """

    # In the future, only final format will be defined once operational data is available
    DATETIME_FORMATS = [
        "%Y%m%d%H%M%S.%f",        # e.g. 20250924121530.123456
        "%Y-%m-%dT%H:%M:%S.%f",   # e.g. 2025-09-24T12:15:30.123456
        "%Y-%m-%d %H:%M:%S",      # e.g. 2025-09-24 12:15:30
        "%Y-%m-%d %H:%M:%S.%f",   # e.g. 2025-09-24 12:15:30.123456
    ]

    def __init__(self, filename, filename_info, filetype_info, orthorect=False):
        """Prepare the class for dataset reading."""
        self._unzipped = unzip_file(filename)
        if self._unzipped:
            filename = self._unzipped
        super().__init__(filename, filename_info, filetype_info, auto_maskandscale=True)

        # Chunk whole rows of pixels so that dask chunks are aligned to the
        # on-disk chunks and to the scans of the instrument.
        # Hold on to row_chunks so we can use it for rechunking interpolated arrays later
        self._row_chunks, chunks = self._chunks_for_file()
        if chunks:
            self._xarray_kwargs["chunks"] = chunks

        # Saves the orthorectification flag
        self.orthorect = orthorect and filetype_info.get("orthorect", True)

        # Saves the interpolation flag
        self.interpolate = filetype_info.get("interpolate", True)

        try:
            longitude = self[filetype_info["cached_longitude"]]
            latitude = self[filetype_info["cached_latitude"]]

            if self.interpolate:
                self.longitude, self.latitude = self._perform_geo_interpolation(longitude, latitude)
                self.longitude = self._rechunk_to_pixel_grid(self.longitude)
                self.latitude = self._rechunk_to_pixel_grid(self.latitude)
            else:
                self.longitude, self.latitude = longitude, latitude

        except KeyError:
            logger.warning("Cached longitude and/or latitude datasets are not correctly defined in YAML file")
            self.longitude, self.latitude = None, None

    def _chunks_for_file(self):
        """Determine a scan-aligned, whole-row dask chunk size for each dimension.

        Returns:
            Tuple of the number of pixel rows per chunk and a mapping of
            dimension name to chunk size suitable for ``xarray.open_dataset``.
            Both are ``None`` if the file does not use any known pixel
            dimension names.

        """
        dim_sizes = self._collect_dim_sizes()
        row_chunks = num_rows = None
        # Map dimension names to chunk sizes
        chunks = {}
        # Look through all possible 2D image array dims (L1b or L2)
        for row_dim, col_dim in PIXEL_DIMS:
            if row_dim not in dim_sizes or col_dim not in dim_sizes:
                continue
            num_rows, num_cols = dim_sizes[row_dim], dim_sizes[col_dim]
            # Use a single dtype for every variable so that all of them are
            # chunked the same way regardless of their on-disk dtype.
            row_chunks = normalize_low_res_chunks(
                ("auto", -1),
                (num_rows, num_cols),
                (ROWS_PER_SCAN, num_cols),
                (1, 1),
                np.float32,
            )[0]
            chunks[row_dim] = row_chunks
            chunks[col_dim] = -1

        if row_chunks is None:
            return None, None

        tie_alt, tie_act = TIE_POINT_DIMS
        if tie_alt in dim_sizes and tie_act in dim_sizes:
            # Interpolating tie points produces one pixel chunk per tie point
            # chunk, so scale the tie point chunks down by the same factor.
            chunks[tie_alt] = max(row_chunks * dim_sizes[tie_alt] // num_rows, 1)
            chunks[tie_act] = -1

        return row_chunks, chunks

    def _collect_dim_sizes(self) -> dict[str, int]:
        """Map dimension name to size for every dimension used by a variable.

        ``NetCDF4FileHandler.collect_dimensions`` does not recurse into groups,
        so the dimensions of a grouped product are not in ``file_content``.
        The per-variable shape and dimension entries always are.

        """
        suffix = "/dimensions"
        dim_sizes: dict[str, int] = {}
        for key, dim_names in self.file_content.items():
            if not key.endswith(suffix):
                continue
            shape = self.file_content.get(key[:-len(suffix)] + "/shape")
            if shape is not None and len(shape) == len(dim_names):
                dim_sizes.update(zip(dim_names, shape))
        return dim_sizes

    def _rechunk_to_pixel_grid(self, variable):
        """Chunk an interpolated variable the same way as the pixel variables.

        Interpolation is done with :meth:`xarray.DataArray.interp`, which
        divides the interpolated dimension evenly by the number of tie point
        chunks. That does not respect the scans of the instrument, so the
        result has to be chunked explicitly to match the pixel variables.

        """
        if self._row_chunks is None:
            return variable
        return variable.chunk({"num_lines": self._row_chunks, "num_pixels": -1})

    def _standardize_dims(self, variable):
        """Standardize dims to y, x."""
        if "num_pixels" in variable.dims:
            variable = variable.rename({"num_pixels": "x", "num_lines": "y"})
        if "num_points_act" in variable.dims:
            variable = variable.rename({"num_points_act": "x", "num_points_alt": "y"})
        if variable.dims[0] == "x":
            variable = variable.transpose("y", "x")
        return variable

    def get_dataset(self, dataset_id, dataset_info):
        """Get dataset using file_key in dataset_info."""
        var_key = dataset_info["file_key"]
        logger.debug("Reading in file to get dataset with key %s.", var_key)

        if var_key == "cached_longitude" and self.longitude is not None:
            variable = self.longitude.copy()
        elif var_key == "cached_latitude" and self.latitude is not None:
            variable = self.latitude.copy()
        else:
            try:
                variable = self[var_key]
            except KeyError:
                logger.warning("Could not find key %s in NetCDF file, no valid Dataset created", var_key)
                return None

            # If the dataset is marked for interpolation, perform the interpolation from tie points to pixels
            if dataset_info.get("interpolate", False) and self.interpolate:
                variable = self._perform_interpolation(variable)
                variable = self._rechunk_to_pixel_grid(variable)

            # Perform the calibration if required
            if dataset_info.get("calibration") is not None:
                variable = self._perform_calibration(variable, dataset_info)

        # Perform the orthorectification if required
        if self.orthorect:
            orthorect_data_name = dataset_info.get("orthorect_data", None)
            if orthorect_data_name is not None:
                variable = self._perform_orthorectification(variable, orthorect_data_name)

        # wrapping longitude between -180 and 180 degrees
        if variable.name == "longitude":
            variable = self.wrap_longitude(variable)

        # Manage the attributes of the dataset
        variable.attrs.setdefault("units", None)

        # Remove possibly incorrect attributes
        for possible_invalid_attr in ("valid_min", "valid_max"):
            variable.attrs.pop(possible_invalid_attr, None)

        variable.attrs.update(dataset_info)
        variable.attrs.update(self._get_global_attributes())
        variable = self._standardize_dims(variable)
        return variable

    def __del__(self):
        """Remove the decompressed temp file, if one was created."""
        super().__del__()   # release the netCDF/h5netcdf handle first, so Windows can drop its lock
        try:
            if getattr(self, "_unzipped", None):
                os.remove(self._unzipped)
        except (AttributeError, OSError):
            pass

    @staticmethod
    def wrap_longitude(longitude_array):
        """Wrap longitude between -180 and 180 degrees."""
        longitude_array = ((longitude_array + 180) % 360) - 180
        return longitude_array

    @staticmethod
    def _perform_interpolation(variable) -> xr.DataArray:
        """Perform the interpolation from tie points to pixel points.

        Args:
            variable: xarray DataArray containing the dataset to interpolate.

        Returns:
            array containing the interpolate values, all the original metadata
            and the updated dimension names.

        """
        interpolated_values = tie_points_interpolation(
            [variable],
            SCAN_ALT_TIE_POINTS,
            TIE_POINTS_FACTOR
        )[0]
        new_variable = interpolated_values.rename(
            num_tie_points_act="num_pixels",
            num_tie_points_alt="num_lines"
        )
        new_variable.name = variable.name
        new_variable.attrs = variable.attrs
        return new_variable

    @staticmethod
    def _perform_geo_interpolation(longitude, latitude):
        """Perform the interpolation of geographic coodinates from tie points to pixel points.

        Args:
            longitude: xarray DataArray containing the longitude dataset to interpolate.
            latitude: xarray DataArray containing the longitude dataset to interpolate.

        Returns:
            tuple of arrays containing the interpolate values, all the original metadata
                    and the updated dimension names.

        """
        interpolated_longitude, interpolated_latitude = tie_points_geo_interpolation(
            longitude,
            latitude,
            SCAN_ALT_TIE_POINTS,
            TIE_POINTS_FACTOR
        )
        new_longitude = interpolated_longitude.rename(
            num_tie_points_act="num_pixels",
            num_tie_points_alt="num_lines"
        )
        new_longitude.name = longitude.name
        new_longitude.attrs = longitude.attrs
        new_latitude = interpolated_latitude.rename(
            num_tie_points_act="num_pixels",
            num_tie_points_alt="num_lines"
        )
        new_latitude.name = latitude.name
        new_latitude.attrs = latitude.attrs
        return new_longitude, new_latitude

    def _perform_orthorectification(self, variable, orthorect_data_name):
        """Perform the orthorectification."""
        raise NotImplementedError

    def _perform_calibration(self, variable, dataset_info):
        """Perform the calibration."""
        raise NotImplementedError

    def _get_global_attributes(self):
        """Create a dictionary of global attributes to be added to all datasets."""
        attributes = {
            "filename": self.filename,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "spacecraft_name": self.spacecraft_name,
            "ssp_lon": self.ssp_lon,
            "sensor": self.sensor,
            "filename_start_time": self.filename_info["sensing_start_time"],
            "filename_end_time": self.filename_info["sensing_end_time"],
            "platform_name": self.spacecraft_name,
            "rows_per_scan": ROWS_PER_SCAN
        }

        # Add a "quality_group" item to the dictionary with all the variables and attributes
        # which are found in the 'quality' group of the VII product
        quality_group = self["quality"]
        quality_dict = {}
        for key in quality_group:
            # Add the values (as Numpy array) of each variable in the group where possible
            try:
                quality_dict[key] = quality_group[key].values
            except ValueError:
                quality_dict[key] = None
        # Add the attributes of the quality group
        quality_dict.update(quality_group.attrs)

        attributes["quality_group"] = quality_dict

        return attributes

    def _parse_datetime(self, datetime_str):
        """Parse datetime string using multiple format attempts."""
        for fmt in self.DATETIME_FORMATS:
            try:
                return dt.datetime.strptime(datetime_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"Unrecognized datetime format: {datetime_str}")

    @property
    def start_time(self):
        """Get observation start time."""
        return self._parse_datetime(self["/attr/sensing_start_time_utc"])

    @property
    def end_time(self):
        """Get observation end time."""
        return self._parse_datetime(self["/attr/sensing_end_time_utc"])

    @property
    def spacecraft_name(self):
        """Return spacecraft name."""
        return PLATFORM_NAME_TRANSLATE.get(self["/attr/spacecraft"], self["/attr/spacecraft"])

    @property
    def sensor(self):
        """Return sensor."""
        return "metimage"

    @property
    def ssp_lon(self):
        """Return subsatellite point longitude."""
        # This parameter is not applicable to METimage
        return None
