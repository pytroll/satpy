"""Generic metadata-driven reader for UVNS-family products.

This module provides the first refactored implementation of the dynamic
Sentinel-4, Sentinel-5, and Sentinel-5P UVNS reader.

The file handler:

* uses Satpy's NetCDF4FileHandler and its existing structural index;
* exposes NetCDF/HDF5 variables dynamically;
* preserves full internal paths as the canonical variable identity;
* assigns deterministic, unique Satpy dataset names;
* translates explicit NetCDF ``coordinates`` attributes into Satpy coordinate
  dataset references;
* supplies missing latitude and longitude ``standard_name`` attributes when
  these can be determined unambiguously from CF-compatible units;
* normalises known byte-string attribute representations;
* performs no shape-based coordinate guessing.

Legacy recovery logic is intentionally not included here. It should be added
later only for representative products that demonstrably require it.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import h5py
import netCDF4
import numpy as np
import xarray as xr
from pyresample.geometry import SwathDefinition

from satpy.readers.core.netcdf import H5NetcdfAccessor, NetCDF4FileHandler

logger = logging.getLogger(__name__)

DIMENSION_RENAMES = {
    "scanline": "y",
    "ground_pixel": "x",
}

LATITUDE_UNITS = {
    "degree_north",
    "degrees_north",
    "degree_n",
    "degrees_n",
}

LONGITUDE_UNITS = {
    "degree_east",
    "degrees_east",
    "degree_e",
    "degrees_e",
}

SYNTHETIC_DIMENSION_PREFIXES = (
    "phony_dim",
    "phony_dimension",
)


_DTYPE_TO_FILL_KEY = {
    np.dtype("int8"): "i1",
    np.dtype("uint8"): "u1",
    np.dtype("int16"): "i2",
    np.dtype("uint16"): "u2",
    np.dtype("int32"): "i4",
    np.dtype("uint32"): "u4",
    np.dtype("int64"): "i8",
    np.dtype("uint64"): "u8",
    np.dtype("float32"): "f4",
    np.dtype("float64"): "f8",
}

#
# Variables below this threshold are opened eagerly.
#
# A value of 1 effectively disables size-based eager loading
# while preserving VLEN special handling.
#
LAZY_LIMIT = 1

@dataclass(frozen=True)
class VariableRecord:
    """Immutable description of a discovered file variable.

    VariableRecord instances form the reader's internal metadata index.

    Records contain:

    * canonical internal path
    * original variable name
    * generated dataset name
    * dimensions
    * shape
    * dtype
    * normalised attributes
    * VLEN metadata

    Records contain metadata only and never hold variable data.
    """

    path: str
    name: str
    dataset_name: str
    dimensions: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    attrs: Mapping[str, Any]
    is_vlen_string: bool = False

class AttributeNormalizer:
    """Normalise backend-dependent NetCDF and HDF5 attribute values."""

    @classmethod
    def normalise_value(cls, value: Any) -> Any:
        """Return a stable representation of an attribute value.

        Existing Python strings are left intact other than removal of trailing
        NUL padding. Scalar byte strings are decoded as UTF-8. NumPy byte-string
        arrays are decoded element by element.

        Numeric values and arrays are returned unchanged.
        """
        if isinstance(value, str):
            return value.rstrip("\x00")

        if isinstance(value, (bytes, np.bytes_)):
            return bytes(value).decode(
                "utf-8",
                errors="strict",
            ).rstrip("\x00")

        if isinstance(value, np.ndarray) and value.dtype.kind == "S":
            decoded = np.char.decode(
                value,
                "utf-8",
                errors="strict",
            )
            return np.char.rstrip(decoded, "\x00")

        if isinstance(value, tuple):
            return tuple(cls.normalise_value(item) for item in value)

        if isinstance(value, list):
            return [cls.normalise_value(item) for item in value]

        return value

    @classmethod
    def normalise_attrs(
        cls,
        attrs: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Normalise all values in an attribute mapping."""
        return {
            str(key): cls.normalise_value(value)
            for key, value in attrs.items()
        }


class DatasetNameRegistry:
    """Generate deterministic Satpy dataset names.

    Unique basenames are preserved.

    Duplicate basenames are disambiguated using the shortest
    unique path suffix.
    """

    def __init__(self, variable_paths: Iterable[str]):
        """Create a registry for the supplied internal paths."""
        self._paths = tuple(
            sorted(self.normalise_path(path) for path in variable_paths)
        )
        self._path_to_name = self._build_name_registry(self._paths)
        self._name_to_path = {
            dataset_name: path
            for path, dataset_name in self._path_to_name.items()
        }

    @staticmethod
    def normalise_path(path: str) -> str:
        """Return a canonical NetCDF internal path without a leading slash."""
        return str(path).strip().lstrip("/")

    @staticmethod
    def _short_name(path: str) -> str:
        """Return the basename of an internal path."""
        return path.rsplit("/", 1)[-1]

    @classmethod
    def _build_name_registry(
        cls,
        paths: tuple[str, ...],
    ) -> dict[str, str]:
        """Generate the shortest deterministic unique name for every path."""
        basename_counts = Counter(cls._short_name(path) for path in paths)
        result: dict[str, str] = {}

        for path in paths:
            parts = path.split("/")
            basename = parts[-1]

            if basename_counts[basename] == 1:
                result[path] = basename
                continue

            result[path] = cls._shortest_unique_suffix(path, paths)

        cls._validate_unique_names(result)
        return result

    @staticmethod
    def _shortest_unique_suffix(
        path: str,
        all_paths: tuple[str, ...],
    ) -> str:
        """Return the shortest path suffix that uniquely identifies a path."""
        parts = path.split("/")

        for suffix_length in range(2, len(parts) + 1):
            suffix = parts[-suffix_length:]

            matching_paths = [
                candidate
                for candidate in all_paths
                if candidate.split("/")[-suffix_length:] == suffix
            ]

            if len(matching_paths) == 1:
                return "__".join(suffix)

        return "__".join(parts)

    @staticmethod
    def _validate_unique_names(mapping: Mapping[str, str]) -> None:
        """Raise if a generated Satpy dataset name is not unique."""
        counts = Counter(mapping.values())
        duplicates = sorted(
            name
            for name, count in counts.items()
            if count > 1
        )

        if duplicates:
            raise ValueError(
                "Could not generate unique dynamic dataset names: "
                f"{duplicates!r}"
            )

    def dataset_name(self, path: str) -> str:
        """Return the Satpy dataset name for an internal variable path."""
        return self._path_to_name[self.normalise_path(path)]

    def variable_path(self, dataset_name: str) -> str:
        """Return the internal path for a Satpy dataset name."""
        return self._name_to_path[dataset_name]

    def contains_path(self, path: str) -> bool:
        """Return whether an internal path exists in the registry."""
        return self.normalise_path(path) in self._path_to_name


class CoordinateResolver:
    """Resolve explicit geographic coordinate relationships.

    Only coordinate variables explicitly referenced through the
    NetCDF ``coordinates`` attribute are considered.

    No file-wide search, shape heuristics or name guessing are used.
    """

    def __init__(
        self,
        records: Mapping[str, VariableRecord],
        names: DatasetNameRegistry,
    ):
        """Create a resolver for an indexed file."""
        self._records = records
        self._names = names
        self._referenced_coordinate_paths: set[str] = set()

    @property
    def referenced_coordinate_paths(self) -> frozenset:
        """Return every coordinate variable referenced by a dataset."""
        return frozenset(self._referenced_coordinate_paths)

    @staticmethod
    def geographic_role(attrs: Mapping[str, Any]) -> str | None:
        """Return latitude or longitude when metadata identifies the role."""
        standard_name = str(
            attrs.get("standard_name", "")
        ).strip().lower()

        if standard_name == "latitude":
            return "latitude"

        if standard_name == "longitude":
            return "longitude"

        units = str(attrs.get("units", "")).strip().lower()

        if units in LATITUDE_UNITS:
            return "latitude"

        if units in LONGITUDE_UNITS:
            return "longitude"

        return None

    @staticmethod
    def _parse_coordinate_paths(value: Any) -> tuple[str, ...]:
        """Parse a NetCDF coordinates attribute into canonical paths."""
        if value is None:
            return ()

        if isinstance(value, str):
            raw_values = value.split()
        elif isinstance(value, (tuple, list, np.ndarray)):
            raw_values = value
        else:
            logger.warning(
                "Ignoring unsupported coordinates attribute type %s",
                type(value).__name__,
            )
            return ()

        return tuple(
            DatasetNameRegistry.normalise_path(str(path))
            for path in raw_values
            if str(path).strip()
        )

    @staticmethod
    def _coordinates_are_compatible(
        source: VariableRecord,
        longitude: VariableRecord,
        latitude: VariableRecord,
    ) -> bool:
        """Validate a geographic coordinate pair against a source variable."""
        if longitude.dimensions != latitude.dimensions:
            logger.warning(
                "Longitude %r and latitude %r have different dimensions: "
                "%r and %r",
                longitude.path,
                latitude.path,
                longitude.dimensions,
                latitude.dimensions,
            )
            return False

        if longitude.shape != latitude.shape:
            logger.warning(
                "Longitude %r and latitude %r have different shapes: "
                "%r and %r",
                longitude.path,
                latitude.path,
                longitude.shape,
                latitude.shape,
            )
            return False

        source_dimensions = set(source.dimensions)
        coordinate_dimensions = set(longitude.dimensions)

        if not coordinate_dimensions.issubset(source_dimensions):
            logger.warning(
                "Coordinate dimensions %r are not a subset of dimensions %r "
                "for dataset %r",
                longitude.dimensions,
                source.dimensions,
                source.path,
            )
            return False

        return True

    def resolve(self, record: VariableRecord) -> tuple[str, ...]:
        """Return Satpy dataset names for explicit geographic coordinates.

        Only variables referenced by the source variable's ``coordinates``
        attribute are considered. No whole-file latitude or longitude search is
        performed.
        """
        coordinate_paths = self._parse_coordinate_paths(
            record.attrs.get("coordinates")
        )

        if not coordinate_paths:
            return ()

        geographic_records: dict[str, VariableRecord] = {}

        for coordinate_reference in coordinate_paths:
            coordinate_record = self._resolve_coordinate_reference(
                coordinate_reference,
                record,
            )

            if coordinate_record is None:
                logger.warning(
                    "Dataset %r references missing coordinate variable %r",
                    record.path,
                    coordinate_reference,
                )
                continue

            role = self.geographic_role(
                coordinate_record.attrs
            )

            if role is None:
                logger.debug(
                    "Ignoring non-geographic coordinate %r referenced by %r",
                    coordinate_record.path,
                    record.path,
                )
                continue

            previous_record = geographic_records.get(role)

            if (
                previous_record is not None
                and previous_record.path != coordinate_record.path
            ):
                logger.warning(
                    "Dataset %r references more than one %s coordinate: "
                    "%r and %r",
                    record.path,
                    role,
                    previous_record.path,
                    coordinate_record.path,
                )
                return ()

            geographic_records[role] = coordinate_record

        if set(geographic_records) != {"longitude", "latitude"}:
            if geographic_records:
                logger.warning(
                    "Dataset %r has an incomplete geographic coordinate pair: %r",
                    record.path,
                    {
                        role: rec.path
                        for role, rec in geographic_records.items()
                    },
                )
            return ()

        longitude_record = geographic_records["longitude"]
        latitude_record = geographic_records["latitude"]

        if not self._coordinates_are_compatible(
            record,
            longitude_record,
            latitude_record,
        ):
            return ()

        self._referenced_coordinate_paths.update(
            (
                longitude_record.path,
                latitude_record.path,
            )
        )

        return (
            self._names.dataset_name(
                longitude_record.path
            ),
            self._names.dataset_name(
                latitude_record.path
            ),
        )

    def _resolve_coordinate_reference(
        self,
        reference: str,
        source_record: VariableRecord,
    ) -> VariableRecord | None:
        """Resolve a coordinate reference.

        Resolution order:

            1. Full path match.

            2. Same-group lookup relative to
               the requesting dataset.

            3. Unique basename lookup.

            4. Ambiguity failure.
        """
        reference = DatasetNameRegistry.normalise_path(
            reference
        )

        #
        # 1. Full-path lookup.
        #

        record = self._records.get(reference)

        if record is not None:
            return record

        #
        # 2. Namespace-local lookup.
        #
        # Example:
        #
        # source:
        #   data/band1a/geolocation_data/solar_zenith_angle
        #
        # coordinate:
        #   longitude
        #
        # candidate:
        #   data/band1a/geolocation_data/longitude
        #

        parent_group = source_record.path.rsplit(
            "/",
            1,
        )[0]

        local_candidate = (
            f"{parent_group}/{reference}"
        )

        record = self._records.get(
            local_candidate
        )

        if record is not None:

            logger.debug(
                "Resolved coordinate %r for %r "
                "via namespace-local lookup: %r",
                reference,
                source_record.path,
                local_candidate,
            )

            return record

        #
        # 3. Global unique-basename lookup.
        #

        matches = [
            candidate
            for candidate in self._records.values()
            if candidate.name == reference
        ]

        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:

            logger.warning(
                "Coordinate reference %r is ambiguous: %r",
                reference,
                [match.path for match in matches],
            )

        return None

class BackendCompatibility:
    """Backend interoperability helpers.

    Encapsulates known behavioural differences between h5netcdf,
    netCDF4, h5py and xarray.

    This class contains backend-specific compatibility logic and is
    intentionally independent of UVNS dataset discovery, coordinate
    handling and Satpy metadata construction.
    """

    def __init__(self, filename):
        """Create backend compatibility helpers for a file."""
        self.filename = filename

        self._h5netcdf_accessor = None
        self._h5netcdf_file_handle = None

    def get_h5netcdf_handle(self):
        """Return an h5netcdf handle for backend inspection.

        This handle is used only for backend compatibility checks.
        It is not part of the normal dataset loading path.
        """
        if self._h5netcdf_file_handle is None:

            self._h5netcdf_accessor = H5NetcdfAccessor()

            self._h5netcdf_file_handle = (
                self._h5netcdf_accessor.create_file_handle(
                    self.filename
                )
            )

        return self._h5netcdf_file_handle

    def requires_compound_fallback(
        self,
        group,
        key,
    ):
        """Determine whether h5netcdf generated an incompatible dtype_view.

        Certain HDF5 compound datatypes are exposed by h5netcdf through a
        NumPy dtype_view whose structure differs from the underlying HDF5
        datatype.

        For example:

            dtype.itemsize      = 8
            dtype_view.itemsize = 6

        h5netcdf later attempts:

            h5ds[key].view(dtype_view)

        which raises:

            ValueError:
                When changing to a smaller dtype, its size must be a
                divisor of the size of original dtype.

        When such a mismatch is detected, the variable should be loaded
        through the netCDF4 fallback path instead.
        """
        fh = self.get_h5netcdf_handle()

        grp = fh if group is None else fh[group]

        var = grp.variables[key]

        dtype = var._h5ds.dtype

        if dtype.fields is None:
            return False

        view = getattr(
            var.datatype,
            "dtype_view",
            None,
        )

        if view is None:
            return False

        if dtype.names != view.names:

            logger.warning(
                "Compound dtype/view field mismatch for %s: "
                "%r != %r",
                key,
                dtype.names,
                view.names,
            )

            return True

        if dtype.itemsize != view.itemsize:

            logger.warning(
                "Compound dtype/view size mismatch for %s: "
                "dtype.itemsize=%s view.itemsize=%s",
                key,
                dtype.itemsize,
                view.itemsize,
            )

            return True

        return False

    def read_compound_via_netcdf4(
        self,
        group,
        key,
    ):
        """Read a compound variable directly using netCDF4.

        This bypasses the standard h5netcdf/xarray loading path for
        compound datatypes that cannot be loaded reliably through the
        normal backend stack.

        Attributes are also read via netCDF4. This preserves NetCDF's
        interpretation of attribute datatypes, including numeric array
        attributes that may otherwise appear as generic object arrays
        through lower-level HDF5 interfaces.
        """
        ds = netCDF4.Dataset(self.filename)

        try:

            grp = ds if group is None else ds[group]

            var = grp.variables[key]

            data = var[...]

            attrs = {}

            for attr_name in var.ncattrs():

                try:

                    #
                    # Read attributes through netCDF4 rather than directly
                    # from the underlying HDF5 layer.
                    #
                    # Several UVNS products were observed to expose some
                    # array-valued metadata differently depending on the
                    # backend used. netCDF4 generally provides the most
                    # faithful interpretation of NetCDF attribute types.
                    #
                    attrs[attr_name] = var.getncattr(
                        attr_name
                    )

                except Exception:

                    #
                    # Preserve recovery of the compound variable even if
                    # an individual attribute cannot be decoded.
                    #
                    logger.warning(
                        "Skipping unreadable attribute %r on %r",
                        attr_name,
                        key,
                    )

            logger.warning(
                "Loaded compound variable via netCDF4 fallback: %s "
                "(dtype=%s)",
                key,
                data.dtype,
            )

            return xr.DataArray(
                data=data,
                dims=var.dimensions,
                attrs=attrs,
                name=key,
            )

        finally:
            ds.close()

#################################################
class DatasetDerivations:
    """Utilities for generating derived datasets."""

    @staticmethod
    def expand_wavelengths(
        coefficients,
        num_channels,
    ):
        """Expand Chebyshev wavelength coefficients.

        Input shape::

            (
                scanline,
                ground_pixel,
                wavelength_coefficients,
            )

        Output shape::

            (
                spectral_channel,
                scanline,
                ground_pixel,
            )
        """
        if coefficients.ndim != 3:
            raise ValueError(
                "Expected wavelength coefficients with shape "
                "(scanline, ground_pixel, wavelength_coefficients), "
                f"got {coefficients.shape!r}"
            )

        coeffs = np.moveaxis(
            coefficients,
            -1,
            0,
        )

        x = np.linspace(
            -1.0,
            1.0,
            num_channels,
        )

        wavelengths = np.polynomial.chebyshev.chebval(
            x,
            coeffs,
        )

        return np.moveaxis(
            wavelengths,
            -1,
            0,
        )

    def _coefficients_have_data(
        self,
        band_path,
        coefficient_name,
    ):
        da = self[
            f"{band_path}/instrument_data/"
            f"{coefficient_name}"
        ]

        fill_value = da.attrs.get("_FillValue")

        data = da.data

        valid = np.isfinite(data)

        if fill_value is not None:
            valid &= (data != fill_value)

        valid_min = da.attrs.get("valid_min")
        valid_max = da.attrs.get("valid_max")

        if valid_min is not None:
            valid &= (data >= valid_min)

        if valid_max is not None:
            valid &= (data <= valid_max)

        return bool(valid.any().compute())

class UVNSFileHandler(NetCDF4FileHandler):
    """Dynamically discover and load UVNS-family file variables."""

    def __init__(
        self,
        filename,
        filename_info,
        filetype_info,
        engine="h5netcdf",
    ):
        """Initialise the file handler and build the dynamic registry."""
        super().__init__(
            filename,
            filename_info,
            filetype_info,
            engine=engine,
        )

        logger.info(
            "Opening UVNS file: %s",
            filename,
        )

        self._backend = BackendCompatibility(
            self.filename,
        )

        self._records = self._build_variable_records()
        self._name_registry = DatasetNameRegistry(self._records)
        self._records = self._assign_dataset_names(self._records)

        self._coordinate_resolver = CoordinateResolver(
            self._records,
            self._name_registry,
        )
        self._dataset_infos = self._build_dataset_infos()
        self._derived_dataset_infos = (
            self._discover_wavelength_derivations()
        )

    # Inherited
    def _collect_variable_info(
        self,
        var_name,
        var_obj,
    ):
        NetCDF4FileHandler._collect_variable_info(
            self,
            var_name,
            var_obj,
        )

        if self.accessor.engine == "h5netcdf":
            try:
                info = h5py.check_string_dtype(
                    var_obj._h5ds.dtype
                )

                is_vlen_string = (
                    info is not None
                    and info.length is None
                )

                self.file_content[
                    var_name + "/is_vlen_string"
                ] = is_vlen_string

            except Exception:
                self.file_content[
                    var_name + "/is_vlen_string"
                ] = False

    def _get_h5netcdf_handle(self):
        """Return a handle used to inspect h5netcdf datatype mappings.

        This handle is used only to compare the original HDF5 compound
        datatype against the dtype_view generated by h5netcdf.

        It is not part of the netCDF4 fallback path.
        """
        if not hasattr(self, "_h5netcdf_file_handle"):

            self._h5netcdf_accessor = H5NetcdfAccessor()

            self._h5netcdf_file_handle = (
                self._h5netcdf_accessor.create_file_handle(
                    self.filename
                )
            )

        return self._h5netcdf_file_handle

    @property
    def start_time(self):
        """Return the product start time."""
        return self.filename_info["start_time"]

    @property
    def end_time(self):
        """Return the product end time."""
        return self.filename_info.get(
            "end_time",
            self.start_time,
        )

    @property
    def sensor_names(self):
        """Return the sensors associated with this file."""
        sensor = self.filename_info.get(
            "sensor",
            self.filetype_info.get("sensor"),
        )

        if sensor is None:
            return set()

        if isinstance(sensor, str):
            return {sensor}

        return set(sensor)

    def _variable_paths(self) -> tuple[str, ...]:
        """Return all variable paths from Satpy's structural index."""
        return tuple(
            sorted(
                key
                for key, value in self.file_content.items()
                if self.accessor.is_variable(value)
            )
        )

    def _build_variable_records(self) -> dict[str, VariableRecord]:
        """Build structural records from Satpy's indexed file content."""
        records: dict[str, VariableRecord] = {}

        for raw_path in self._variable_paths():
            path = DatasetNameRegistry.normalise_path(raw_path)

            attrs = self._get_indexed_variable_attrs(raw_path)

            dimensions = tuple(
                self.file_content[raw_path + "/dimensions"]
            )
            shape = tuple(
                self.file_content[raw_path + "/shape"]
            )
            dtype = str(
                self.file_content[raw_path + "/dtype"]
            )

            self._warn_for_synthetic_dimensions(
                path,
                dimensions,
            )

            is_vlen_string = self.file_content.get(
                raw_path + "/is_vlen_string",
                False,
            )

            records[path] = VariableRecord(
                path=path,
                name=path.rsplit("/", 1)[-1],
                dataset_name="",
                dimensions=dimensions,
                shape=shape,
                dtype=dtype,
                is_vlen_string=is_vlen_string,
                attrs=attrs,
            )

        return records

    def _assign_dataset_names(
        self,
        records: Mapping[str, VariableRecord],
    ) -> dict[str, VariableRecord]:
        """Add unique Satpy names to structural variable records."""
        return {
            path: VariableRecord(
                path=record.path,
                name=record.name,
                dataset_name=self._name_registry.dataset_name(path),
                dimensions=record.dimensions,
                shape=record.shape,
                dtype=record.dtype,
                is_vlen_string=record.is_vlen_string,
                attrs=record.attrs,
            )
            for path, record in records.items()
        }

    @staticmethod
    def _warn_for_synthetic_dimensions(
        path: str,
        dimensions: tuple[str, ...],
    ) -> None:
        """Warn when a backend-created synthetic dimension is present."""
        synthetic = [
            dimension
            for dimension in dimensions
            if str(dimension).startswith(
                SYNTHETIC_DIMENSION_PREFIXES
            )
        ]

        if synthetic:
            logger.warning(
                "Variable %r uses synthetic dimensions %r; no semantic "
                "dimension inference is performed by the normal UVNS path",
                path,
                synthetic,
            )

    def _build_dataset_infos(self) -> dict[str, dict[str, Any]]:
        """Build and cache Satpy dataset descriptions."""
        dataset_infos: dict[str, dict[str, Any]] = {}

        for path, record in self._records.items():
            ds_info = dict(record.attrs)

            # Preserve the source attribute for diagnostics, but don't pass its
            # internal NetCDF paths directly to Satpy.
            raw_coordinates = ds_info.pop("coordinates", None)

            ds_info.update({
                "name": record.dataset_name,
                "file_key": path,
                "file_type": self.filetype_info["file_type"],
                "source_dimensions": record.dimensions,
                "source_shape": record.shape,
            })

            if raw_coordinates is not None:
                ds_info["source_coordinates"] = raw_coordinates

            geographic_role = self._coordinate_resolver.geographic_role(
                record.attrs
            )

            # Latitude and longitude datasets must not depend on themselves.
            if geographic_role is None:
                coordinates = self._coordinate_resolver.resolve(record)

                if coordinates:
                    ds_info["coordinates"] = coordinates

            dataset_infos[record.dataset_name] = ds_info

        return dataset_infos

    ######
    def _build_xarray_kwargs(
        self,
        kwargs_override,
    ):
        """Build xarray open_dataset keyword arguments."""
        kwargs = dict(self._xarray_kwargs)

        kwargs.update(
            kwargs_override
        )

        return kwargs

    def _apply_vlen_rules(
        self,
        file_key,
        kwargs,
    ):
        """Apply VLEN-string compatibility handling.

        Dask-backed lazy access to some HDF5 VLEN string variables has
        been observed to trigger backend crashes. Open these variables
        eagerly by disabling chunked loading.
        """
        record = self._records[file_key]

        if (
            record.is_vlen_string
            and kwargs.get("chunks") is not None
        ):
            logger.info(
                "Opening VLEN string variable eagerly: %s",
                file_key,
            )

            kwargs = kwargs.copy()

            kwargs["chunks"] = None

        return kwargs

    def _maybe_use_compound_fallback(
        self,
        group,
        key,
        kwargs,
    ):
        """Return a backend fallback result if required.

        Returns
        -------
        xarray.DataArray | None
            Loaded variable when a fallback path is required,
            otherwise None.
        """
        engine = kwargs.get(
            "engine"
        )

        if engine != "h5netcdf":
            return None

        if not self._backend.requires_compound_fallback(
            group,
            key,
        ):
            return None

        file_key = (
            key
            if group is None
            else f"{group}/{key}"
        )

        logger.warning(
            "Compound variable uses incompatible h5netcdf dtype_view; "
            "loading via netCDF4 instead: %s",
            file_key,
        )

        return self._backend.read_compound_via_netcdf4(
            group,
            key,
        )

    def _open_xarray_variable(
        self,
        group,
        key,
        kwargs,
    ):
        """Open a variable through xarray."""
        file_key = (
            key
            if group is None
            else f"{group}/{key}"
        )

        try:

            with xr.open_dataset(
                self.filename,
                group=group,
                **kwargs,
            ) as nc:

                val = nc[key]

        except Exception:

            logger.exception(
                "xarray backend failed opening %s",
                file_key,
            )
            raise

        return val

    ######Overrideen inheritance methods. #######
    ######
    ######
    ######
    def _get_var_from_xr(
        self,
        group,
        key,
        **kwargs_override,
    ):
        """Load a variable through xarray or a compatible fallback."""
        kwargs = self._build_xarray_kwargs(
            kwargs_override
        )

        file_key = (
            key
            if group is None
            else f"{group}/{key}"
        )

        kwargs = self._apply_vlen_rules(
            file_key,
            kwargs,
        )

        fallback = self._maybe_use_compound_fallback(
            group,
            key,
            kwargs,
        )

        if fallback is not None:
            return fallback

        return self._open_xarray_variable(
            group,
            key,
            kwargs,
        )


    def _get_var_from_netcdf4(self, group, key):

        logger.warning(
            "Falling back to netCDF4 backend for %s",
            key if group is None else f"{group}/{key}",
        )

        result = self._get_var_from_xr(
            group,
            key,
            engine="netcdf4",
        )

        return result

    def _repair_handlers(self):
        """Return registered metadata repair paths."""
        return (
            (
                self._is_time_decode_error,
                self._get_var_from_cf_time_repair,
                "CF time repair",
            ),
            (
                self._is_dimension_scalar_error,
                self._get_var_from_dimension_repair,
                "Dimension repair",
            ),
        )

    def _recover_variable(
        self,
        key,
        group,
        key_name,
        exc,
    ):
        """Attempt registered repair paths after a load failure."""
        logger.warning(
            "Primary load failed for %s: %s",
            key,
            exc,
        )

        for detector, repair, description in self._repair_handlers():

            if not detector(exc):
                continue

            logger.warning(
                "Attempting %s for %s",
                description,
                key,
            )

            try:

                return repair(
                    group,
                    key_name,
                )

            except Exception as repair_exc:

                logger.warning(
                    "%s failed for %s: %s",
                    description,
                    key,
                    repair_exc,
                )

        logger.warning(
            "Attempting netCDF4 backend fallback for %s",
            key,
        )

        return self._get_var_from_netcdf4(
            group,
            key_name,
        )

    def _get_variable(self, key, val):
        """Get a variable from the file."""
        if key in self.cached_file_content:
            return self.cached_file_content[key]

        parts = key.rsplit("/", 1)

        if len(parts) == 2:
            group, key_name = parts
        else:
            group = None
            key_name = key

        try:

            if self.file_handle is not None:
                result = self._get_var_from_filehandle(
                    group,
                    key_name,
                )
            else:
                result = self._get_var_from_xr(
                    group,
                    key_name,
                )

        except Exception as exc:

            result = self._recover_variable(
                key,
                group,
                key_name,
                exc,
            )

        self.cached_file_content[key] = result

        return result

    ##############################################

    def available_datasets(self, configured_datasets=None):
        """Report configured and dynamically discovered datasets."""
        handled_paths: set[str] = set()

        for is_available, configured_info in (
            configured_datasets or []
        ):
            ds_info = configured_info.copy()
            file_key = ds_info.get(
                "file_key",
                ds_info.get("name"),
            )

            normalised_file_key = (
                DatasetNameRegistry.normalise_path(file_key)
                if file_key is not None
                else None
            )

            if (
                self.file_type_matches(ds_info["file_type"])
                and normalised_file_key in self._records
            ):
                handled_paths.add(normalised_file_key)

                dynamic_name = self._name_registry.dataset_name(
                    normalised_file_key
                )
                dynamic_info = self._dataset_infos[
                    dynamic_name
                ]

                merged_info = dynamic_info.copy()
                merged_info.update(ds_info)
                merged_info["file_key"] = normalised_file_key

                yield True, merged_info
                continue

            yield is_available, ds_info

        for path, record in self._records.items():
            if path in handled_paths:
                continue

            yield True, self._dataset_infos[
                record.dataset_name
            ].copy()

        for ds_info in self._derived_dataset_infos.values():
            yield True, ds_info.copy()

    def get_dataset(self, ds_id, ds_info):
        """Load and normalise a dataset by its identifier and metadata configuration.

        This method extracts the target dataset from the file handler, standardises
        its dimensions, cleans up metadata attributes, and applies the expected
        Satpy dataset name.
        """

        derived_type = ds_info.get(
            "derived_type"
        )

        if derived_type is not None:
            return self._get_derived_dataset(
                ds_info
            )

        file_key = DatasetNameRegistry.normalise_path(
            ds_info.get("file_key", ds_id["name"])
        )

        logger.debug(
            "Loading dataset %s",
            ds_id["name"],
        )
        data = self[file_key]

        if data is None:
            raise RuntimeError(
                f"{file_key} returned None"
            )

        data = self._normalise_dimensions(data)

        attrs = AttributeNormalizer.normalise_attrs(data.attrs)

        attrs.update(
            self._public_dataset_metadata(ds_info)
        )

        data.attrs = attrs

        if data.name != ds_id["name"]:

            data = data.rename(ds_id["name"])

        return data

    def _get_indexed_variable_attrs(
        self,
        variable_path: str,
    ) -> dict[str, Any]:
        """Return attributes copied into Satpy's structural index."""
        attr_prefix = variable_path + "/attr/"

        raw_attrs = {
            key[len(attr_prefix):]: value
            for key, value in self.file_content.items()
            if key.startswith(attr_prefix)
        }

        return AttributeNormalizer.normalise_attrs(raw_attrs)

    def iter_dataset_infos(self):
        """Return cached discovery metadata."""
        yield from self._dataset_infos.items()

    def get_area_def(self, dsid):
        """Create a SwathDefinition for datasets with lon/lat coordinates."""
        ds_name = dsid["name"]

        if ds_name in self._derived_dataset_infos:
            return None

        record = self._records[
            self._name_registry.variable_path(ds_name)
        ]

        #
        # Geographic coordinate datasets do not themselves
        # have area definitions.
        #
        if (
            self._coordinate_resolver.geographic_role(record.attrs)
            is not None
        ):
            return None

        ds_info = self._dataset_infos.get(ds_name)

        if ds_info is None:
            return None

        coordinates = ds_info.get("coordinates")

        if not coordinates:
            return None

        lon_name, lat_name = coordinates

        lon_info = self._dataset_infos.get(lon_name)
        lat_info = self._dataset_infos.get(lat_name)

        if lon_info is None or lat_info is None:
            logger.warning(
                "Missing coordinate dataset metadata for %s "
                "(lon=%s, lat=%s)",
                ds_name,
                lon_name,
                lat_name,
            )
            return None

        lon_dsid = {"name": lon_name}
        lat_dsid = {"name": lat_name}

        lons = self.get_dataset(
            lon_dsid,
            lon_info,
        )

        lats = self.get_dataset(
            lat_dsid,
            lat_info,
        )

        #
        # TROPOMI-style coordinates:
        #
        #     (time, y, x)
        #
        # with time == 1.
        #
        if (
            lons.ndim == 3
            and lats.ndim == 3
            and lons.shape[0] == 1
            and lats.shape[0] == 1
        ):
            logger.info(
                "Collapsing singleton time dimension "
                "for coordinate pair %r/%r",
                lon_name,
                lat_name,
            )

            lons = lons.isel(time=0)
            lats = lats.isel(time=0)

        #
        # Reject anything pyresample still can't handle.
        #
        if lons.ndim > 2 or lats.ndim > 2:
            raise ValueError(
                "Cannot build SwathDefinition from "
                f"{lons.shape=} {lats.shape=}"
            )

        return SwathDefinition(
            lons=lons,
            lats=lats,
        )



    @staticmethod
    def _normalise_dimensions(data):
        """Rename dimensions to Satpy/xarray conventions."""
        rename_mapping = {
            old_name: new_name
            for old_name, new_name in DIMENSION_RENAMES.items()
            if old_name in data.dims
        }

        if rename_mapping:
            data = data.rename(rename_mapping)

        counts = Counter(data.dims)

        duplicates = {
            dim
            for dim, count in counts.items()
            if count > 1
        }

        if not duplicates:
            return data

        seen = Counter()
        new_dims = []

        for dim in data.dims:

            if dim not in duplicates:
                new_dims.append(dim)
                continue

            idx = seen[dim]

            if idx == 0:
                suffix = "x"
            elif idx == 1:
                suffix = "y"
            elif idx == 2:
                suffix = "z"
            else:
                suffix = str(idx)

            new_dims.append(
                f"{dim}_{suffix}"
            )

            seen[dim] += 1

        logger.warning(
            "Normalising duplicate dimensions %r -> %r",
            data.dims,
            tuple(new_dims),
        )

        return xr.DataArray(
            data=data.data,
            dims=tuple(new_dims),
            attrs=data.attrs,
            name=data.name,
        )

    @staticmethod
    def _complete_coordinate_standard_name(
        attrs: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Complete a missing latitude or longitude standard name."""
        result = dict(attrs)

        if result.get("standard_name"):
            return result

        units = str(
            result.get("units", "")
        ).strip().lower()

        if units in LATITUDE_UNITS:
            result["standard_name"] = "latitude"
        elif units in LONGITUDE_UNITS:
            result["standard_name"] = "longitude"

        return result

    @staticmethod
    def _public_dataset_metadata(
        ds_info: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return dataset metadata suitable for an xarray result."""
        internal_keys = {
            "file_key",
            "file_type",
            "source_dimensions",
            "source_shape",
            "source_coordinates",
        }

        return {
            key: value
            for key, value in ds_info.items()
            if key not in internal_keys
        }

    ####
    # Generic Metadata Repair Framework
    # ---------------------------------
    #
    # These repair paths address metadata and backend
    # interoperability issues observed across UVNS products.
    #
    # Existing repair paths:
    #
    # * CF time repair
    # * Dimension/scalar collision repair
    # * VLEN handling
    # * Compound dtype repair
    #
    def _is_time_decode_error(self, exc):
        msg = str(exc)

        return (
            "unable to decode time units" in msg
            or "decode_cf_datetime" in msg
            or "OutOfBoundsTimedelta" in msg
        )

    def _is_dimension_scalar_error(self, exc):

        return (
            "already exists as a scalar variable"
            in str(exc)
        )

    def _get_var_from_cf_time_repair(
        self,
        group,
        key,
        **kwargs_override,
    ):
        """Recover from CF time decoding failures caused by default fill values.

        This handles errors caused by undeclared default NetCDF fill values.
        """
        kwargs = dict(self._xarray_kwargs)
        kwargs.update(kwargs_override)

        #
        # Open raw.
        #
        kwargs["decode_times"] = False
        logger.warning(
            "Applying CF time repair for %s",
            key,
        )

        with xr.open_dataset(
            self.filename,
            group=group,
            **kwargs,
        ) as ds:

            ds = self._mask_implicit_fill_values(ds)

            #
            # Retry standard CF decoding.
            #
            ds = xr.decode_cf(ds)

            val = ds[key]

            #if not val.chunks or val.size < LAZY_LIMIT:
            #    val.load()
            val.load()
            val.attrs["_cf_time_repair"] = True

            return val

    def _mask_implicit_fill_values(self, ds):
        """Mask implicit NetCDF default fill values.

        This is only intended as preparation for CF decoding when
        decode_times=True has failed.
        """
        ds = ds.copy()

        for name in ds.data_vars:

            var = ds[name]

            #
            # Respect explicit metadata.
            #
            if (
                "_FillValue" in var.attrs
                or "missing_value" in var.attrs
            ):
                continue

            dtype = np.dtype(var.dtype)

            fill_key = _DTYPE_TO_FILL_KEY.get(dtype)

            if fill_key is None:
                continue

            fill_value = netCDF4.default_fillvals[fill_key]

            #
            # Cheap detection.
            #
            try:
                has_fill = bool(
                    (var == fill_value).any().compute()
                )
            except Exception:
                continue

            if not has_fill:
                continue

            logger.warning(
                "Masking implicit NetCDF fill value %r "
                "for variable %r",
                fill_value,
                name,
            )

            ds[name] = var.where(var != fill_value)

        return ds

    def _get_var_from_dimension_repair(
        self,
        group,
        key,
        **kwargs_override,
    ):
        bad_vars = self._remove_dimension_size_scalars(group)

        kwargs = dict(self._xarray_kwargs)
        kwargs.update(kwargs_override)

        logger.warning(
            "Applying dimension repair for %s",
            key,
        )

        with xr.open_dataset(
            self.filename,
            group=group,
            drop_variables=bad_vars,
            **kwargs,
        ) as ds:

            val = ds[key]

            if not val.chunks or val.size < LAZY_LIMIT:
                val.load()

            val.attrs["_dimension_repair"] = True

            return val


    def _remove_dimension_size_scalars(self, group):
        """Identify scalar variables that duplicate dimension sizes."""
        ds = netCDF4.Dataset(self.filename)

        try:

            g = ds if group is None else ds[group]

            bad_vars = []

            for name, var in g.variables.items():

                if name not in g.dimensions:
                    continue

                if var.shape != ():
                    continue

                try:
                    value = var[...].item()
                except Exception:
                    continue

                dim_size = len(g.dimensions[name])

                if value == dim_size:

                    logger.warning(
                        "Detected dimension-size scalar: %s=%s",
                        name,
                        value,
                    )

                    bad_vars.append(name)

            return bad_vars

        finally:
            ds.close()

    def _discover_wavelength_derivations(self):
        """Discover wavelength datasets."""

        derived = {}
        processed_bands = set()

        file_type = self.filetype_info["file_type"]

        for path, record in self._records.items():

            if record.name not in (
                "nominal_wavelength_coefficients",
                "calibrated_wavelength_coefficients",
            ):
                continue

            band_path = path.rsplit(
                "/instrument_data/",
                1,
            )[0]

            if band_path in processed_bands:
                continue

            processed_bands.add(band_path)

            prefix = band_path.replace("/", "__")

            nominal_path = (
                f"{band_path}/instrument_data/"
                "nominal_wavelength_coefficients"
            )

            calibrated_path = (
                f"{band_path}/instrument_data/"
                "calibrated_wavelength_coefficients"
            )

            nominal_valid = False
            calibrated_valid = False

            if nominal_path in self._records:
                nominal_valid = self._coefficients_have_data(
                    band_path,
                    "nominal_wavelength_coefficients",
                )

            if calibrated_path in self._records:
                calibrated_valid = self._coefficients_have_data(
                    band_path,
                    "calibrated_wavelength_coefficients",
                )

            if nominal_valid or calibrated_valid:
                derived[f"{prefix}__wavelength"] = {
                    "name": f"{prefix}__wavelength",
                    "file_type": file_type,
                    "derived_type": "best",
                    "band_path": band_path,
                }

            if nominal_valid:
                derived[f"{prefix}__nominal_wavelength"] = {
                    "name": f"{prefix}__nominal_wavelength",
                    "file_type": file_type,
                    "derived_type": "nominal",
                    "band_path": band_path,
                }

            if calibrated_valid:
                derived[f"{prefix}__calibrated_wavelength"] = {
                    "name": f"{prefix}__calibrated_wavelength",
                    "file_type": file_type,
                    "derived_type": "calibrated",
                    "band_path": band_path,
                }

        return derived

    def _spectral_channel_count(
        self,
        band_path,
    ):
        """Return spectral channel count for a band."""
        return self[
            f"{band_path}/spectral_channel"
        ].size

    def _coefficients_have_data(
        self,
        band_path,
        coefficient_name,
    ):
        """Return True if coefficient data contains valid values."""

        da = self[
            f"{band_path}/instrument_data/"
            f"{coefficient_name}"
        ]

        fill_value = da.attrs.get("_FillValue")
        valid_min = da.attrs.get("valid_min")
        valid_max = da.attrs.get("valid_max")

        data = da.data

        valid = np.isfinite(data)

        if fill_value is not None:
            valid &= (data != fill_value)

        if valid_min is not None:
            valid &= (data >= valid_min)

        if valid_max is not None:
            valid &= (data <= valid_max)

        return bool(valid.any().compute())

    def _get_coefficients(
        self,
        band_path,
        coefficient_name,
    ):
        """Load a coefficient variable."""
        data = self[
            f"{band_path}/instrument_data/"
            f"{coefficient_name}"
        ]

        if isinstance(data, xr.DataArray):
            return data.data

        return data

    def _select_wavelength_coefficients(
        self,
        band_path,
    ):
        """Select best available wavelength coefficients."""

        if self._coefficients_have_data(
            band_path,
            "calibrated_wavelength_coefficients",
        ):
            logger.info(
                "Using calibrated wavelength coefficients."
            )

            return (
                "calibrated_wavelength_coefficients",
                "calibrated",
            )

        logger.info(
            "Calibrated wavelength coefficients unavailable; "
            "using nominal wavelength coefficients."
        )

        return (
            "nominal_wavelength_coefficients",
            "nominal",
        )


    def _create_wavelength_dataset(
        self,
        *,
        band_path,
        coefficient_name,
        dataset_name,
        long_name,
        source,
    ):
        """Create a wavelength DataArray."""

        coeffs = self._get_coefficients(
            band_path,
            coefficient_name,
        )

        wavelengths = (
            DatasetDerivations.expand_wavelengths(
                coeffs,
                self._spectral_channel_count(
                    band_path
                ),
            )
        )

        return xr.DataArray(
            wavelengths,
            dims=(
                "spectral_channel",
                "y",
                "x",
            ),
            name=dataset_name,
            attrs = {
                "long_name": long_name,
                "wavelength_source": source,
                "derived_from": coefficient_name,
            },
        )

    def _get_nominal_wavelength(
        self,
        band_path,
        dataset_name,
    ):
        """Generate wavelengths from nominal coefficients."""

        return self._create_wavelength_dataset(
            band_path=band_path,
            coefficient_name="nominal_wavelength_coefficients",
            dataset_name=dataset_name,
            long_name="Nominal wavelength",
            source="nominal",
        )


    def _get_calibrated_wavelength(
        self,
        band_path,
        dataset_name,
    ):
        """Generate wavelengths from calibrated coefficients."""

        coeffs = self._get_coefficients(
            band_path,
            "calibrated_wavelength_coefficients",
        )

        if not self._coefficients_have_data(
                band_path,
                "calibrated_wavelength_coefficients",
        ):

            raise KeyError(
                "No calibrated wavelength coefficients available."
            )

        return self._create_wavelength_dataset(
            band_path=band_path,
            coefficient_name="calibrated_wavelength_coefficients",
            dataset_name=dataset_name,
            long_name="Calibrated wavelength",
            source="calibrated",
        )


    def _get_best_wavelength(
        self,
        band_path,
        dataset_name,
    ):
        """Generate wavelengths using the preferred solution."""

        coefficient_name, source = (
            self._select_wavelength_coefficients(
                band_path
            )
        )

        return self._create_wavelength_dataset(
            band_path=band_path,
            coefficient_name=coefficient_name,
            dataset_name=dataset_name,
            long_name="Wavelength",
            source=source,
        )


    def _get_derived_dataset(
        self,
        ds_info,
    ):
        """Load a derived dataset."""

        band_path = ds_info["band_path"]
        dataset_name = ds_info["name"]

        match ds_info["derived_type"]:

            case "best":
                return self._get_best_wavelength(
                    band_path,
                    dataset_name,
                )

            case "nominal":
                return self._get_nominal_wavelength(
                    band_path,
                    dataset_name,
                )

            case "calibrated":
                return self._get_calibrated_wavelength(
                    band_path,
                    dataset_name,
                )

        raise KeyError(
            ds_info["derived_type"]
        )


#################################################
