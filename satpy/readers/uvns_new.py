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

import numpy as np
import xarray as xr
import dask.array as da
import netCDF4
import satpy

from satpy.readers.core.netcdf import NetCDF4FileHandler
from satpy.readers.core.netcdf import H5NetcdfAccessor

from pyresample.geometry import SwathDefinition

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




@dataclass(frozen=True)
class VariableRecord:
    """Structural and metadata description of one file variable."""

    path: str
    name: str
    dataset_name: str
    dimensions: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    attrs: Mapping[str, Any]


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
    """Map complete internal variable paths to unique Satpy dataset names."""

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
    """Resolve explicit geographic coordinate references."""

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
        """
        Resolve a coordinate reference.

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


class UVNSFileHandler(NetCDF4FileHandler):
    """Dynamically discover and load UVNS-family file variables."""

    def __init__(
        self,
        filename,
        filename_info,
        filetype_info,
        engine="netcdf4",
    ):
        """Initialise the file handler and build the dynamic registry."""

        engine="h5netcdf"
        xarray_kwargs={'chunks':None}
        super().__init__(
            filename,
            filename_info,
            filetype_info,
            engine=engine,
        )

        print(f'Filename: {filename}')
        self._records = self._build_variable_records()
        self._name_registry = DatasetNameRegistry(self._records)
        self._records = self._assign_dataset_names(self._records)
        self._coordinate_resolver = CoordinateResolver(
            self._records,
            self._name_registry,
        )
        self._dataset_infos = self._build_dataset_infos()

    def _get_fallback_handle(self):
        if not hasattr(self, "_fallback_file_handle"):
            self._fallback_accessor = H5NetcdfAccessor()
            self._fallback_file_handle = (
                self._fallback_accessor.create_file_handle(self.filename)
            )
        return self._fallback_file_handle

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

            records[path] = VariableRecord(
                path=path,
                name=path.rsplit("/", 1)[-1],
                dataset_name="",
                dimensions=dimensions,
                shape=shape,
                dtype=dtype,
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


    ###### Overrideen inheritance methods. #######
    ######
    ######
    def _get_var_from_xr(self, group, key, **kwargs_override):

        kwargs = dict(self._xarray_kwargs)
        kwargs.update(kwargs_override)

        with xr.open_dataset(
            self.filename,
            group=group,
            **kwargs,
        ) as nc:

            val = nc[key]

            if not val.chunks:
                val.load()

        return val

    def _get_var_from_netcdf4(self, group, key):
        return self._get_var_from_xr(
            group,
            key,
            engine="netcdf4",
        )

    '''
    def _get_var_from_hdf5(self, group, key):
        """Load a variable directly from the underlying HDF5/NetCDF4 file.

        Used as a fallback when the normal NetCDF4/xarray loading path
        fails for unsupported datatypes or metadata.
        """
        print('_get_var_from_hdf5')
        print('arr = xr.DataArray(v[:], ...')

        fh       = self._get_fallback_handle()

        if group is None:
            g = fh
        else:
            g = fh[group]

        v = g[key]

        accessor = self._fallback_accessor
        attrs = accessor.get_object_attrs(v)

        arr = xr.DataArray(
            da.from_array(v),
            dims=v.dimensions,
            attrs=attrs,
            name=v.name,
        )

        self._fallback_file_handle.close()
        del self._fallback_file_handle

        print("FALLBACK CLOSED")

        arr.attrs["_fallback"] = True

        return arr
    '''

    def _check_var_validity(self, key):
        v = self.file_content[key]

        print("VAR TYPE:", type(v))

        try:
            print("VAR NAME:", v.name)
            print("NAME OK")
        except Exception as e:
            print("NAME FAILED:", e)

        try:
            print("VAR SHAPE:", v.shape)
            print("SHAPE OK")
        except Exception as e:
            print("SHAPE FAILED:", e)

        try:
            print("VAR DTYPE:", v.dtype)
            print("DTYPE OK")
        except Exception as e:
            print("DTYPE FAILED:", e)

        try:
            print("VAR DIMENSIONS:", v.dimensions)
            print("DIMS OK")
        except Exception as e:
            print("DIMS FAILED:", e)

            print("GROUP:", v.group())

        print("GROUP PATH:", v.group().path)

        try:
            print("FILEPATH:", v.group().filepath())
        except Exception as e:
            print("FILEPATH FAILED:", e)


        grp = v.group()

        print(type(grp))

        try:
            print(grp.variables.keys())
            print("GROUP LIVE")
        except Exception as e:
            print("GROUP DEAD", e)

        try:
            print(v[:5])
            print("READ OK")
        except Exception as e:
            print("READ FAIL", e)


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

            logger.warning(
                "Primary load failed for %s: %s",
                key,
                exc,
            )

            #
            # First attempt: CF time repair.
            #
            if self._is_time_decode_error(exc):

                logger.warning(
                    "Attempting CF time repair for %s",
                    key,
                )

                try:

                    result = self._get_var_from_cf_time_repair(
                        group,
                        key_name,
                    )

                except Exception as repair_exc:

                    logger.warning(
                        "CF time repair failed for %s: %s",
                        key,
                        repair_exc,
                    )

                else:

                    self.cached_file_content[key] = result
                    return result

            #
            # Second attempt: dimension/scalar collision repair.
            #
            elif self._is_dimension_scalar_error(exc):

                logger.warning(
                    "Attempting dimension repair for %s",
                    key,
                )

                try:

                    result = self._get_var_from_dimension_repair(
                        group,
                        key_name,
                    )

                except Exception as repair_exc:

                    logger.warning(
                        "Dimension repair failed for %s: %s",
                        key,
                        repair_exc,
                    )

                else:

                    self.cached_file_content[key] = result
                    return result

            #
            # Third attempt: netCDF4 backend.
            #
            try:

                result = self._get_var_from_netcdf4(
                    group,
                    key_name,
                )

            except Exception as exc2:

                logger.warning(
                    "netCDF4 retry failed for %s: %s",
                    key,
                    exc2,
                )

                raise


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

    def get_dataset(self, ds_id, ds_info):
        file_key = DatasetNameRegistry.normalise_path(
            ds_info.get("file_key", ds_id["name"])
        )

        print("GET_DATASET START", ds_id["name"])
        print("FILE KEY", file_key)

        print("BEFORE self[file_key]")
        data = self[file_key]
        if data is None:
            raise RuntimeError(
                f"{file_key} returned None"
            )
        print("AFTER self[file_key]")

        print("BEFORE _normalise_dimensions")
        data = self._normalise_dimensions(data)
        print("AFTER _normalise_dimensions")

        print("BEFORE attr normalisation")
        attrs = AttributeNormalizer.normalise_attrs(data.attrs)
        print("AFTER attr normalisation")

        attrs.update(
            self._public_dataset_metadata(ds_info)
        )

        print("BEFORE assign attrs")
        data.attrs = attrs
        print("AFTER assign attrs")

        if data.name != ds_id["name"]:
            print("BEFORE rename")
            data = data.rename(ds_id["name"])
            print("AFTER rename")

        print("GET_DATASET END", ds_id["name"])

        return data

    '''
    def get_dataset(self, ds_id, ds_info):
        """Load one previously discovered dataset."""
        file_key = DatasetNameRegistry.normalise_path(
            ds_info.get("file_key", ds_id["name"])
        )

        logger.debug(
            "Loading UVNS dataset %r from %r",
            ds_id["name"],
            file_key,
        )

        data = self[file_key]

        data = self._normalise_dimensions(data)

        attrs = AttributeNormalizer.normalise_attrs(
            data.attrs
        )
        attrs.update(
            self._public_dataset_metadata(ds_info)
        )

        if (
            file_key
            in self._coordinate_resolver.referenced_coordinate_paths
        ):
            attrs = self._complete_coordinate_standard_name(
                attrs
            )

        data.attrs = attrs

        if data.name != ds_id["name"]:
            data = data.rename(ds_id["name"])

        return data
    '''

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
        print("GET_AREA_DEF CALLED")
        print(dsid)

        ds_name = dsid["name"]

        record = self._records[
            self._name_registry.variable_path(ds_name)
        ]


        if (
                self._coordinate_resolver.geographic_role(record.attrs)
                is not None
        ):
            print("GEOGRAPHIC COORDINATE -> NO AREA")
            print("RETURNING NONE AREA FOR", ds_name)
            return None

        ds_info = self._dataset_infos.get(ds_name)
        if ds_info is None:
            print("RETURNING NONE AREA FOR DS_INFO", ds_name)
            return None

        coordinates = ds_info.get("coordinates")
        if not coordinates:
            print("RETURNING NONE AREA FOR COORDINATES", ds_name)
            return None

        lon_name, lat_name = coordinates

        lon_info = self._dataset_infos.get(lon_name)
        lat_info = self._dataset_infos.get(lat_name)

        if lon_info is None or lat_info is None:
            print("RETURNING NONE AREA FOR LON/LAT INFO", ds_name)
            return None

        lon_dsid = {"name": lon_name}
        lat_dsid = {"name": lat_name}

        lons = self.get_dataset(lon_dsid, lon_info)
        lats = self.get_dataset(lat_dsid, lat_info)

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
        """Rename supported spatial dimensions to Satpy conventions."""
        rename_mapping = {
            old_name: new_name
            for old_name, new_name in DIMENSION_RENAMES.items()
            if old_name in data.dims
        }

        if not rename_mapping:
            return data

        return data.rename(rename_mapping)

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

    ### Specific Modifications and Repairs
    ###
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
        """Recover from CF time decoding failures caused by
        undeclared default NetCDF fill values.
        """

        kwargs = dict(self._xarray_kwargs)
        kwargs.update(kwargs_override)

        #
        # Open raw.
        #
        kwargs["decode_times"] = False

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

            if not val.chunks:
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

        with xr.open_dataset(
            self.filename,
            group=group,
            drop_variables=bad_vars,
            **kwargs,
        ) as ds:

            val = ds[key]

            if not val.chunks:
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
