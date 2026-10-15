"""Writer for Zarr stores.

The ``zarr`` writer stores Satpy datasets with :meth:`xarray.Dataset.to_zarr`. Data are written as they
are in the Scene: no enhancement is applied, and dtypes, units and fill values are kept.

Example usage
-------------

All datasets are saved as variables of one store, so they must share the same grid (resample the Scene
first if they do not). The default filename pattern, ``{name}_{start_time:%Y%m%d_%H%M%S}.zarr``, only
works when saving a single dataset, so specify a ``filename`` for stores with several variables:

    >>> scn.save_datasets(writer="zarr", filename="/data/zarr/{platform_name}_{start_time:%Y%m%d_%H%M%S}.zarr")

The filename is formatted with the metadata of the saved datasets as combined by
:func:`~satpy.dataset.metadata.combine_metadata`: only attributes with the same value in every dataset
are available, ``start_time`` is the earliest and ``end_time`` the latest of all datasets, and ``name``
is only available when saving one dataset.

A filename ending in ``.zarr`` selects this writer automatically:

    >>> scn.save_dataset("C13", filename="/data/zarr/{platform_name}_{name}.zarr")

Options of :meth:`xarray.Dataset.to_zarr` that the writer does not expose directly can be passed with
``xarray_kwargs``; they override anything the writer chose:

    >>> scn.save_datasets(writer="zarr", xarray_kwargs={"write_empty_chunks": False, "safe_chunks": False})

Before writing, the writer removes Satpy-internal attributes (``area``, ``_satpy_id``, ``None`` values),
converts the remaining attributes to JSON (datetimes become ISO strings, sets become lists), drops the
``crs`` coordinate, adds CF ``standard_name``/``units`` to ``x`` and ``y``, discards any encoding
carried over from the reader and rechunks irregular dask chunks, which zarr cannot store.

Separate stores and remote targets
----------------------------------

``split_variables=True`` writes every dataset to its own store instead, named with the default pattern:

    >>> scn.save_datasets(writer="zarr", split_variables=True, base_dir="/data/zarr")

To save other combinations of datasets, call ``save_datasets`` once per store and select the datasets
with ``datasets=``. Datasets on different grids can also share one store as separate zarr groups, using
the ``group`` option of :meth:`xarray.Dataset.to_zarr`:

    >>> scn.save_datasets(writer="zarr", datasets=["C01", "C03"], filename="/data/zarr/abi.zarr",
    ...                   xarray_kwargs={"group": "1km"})
    >>> scn.save_datasets(writer="zarr", datasets=["C13"], filename="/data/zarr/abi.zarr", mode="a",
    ...                   xarray_kwargs={"group": "2km"})

Filenames may be fsspec URLs, for example on S3 (requires ``s3fs``), with ``storage_options`` passed to
fsspec:

    >>> scn.save_datasets(writer="zarr", filename="s3://bucket/satpy/{name}.zarr",
    ...                   storage_options={"anon": False})

Instead of a filename, ``store`` accepts any zarr ``StoreLike`` (for example
:class:`zarr.storage.MemoryStore`). With ``split_variables=True`` and more than one dataset, ``store``
must be a dictionary of stores keyed by dataset name. A plain ``dict`` is always interpreted as this
mapping; wrap a dict-backed store in :class:`zarr.storage.MemoryStore`.

Zarr format and consolidated metadata
-------------------------------------

``zarr_format`` defaults to the ``zarr_format`` of the writer YAML configuration, if set, and otherwise
to the default of the installed zarr-python (3 for zarr-python 3). Consolidated metadata is written for
zarr format 2 and not for format 3, where it is not part of the specification. Both can be changed
with ``zarr_format=`` and ``consolidated=``.

Appending
---------

``append_dim="time"`` adds a length-1 ``time`` dimension, using the earliest ``start_time`` of the
saved datasets, to data that do not have one and appends along it. If the store does not exist yet it
is created, so the same call works for the first and every following write:

    >>> for filenames in files_per_time_step:
    ...     scn = Scene(filenames, reader="abi_l1b")
    ...     scn.load(["C13"])
    ...     scn.save_datasets(writer="zarr", filename="/data/zarr/abi_c13.zarr", append_dim="time")

Times are stored as microseconds since 1970-01-01, so later appends with sub-second times round-trip
exactly; pass ``encoding={"time": {...}}`` to override. When appending, the zarr format and the
consolidated metadata setting are taken from the existing store and ``zarr_format`` is ignored. New data
are rechunked to the store's chunks (``align_chunks=True``). Appending the same time twice is not
detected.

Without ``append_dim``, ``mode`` keeps xarray's default: a new store is created and writing to an
existing one fails. Use ``mode="w"`` to overwrite.

Delayed writing
---------------

With ``compute=False`` a list of :class:`dask.delayed.Delayed` objects is returned, to be passed to
:func:`~satpy.writers.core.compute.compute_writer_results`. xarray writes the store metadata and
in-memory coordinates immediately, so a store whose delayed result is never computed only contains
metadata and blocks a later write without ``mode="w"``.

Icechunk
--------

An `Icechunk <https://icechunk.io>`_ ``Session`` can be passed as ``store``. Icechunk requires zarr
format 3, so ``zarr_format`` is forced to 3 and consolidated metadata is disabled. With ``commit``, the
session is committed after the data are written, inside the same dask computation:

    >>> import icechunk
    >>> repo = icechunk.Repository.open_or_create(icechunk.local_filesystem_storage("/data/repo"))
    >>> session = repo.writable_session("main")
    >>> scn.save_datasets(writer="zarr", store=session, datasets=["C07", "C13"], commit="Add ABI scan")

Without ``commit`` the changes stay uncommitted; call ``session.commit("message")`` yourself after
computing. A committed session is read-only, so every append needs a new ``repo.writable_session()``.
The writable session cannot be pickled, so Icechunk targets only work with dask's threaded scheduler,
not with ``dask.distributed``.

"""
from __future__ import annotations

import copy
import datetime as dt
import importlib.metadata
import json
import logging
import os
import typing
import warnings
from dataclasses import dataclass

import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from packaging.version import Version
from trollsift import compose

from satpy.cf.coords import add_xy_coords_attrs
from satpy.dataset.metadata import combine_metadata
from satpy.utils import chunks_are_irregular, regular_chunks_from_irregular_chunks
from satpy.writers.core.base import Writer
from satpy.writers.core.compute import compute_writer_results

if typing.TYPE_CHECKING:
    from collections.abc import Hashable, Iterable
    from os import PathLike
    from typing import Any

    from dask.delayed import Delayed

LOG = logging.getLogger(__name__)

_SATPY_KWARGS = ("overlay", "decorate", "config_files")
_APPEND_MODES = ("a", "a-", "r+")
_MIN_VERSIONS = {"zarr": Version("3.0.0"), "xarray": Version("2025.6.0")}
# Without explicit units, xarray picks "days since <first time>" and later sub-second appends
# cannot be decoded anymore (specs/zarr-writer.md, Background item 3).
_TIME_ENCODING = {
    "units": "microseconds since 1970-01-01 00:00:00",
    "calendar": "proleptic_gregorian",
    "dtype": "int64",
}


@dataclass
class _Target:
    """One zarr store written by one ``to_zarr`` call."""

    name: str
    data_arrays: list[xr.DataArray]
    attrs: dict
    store: Any = None
    session: Any = None


@dataclass(frozen=True)
class _StoreState:
    """Metadata of an existing store that an append has to match."""

    zarr_format: int
    consolidated: bool
    variables: frozenset[str]


class ZarrWriter(Writer):
    """Write datasets to Zarr stores with :meth:`xarray.Dataset.to_zarr`.

    See the :mod:`module documentation <satpy.writers.zarr>` for usage examples.

    """

    def __init__(self, name=None, filename=None, base_dir=None, **kwargs):
        """Initialize the writer and load its YAML configuration."""
        super().__init__(name=name, filename=filename, base_dir=base_dir,
                         default_config_filename="writers/zarr.yaml", **kwargs)

    def get_filename(self, **kwargs):
        """Compose the output filename, without creating local directories for URL patterns."""
        if self.filename_parser is not None and _is_url(self.filename_parser.fmt):
            self._prepare_metadata_for_filename_formatting(kwargs)
            return self.filename_parser.compose(kwargs)
        return super().get_filename(**kwargs)

    def save_dataset(
            self,
            dataset: xr.DataArray,
            filename: str | PathLike | None = None,
            fill_value: float | int | None = None,
            compute: bool = True,
            units: str | None = None,
            **kwargs,
    ) -> list[Delayed] | list[Any]:
        """Save one dataset to a zarr store.

        See :meth:`save_datasets` for the keyword arguments. Unit conversion is not supported, so
        ``units`` is ignored with a warning like other unknown keyword arguments.

        """
        if units is not None:
            kwargs["units"] = units
        return self.save_datasets([dataset], compute=compute, filename=filename, fill_value=fill_value, **kwargs)

    def save_datasets(
            self,
            datasets: Iterable[xr.DataArray],
            compute: bool = True,
            *,
            filename: str | PathLike | None = None,
            store: Any = None,
            storage_options: dict | None = None,
            split_variables: bool = False,
            commit: str | None = None,
            fill_value: float | int | None = None,
            append_dim: Hashable | None = None,
            mode: str | None = None,
            zarr_format: int | None = None,
            consolidated: bool | None = None,
            encoding: dict[str, dict] | None = None,
            xarray_kwargs: dict | None = None,
            **kwargs,
    ) -> list[Delayed] | list[Any]:
        """Save datasets to one or more zarr stores.

        Args:
            datasets: DataArrays to write.
            compute: Write the data now. If ``False``, return a list of
                :class:`~dask.delayed.Delayed` objects instead.
            filename: Local path, fsspec URL or trollsift pattern of the store. Formatted with the
                metadata shared by the datasets of the store (see
                :func:`~satpy.dataset.metadata.combine_metadata`); ``name`` is only available for
                stores holding one dataset. Defaults to the ``filename`` pattern given when creating
                the writer.
            store: A zarr ``StoreLike`` or Icechunk ``Session`` to write to instead of ``filename``.
                With ``split_variables``, a dict of them keyed by dataset name.
            storage_options: fsspec options for remote URL targets.
            split_variables: Write every dataset to its own store instead of all datasets to one store.
            commit: Commit message for Icechunk sessions passed as ``store``. The commit happens after
                the data are written.
            fill_value: ``_FillValue`` encoding of all data variables.
            append_dim: Dimension to append along. ``"time"`` is added from ``start_time`` when missing.
            mode: Passed to :meth:`xarray.Dataset.to_zarr`. ``None`` creates a new store, or appends
                when ``append_dim`` is given.
            zarr_format: Zarr format (2 or 3) of new stores.
            consolidated: Write consolidated metadata. Defaults to ``True`` for format 2 and ``False``
                for format 3, or to the existing store's state when appending.
            encoding: Per-variable encoding for :meth:`xarray.Dataset.to_zarr`, merged over the
                writer's defaults. Variables with ``chunks`` here are not rechunked by the writer.
            xarray_kwargs: Other :meth:`xarray.Dataset.to_zarr` keyword arguments. They override the
                arguments chosen by the writer, so ``encoding`` passed here replaces the writer's
                encoding entirely.
            kwargs: Satpy keyword arguments that do not apply to this writer (``overlay``,
                ``decorate``) are ignored; anything else is ignored with a warning.

        Returns:
            The computed results if ``compute`` is ``True``, else a list of Delayed objects.

        """
        _check_versions()
        _warn_unknown_kwargs(kwargs)
        xarray_kwargs = dict(xarray_kwargs or {})
        targets = _make_targets(datasets, split_variables)
        self._resolve_stores(targets, filename, store, split_variables)
        _check_commit(targets, commit)
        delayeds = []
        for target in targets:
            target_storage_options = _target_storage_options(target, storage_options)
            existing = None
            if append_dim is not None or mode in _APPEND_MODES or "region" in xarray_kwargs:
                existing = _existing_store_state(target.store, xarray_kwargs.get("group"), target_storage_options)
            ds = _build_dataset(target.data_arrays)
            ds = _maybe_expand_append_dim(ds, append_dim, target.attrs)
            ds = _regularize_chunks(ds, encoding)
            target_encoding = _build_encoding(ds, encoding, fill_value, existing)
            ds = _drop_conflicting_fill_value_attrs(ds, target_encoding, existing)
            to_zarr_kwargs = self._resolve_to_zarr_kwargs(
                target, existing, mode=mode, append_dim=append_dim, zarr_format=zarr_format,
                consolidated=consolidated, storage_options=target_storage_options, encoding=target_encoding,
                xarray_kwargs=xarray_kwargs)
            delayeds.append(_write(ds, target.store, to_zarr_kwargs))
        delayeds = _chain_commits(delayeds, targets, commit)
        if compute:
            return compute_writer_results([delayeds])
        return delayeds

    def _resolve_stores(self, targets: list[_Target], filename, store, split_variables: bool) -> None:
        if store is None:
            for target in targets:
                target.store = self._target_filename(filename, target)
            _check_unique_filenames(targets)
            return
        if filename is not None:
            raise ValueError("'filename' and 'store' cannot be used together.")
        _assign_stores(targets, store, split_variables)

    def _target_filename(self, filename, target: _Target) -> str:
        attrs = dict(target.attrs)
        if len(target.data_arrays) == 1:
            attrs["name"] = target.name
        try:
            if filename is None:
                return self.get_filename(**attrs)
            self._prepare_metadata_for_filename_formatting(attrs)
            return compose(os.fspath(filename), attrs)
        except KeyError as err:
            if len(target.data_arrays) == 1:
                raise ValueError(f"Cannot create the zarr filename, the dataset has no {err} attribute.") from err
            raise ValueError(f"Cannot create the zarr filename, the datasets have no common {err} attribute. "
                             "Only attributes equal in all datasets of a store, never 'name', can be used, so "
                             "specify a 'filename' for multi-variable zarr stores or use split_variables=True."
                             ) from err

    def _resolve_to_zarr_kwargs(self, target: _Target, existing: _StoreState | None, *, mode, append_dim,
                                zarr_format, consolidated, storage_options, encoding, xarray_kwargs) -> dict:
        is_icechunk = _is_icechunk(target.store)
        if is_icechunk and zarr_format == 2:
            raise ValueError(f"Icechunk only supports zarr format 3, but zarr_format=2 was requested for "
                             f"{target.name!r}.")
        kwargs: dict[str, Any] = {"mode": mode, "encoding": encoding, "align_chunks": True, "compute": False}
        if storage_options is not None:
            kwargs["storage_options"] = storage_options
        # xarray fails to append to a store that does not exist yet or that has another zarr_format
        if existing is None:
            target_format = zarr_format or self.info.get("zarr_format") or _default_zarr_format()
            default_consolidated = target_format == 2
        else:
            target_format, default_consolidated = existing.zarr_format, existing.consolidated
            kwargs["append_dim"] = append_dim
        if consolidated is None:
            consolidated = default_consolidated
        if is_icechunk:
            target_format, consolidated = 3, False
        kwargs.update(zarr_format=target_format, consolidated=consolidated)
        kwargs.update(xarray_kwargs)
        return kwargs


def _check_versions() -> None:
    for package, min_version in _MIN_VERSIONS.items():
        version = _package_version(package)
        if version < min_version:
            raise ImportError(f"The 'zarr' writer requires {package}>={min_version}, but {package} {version} "
                              "is installed.")


def _package_version(package: str) -> Version:
    return Version(importlib.metadata.version(package))


def _default_zarr_format() -> int:
    import zarr

    return zarr.config.get("default_zarr_format")


def _warn_unknown_kwargs(kwargs: dict) -> None:
    unknown = sorted(key for key in kwargs if key not in _SATPY_KWARGS)
    if unknown:
        warnings.warn(f"The 'zarr' writer ignores these unknown keyword arguments: {unknown}", stacklevel=3)


def _variable_name(data_arr: xr.DataArray) -> str:
    name = data_arr.attrs.get("name")
    if name is None and isinstance(data_arr.name, str):
        name = data_arr.name
    if name is None:
        LOG.warning("DataArray has no 'name' attribute, writing it as 'data'.")
        name = "data"
    return str(name)


def _make_targets(datasets: Iterable[xr.DataArray], split_variables: bool) -> list[_Target]:
    data_arrays = list(datasets)
    if not data_arrays:
        return []
    if split_variables:
        return [_Target(_variable_name(data_arr), [data_arr], combine_metadata(data_arr))
                for data_arr in data_arrays]
    names = [_variable_name(data_arr) for data_arr in data_arrays]
    return [_Target(", ".join(names), data_arrays, combine_metadata(*data_arrays))]


def _check_unique_filenames(targets: list[_Target]) -> None:
    seen: dict[str, str] = {}
    for target in targets:
        if target.store in seen:
            raise ValueError(f"Datasets {seen[target.store]!r} and {target.name!r} would both be written to "
                             f"{target.store!r}. Include '{{name}}' in the filename.")
        seen[target.store] = target.name


def _assign_stores(targets: list[_Target], store, split_variables: bool) -> None:
    if isinstance(store, dict):
        if not split_variables:
            raise ValueError("A dict of stores requires split_variables=True; otherwise all datasets are "
                             "written to one store.")
        target_names = [target.name for target in targets]
        if len(set(target_names)) != len(target_names):
            raise ValueError(f"Dataset names are not unique, cannot assign stores by name: {target_names}")
        for target in targets:
            if target.name not in store:
                raise ValueError(f"No store given for dataset {target.name!r}.")
            target.store = store[target.name]
    elif len(targets) == 1:
        targets[0].store = store
    else:
        raise ValueError(f"With split_variables=True, a single 'store' cannot hold the {len(targets)} datasets "
                         f"{[target.name for target in targets]}. Pass a dict of stores keyed by dataset name.")
    for target in targets:
        if _is_icechunk_session(target.store):
            target.session = target.store
            target.store = target.session.store


def _check_commit(targets: list[_Target], commit: str | None) -> None:
    if commit is None:
        return
    for target in targets:
        if target.session is None:
            raise ValueError(f"'commit' requires an Icechunk Session as store, but {target.name!r} is not "
                             "written to one.")


def _is_url(target) -> bool:
    return isinstance(target, (str, os.PathLike)) and "://" in os.fspath(target)


def _is_remote(target) -> bool:
    return _is_url(target) and not os.fspath(target).startswith(("file://", "memory://"))


def _target_storage_options(target: _Target, storage_options: dict | None) -> dict | None:
    if storage_options is None or _is_remote(target.store):
        return storage_options
    LOG.debug("Ignoring 'storage_options' for target %r, which is not a remote URL.", target.name)
    return None


def _is_icechunk(obj) -> bool:
    return type(obj).__module__.split(".")[0] == "icechunk"


def _is_icechunk_session(obj) -> bool:
    return _is_icechunk(obj) and hasattr(obj, "commit") and hasattr(obj, "store")


def _existing_store_state(store, group: str | None, storage_options: dict | None) -> _StoreState | None:
    import zarr

    open_kwargs = {} if storage_options is None else {"storage_options": storage_options}
    try:
        zarr_group = zarr.open_group(store, mode="r", path=group, **open_kwargs)
    except (FileNotFoundError, zarr.errors.NodeNotFoundError):
        return None
    metadata = zarr_group.metadata
    return _StoreState(metadata.zarr_format, metadata.consolidated_metadata is not None,
                       frozenset(zarr_group.array_keys()))


def _build_dataset(data_arrays: list[xr.DataArray]) -> xr.Dataset:
    prepared: dict[str, xr.DataArray] = {}
    for data_arr in data_arrays:
        name = _variable_name(data_arr)
        if name in prepared:
            raise ValueError(f"More than one dataset named {name!r} in one store.")
        prepared[name] = _prepare_data_array(data_arr)
    try:
        xr.align(*prepared.values(), join="exact")
        ds = xr.Dataset(prepared)
    except ValueError as err:
        raise ValueError(f"Datasets {list(prepared)} cannot be written to one store because their coordinates "
                         "differ. Resample them to a common area, use split_variables=True, or save them with "
                         "separate save_datasets calls.") from err
    ds.attrs["history"] = f"Created by pytroll/satpy on {dt.datetime.now(dt.timezone.utc)}"
    return ds.drop_encoding()


def _prepare_data_array(data_arr: xr.DataArray) -> xr.DataArray:
    data_arr = add_xy_coords_attrs(data_arr.copy(deep=False))
    data_arr.attrs = _sanitize_attrs(data_arr.attrs)
    for coord in data_arr.coords.values():
        coord.attrs = _json_safe(coord.attrs)
    return data_arr


def _sanitize_attrs(attrs: dict) -> dict:
    attrs = {key: val for key, val in attrs.items()
             if val is not None and key not in ("area", "_last_resampler") and not key.startswith("_satpy")}
    ancillary_variables = attrs.pop("ancillary_variables", None)
    if ancillary_variables:
        attrs["ancillary_variables"] = [
            anc.attrs.get("name", anc.name) if isinstance(anc, xr.DataArray) else anc
            for anc in ancillary_variables
        ]
    return _json_safe(attrs)


def _json_safe(attrs: dict) -> dict:
    return json.loads(json.dumps(attrs, default=_json_default))


def _json_default(obj):
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()
    if isinstance(obj, np.datetime64):
        return np.datetime_as_string(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=str)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    return str(obj)


def _maybe_expand_append_dim(ds: xr.Dataset, append_dim, attrs: dict) -> xr.Dataset:
    if append_dim is None or append_dim in ds.dims:
        return ds
    if append_dim in ds.coords:
        return ds.expand_dims(append_dim)
    if append_dim != "time":
        raise ValueError(f"append_dim={append_dim!r} is not a dimension of the data; only 'time' can be added "
                         "by the writer.")
    start_time = attrs.get("start_time")
    if start_time is None:
        raise ValueError("append_dim='time' needs a 'start_time' attribute to create the time coordinate.")
    return ds.expand_dims(time=[_time_value(start_time)])


def _time_value(start_time) -> np.datetime64:
    timestamp = pd.Timestamp(start_time)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.to_datetime64().astype("datetime64[us]")


def _regularize_chunks(ds: xr.Dataset, encoding: dict | None) -> xr.Dataset:
    encoding = encoding or {}
    data_vars, coords = {}, {}
    for name, var in ds.variables.items():
        if not isinstance(var.data, da.Array) or "chunks" in encoding.get(name, {}):
            continue
        if not chunks_are_irregular(var.data.chunks):
            continue
        rechunked = (var.dims, var.data.rechunk(regular_chunks_from_irregular_chunks(var.data.chunks)), var.attrs)
        if name in ds.coords:
            coords[name] = rechunked
        else:
            data_vars[name] = rechunked
    return ds.assign(data_vars).assign_coords(coords)


def _build_encoding(ds: xr.Dataset, user_encoding: dict | None, fill_value, existing: _StoreState | None) -> dict:
    user_encoding = copy.deepcopy(user_encoding or {})
    # xarray refuses encoding for variables that already exist in the store
    stored = existing.variables if existing is not None else frozenset()
    encoding: dict[str, dict] = {}
    if _is_datetime_variable(ds, "time") and "time" not in user_encoding and "time" not in stored:
        encoding["time"] = dict(_TIME_ENCODING)
    if fill_value is not None:
        for name, var in ds.data_vars.items():
            if name not in stored:
                encoding[name] = {"_FillValue": np.asarray(fill_value, dtype=var.dtype).item()}
    for name, var_encoding in user_encoding.items():
        encoding.setdefault(name, {}).update(var_encoding)
    return encoding


def _is_datetime_variable(ds: xr.Dataset, name: str) -> bool:
    return name in ds.variables and np.issubdtype(ds[name].dtype, np.datetime64)


def _drop_conflicting_fill_value_attrs(ds: xr.Dataset, encoding: dict, existing: _StoreState | None) -> xr.Dataset:
    stored = existing.variables if existing is not None else frozenset()
    for name, var in ds.variables.items():
        if name in stored or "_FillValue" in encoding.get(name, {}):
            var.attrs.pop("_FillValue", None)
    return ds


def _write(ds: xr.Dataset, store, to_zarr_kwargs: dict) -> Delayed:
    with warnings.catch_warnings():
        if to_zarr_kwargs.get("zarr_format") == 3 and to_zarr_kwargs.get("consolidated"):
            warnings.filterwarnings("ignore", message="Consolidated metadata is currently not part",
                                    category=UserWarning)
        return ds.to_zarr(store, **to_zarr_kwargs)


def _chain_commits(delayeds: list[Delayed], targets: list[_Target], commit: str | None) -> list[Delayed]:
    if commit is None:
        return delayeds
    writes_per_session: dict[int, tuple[Any, list[Delayed]]] = {}
    for delayed, target in zip(delayeds, targets):
        writes_per_session.setdefault(id(target.session), (target.session, []))[1].append(delayed)
    return [dask.delayed(_commit_after, pure=False)(session, commit, writes)
            for session, writes in writes_per_session.values()]


def _commit_after(session, message: str, _writes) -> str:
    return session.commit(message)
