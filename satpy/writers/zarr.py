"""Writer for Zarr stores.

The ``zarr`` writer stores Satpy datasets with :meth:`xarray.Dataset.to_zarr`. Data are written as they
are in the Scene: no enhancement is applied, and dtypes, units, and fill values are kept. Only
``save_dataset`` can convert units, with ``units=`` (requires ``pint-xarray``).

Example usage
-------------

In the default use case all datasets are saved as variables of one zarr store. This means they must share the
same grid (resample the Scene first if they do not). A ``filename`` must be specified when multiple datasets
are stored.

    >>> scn.save_datasets(writer="zarr", filename="/data/zarr/{platform_name}_{start_time:%Y%m%d_%H%M%S}.zarr")

Variables in the resulting store are named by their Satpy name (``.attrs["name"]``).
If only one dataset is present or if ``split_variables`` is used (see Variables and Groups
below) a default filename pattern is used (``{name}_{start_time:%Y%m%d_%H%M%S}.zarr``).
The filename is formatted with the metadata of the saved datasets as combined by
:func:`~satpy.dataset.metadata.combine_metadata`: only attributes with the same value in every dataset
are available, ``start_time`` is the earliest and ``end_time`` the latest of all datasets, and ``name``
is only available when saving one dataset.

Options of :meth:`xarray.Dataset.to_zarr` that the writer does not expose directly can be passed with
``xarray_kwargs``; they override any defaults the Zarr writer would have used otherwise:

    >>> scn.save_datasets(writer="zarr", filename="/data/zarr/abi.zarr",
    ...                   xarray_kwargs={"write_empty_chunks": False, "safe_chunks": False})

Some xarray options are made available as arguments to ``save_datasets``, such as ``mode`` or ``encoding``, but
can only be specified one of these ways (argument to ``save_datasets`` or keyword argument in
``xarray_kwargs``) and will error if provided as both.

Variables and Groups
--------------------

``split_variables=True`` writes every dataset to its own zarr store, named with the default file pattern
mentioned previously:

    >>> scn.save_datasets(writer="zarr", split_variables=True, base_dir="/data/zarr")

To save other combinations of datasets, call ``save_datasets`` once per store and select the datasets
with ``datasets=``. Datasets on different grids can also share one store as separate zarr groups, using
the ``group`` option of :meth:`xarray.Dataset.to_zarr`:

    >>> scn.save_datasets(writer="zarr", datasets=["C01", "C03"], filename="/data/zarr/abi.zarr",
    ...                   xarray_kwargs={"group": "1km"})
    >>> scn.save_datasets(writer="zarr", datasets=["C13"], filename="/data/zarr/abi.zarr",
    ...                   xarray_kwargs={"group": "2km"})

Store locations
---------------

Filenames may be fsspec URLs, for example on S3 (requires ``s3fs``), with ``storage_options`` passed to
fsspec.

    >>> scn.save_datasets(writer="zarr", filename="s3://bucket/satpy/abi_{start_time:%Y%m%d_%H%M%S}.zarr",
    ...                   storage_options={"anon": False})

Instead of a filename, a ``store`` keyword argument can be used and accepts any zarr ``StoreLike``
(for example :class:`zarr.storage.MemoryStore`). ``filename`` and ``store`` cannot be used together.

    >>> from zarr.storage import MemoryStore
    >>> store = MemoryStore()
    >>> scn.save_datasets(writer="zarr", datasets=["C07", "C13"], store=store)

.. note::

   When opening a zarr store with xarray use ``xr.open_zarr(store)``. Include ``consolidated=False``
   to match the default storage method of this zarr writer and avoid unnecessary data accesses.

With ``split_variables=True`` and more than one dataset, ``store`` must be a dictionary that maps every
dataset name to a different store.

    >>> store_c07 = MemoryStore()
    >>> store_c13 = MemoryStore()
    >>> scn.save_datasets(writer="zarr", datasets=["C07", "C13"], split_variables=True,
    ...                   store={"C07": store_c07, "C13": store_c13})

To write several datasets to the same store, use the default
``split_variables=False`` behavior.
An Icechunk ``Session`` can be passed as ``store`` as well (see Icechunk).

Encoding
--------

Each variable is encoded with its ``DataArray.encoding``, overridden key by key by the ``fill_value``
argument and then by the ``encoding`` argument:

    >>> scn["C13"].encoding = {"dtype": "int16", "scale_factor": 0.01, "add_offset": 200.0, "_FillValue": -1}
    >>> scn.save_datasets(writer="zarr", filename="/data/zarr/abi.zarr", encoding={"C13": {"chunks": (512, 512)}})

Keys that :meth:`xarray.Dataset.to_zarr` does not use, such as the netCDF ``zlib`` or ``chunksizes`` a
reader may leave, and the ``compressor`` of zarr format 2 are dropped. The rest of an encoding left by a
reader is applied, so data that a reader returns unchanged are stored with the dtype, scaling, and fill
value of the input file; operations that change the values (e.g. calibration or resampling) drop the
encoding.
To store data unpacked that kept a reader's encoding remove the encoding by calling
``data_arr = data_arr.drop_encoding()`` first.
Codecs in ``compressors``, ``filters``, and ``serializer`` must be zarr format 3 codecs, such as
:mod:`zarr.codecs` classes or the :mod:`numcodecs:numcodecs.zarr3` wrappers.
Irregular dask chunks, which zarr
cannot store, are rechunked to regular ones unless the encoding sets ``chunks``.

Zarr format 3 separates the ``fill_value`` of an array, which zarr returns for chunks that were never
written, from the CF ``_FillValue`` attribute that marks missing data in xarray and Satpy. The writer
sets the ``fill_value`` of new arrays to the ``_FillValue`` encoding or, for integer data,
to the ``_FillValue`` attribute. This allows zarr to skip chunks that contain only the fill value.
For writing float data with no ``_FillValue`` encoding, NaN is the default ``fill_value``.
A ``fill_value`` in the ``encoding`` argument takes precedence; one in ``DataArray.encoding`` is dropped.
When reading a zarr store with xarray, values equal to ``_FillValue`` will be masked and returned
as NaN.

Attributes and coordinates
--------------------------

Before writing, the writer removes Satpy-internal attributes (``area``, ``_last_resampler``, keys starting
with ``_satpy``, ``None`` values), replaces ``ancillary_variables`` by their names, converts the remaining
attributes to JSON with :func:`~satpy.writers.zarr.attrs_to_safe_json`, drops the ``crs`` coordinate, and
adds CF ``standard_name``/``units`` to ``x`` and ``y``.

Geolocation
-----------

Data on an ``AreaDefinition`` get the attributes of the ``proj`` and ``spatial`` conventions (version
0.1) of the `Zarr conventions framework <https://github.com/zarr-conventions>`_ used by `GeoZarr
<https://geozarr.org>`_: the CRS as ``proj:wkt2``, ``proj:projjson``, and, if it has one, ``proj:code``,
and the affine ``spatial:transform``, ``spatial:bbox``, and ``spatial:shape`` of the ``y``/``x`` grid.
When all variables share one CRS, the ``proj:`` attributes are also written to the group they are written
to. The root group is the default group if ``group`` is not set in ``xarray_kwargs``.
``include_geozarr=False`` leaves them out.

``include_grid_mapping=True`` adds a CF grid mapping coordinate named ``spatial_ref``, the layout that
rioxarray currently reads and writes:

    >>> scn.save_datasets(writer="zarr", filename="/data/zarr/abi.zarr", include_grid_mapping=True)

Swath data get 2D ``longitude`` and ``latitude`` coordinates instead, because the conventions do not
cover them. ``include_lonlats=True`` adds these coordinates to gridded data as well and
``include_lonlats=False`` never adds them.

Data on other area types get neither and are written without georeference, with a warning. This
includes a ``StackedAreaDefinition`` whose parts pyresample could not merge into one grid, such as SEVIRI
HRV data whose upper and lower windows are shifted. Pass ``include_lonlats=True`` to add longitude and
latitude coordinates to them, or resample the data to a grid first.

CF conventions
--------------

``convert_to_cf=True`` prepares the data with the CF conversion of the ``cf`` writer instead, including
its grid mapping variable, longitude/latitude coordinates, and attribute encoding, and ignores
``include_lonlats``, ``include_grid_mapping``, and ``include_geozarr``. It requires the netCDF4 or
h5netcdf package. Options of :func:`~satpy.cf.datasets.collect_cf_datasets` are passed with
``cf_kwargs``:

    >>> scn.save_datasets(writer="zarr", filename="/data/zarr/abi.zarr", convert_to_cf=True,
    ...                   cf_kwargs={"include_lonlats": False, "exclude_attrs": ["raw_metadata"]})

``cf_kwargs["epoch"]`` sets the units of times that exist before the conversion and defaults to the units
described under Appending. ``time_bnds`` is encoded like ``time``. Names starting with a digit get the
``numeric_name_prefix`` of ``cf_kwargs``, ``"CHANNEL_"`` by default, both as variable names and as keys of
the ``encoding`` argument, so ``encoding={"4": ...}`` applies to the variable ``CHANNEL_4``.

Zarr format and consolidated metadata
-------------------------------------

The writer only writes zarr format 3. :meth:`xarray.Dataset.to_zarr` also supports zarr format 2, but this
writer does not. It does not detect existing format 2 stores either: with ``mode="a"`` or ``append_dim`` it
writes a separate format 3 hierarchy into the same location, which zarr-python then reads instead of the
format 2 data.

Consolidated metadata is not part of the zarr format 3 specification and is not written by default.
``consolidated=True`` writes it, which mainly speeds up opening stores with many variables on object
storage such as S3.

:func:`xarray.open_zarr` looks for consolidated metadata by default and emits a ``RuntimeWarning`` when a
store has none, so open stores written with the defaults with ``consolidated=False``:

    >>> import xarray as xr
    >>> ds = xr.open_zarr("/data/zarr/abi_c13.zarr", consolidated=False)

Open modes
----------

``mode`` is passed to :meth:`xarray.Dataset.to_zarr` and decides what happens to an existing store, or to
an existing group when ``group`` is set in ``xarray_kwargs``:

- ``"w-"`` creates the store or group and raises a ``FileExistsError`` if it exists already. This is the
  default without ``append_dim``. It can add a new group to an existing store.
- ``"w"`` creates the store or group and replaces an existing one. Without ``group`` the whole store is
  deleted, including its other groups; with ``group`` only that group is replaced.
- ``"a"`` creates the store or group if it does not exist yet and otherwise writes into it: stored
  variables that are written again are overwritten, new variables are added, and the other stored
  variables are kept. Overwritten variables must have their stored sizes and keep their stored encoding.
  With ``append_dim``, the variables without the append dimension are rewritten with every write.
- ``"a-"`` is the default with ``append_dim``. It works like ``"a"``, except that stored variables without
  the append dimension, such as ``x`` and ``y``, keep the values of the first write. Without
  ``append_dim`` it is the same as ``"a"``.
- ``"r+"`` only overwrites variables that are already stored, in a store that must exist; new variables
  raise a ``ValueError``. It is the default when ``region`` is set in ``xarray_kwargs``.

Unless ``mode="w"`` replaces the whole store, ``consolidated`` defaults to the setting of the existing
store.

Appending
---------

``append_dim="time"`` adds a length-1 ``time`` dimension, using the earliest ``start_time`` of the
saved datasets, to data that do not have one and appends along it. If the store does not exist yet it
is created, so the same call works for the first and every following write:

    >>> for filenames in files_per_time_step:
    ...     scn = Scene(filenames, reader="abi_l1b")
    ...     scn.load(["C13"])
    ...     scn.save_datasets(writer="zarr", filename="/data/zarr/abi_c13.zarr", append_dim="time")

Times are stored as microseconds since 1970-01-01 unless the ``time`` encoding sets other units, so later
appends with sub-second times round-trip exactly. New data are rechunked to the store's chunks
(``align_chunks=True``).
Entries of the ``encoding`` argument for variables that are already in the store are not used.

Appended steps are added at the end of the store in the order they are written; the writer does not
compare their times with the stored ones. Appending a ``start_time`` that is already stored neither
replaces nor skips the stored step: the store then holds two steps with that time, and ``sel`` returns
both. Appending an earlier time leaves the ``time`` coordinate unsorted, so slicing it by time raises a
``KeyError``. To avoid both, check the stored times, for example
``xr.open_zarr(store, consolidated=False)["time"]``, before appending.

Attributes of variables that are already in the store keep the values of the first write, so their
``start_time`` and ``end_time`` describe the first time step; the ``time`` coordinate holds the
``start_time`` of every step. The attributes of the root group are replaced by those of the latest write.

Coordinates that differ for every step, such as ``acq_time`` and the lon/lats of swath data, get the
append dimension too. The ``x``, ``y``, and lon/lat coordinates of an area, and the other variables
without the append dimension, keep the values of the first write, as appends default to xarray's
``mode="a-"``. Pass ``mode="a"`` to rewrite them with every write.

The writer does not compare appended data with the stored data. xarray raises when their dimensions or
sizes differ, but data on another area of the same size, or with the same number of other ``bands``, can
be appended without an error. With ``mode="a-"`` they then get the stored ``x``, ``y``, and ``bands``; with
``mode="a"`` they replace those of the earlier steps.

Without ``append_dim``, ``mode`` keeps xarray's default: a new store is created and writing to an
existing one fails. Use ``mode="w"`` to overwrite. With ``append_dim="time"``, ``mode="w"`` replaces the
store with one holding a single time step, which later appends extend.

Delayed writing
---------------

With ``compute=False`` a list of :class:`dask.delayed.Delayed` objects is returned, to be passed to
:func:`~satpy.writers.core.compute.compute_writer_results`. xarray writes the store metadata and
in-memory coordinates immediately, so a store whose delayed result is never computed only contains
metadata and blocks a later write without ``mode="w"``.

This also happens to the stores written before a store that xarray rejects, as with
``split_variables=True``. Errors xarray raises for one store, such as a store to create that exists
already, or appended data whose dimensions or sizes differ from the stored ones, leave the stores before
it with metadata only, or with an appended step without data. Errors in the writer's own arguments, such
as ``encoding`` entries for unknown variables or a missing ``start_time`` with ``append_dim="time"``,
raise before any store is written.

Icechunk
--------

An `Icechunk <https://icechunk.io>`_ ``Session`` can be passed as ``store``. Icechunk stores do not
support consolidated metadata, so none is written to them, even with ``consolidated=True``.
With ``commit``, the session is committed after the data are written, inside the same dask computation:

    >>> import icechunk
    >>> repo = icechunk.Repository.open_or_create(icechunk.local_filesystem_storage("/data/repo"))
    >>> session = repo.writable_session("main")
    >>> scn.save_datasets(writer="zarr", store=session, datasets=["C07", "C13"], commit="Add ABI scan")

Without ``commit`` the changes stay uncommitted; call ``session.commit("message")`` yourself after
computing. A committed session is read-only, so every append needs a new ``repo.writable_session()``.
The writable session cannot be pickled, so Icechunk targets only work with dask's threaded scheduler,
not with ``dask.distributed``.

.. todo::

    - Default filename: the pattern of the writer YAML configuration,
      ``{name}_{start_time:%Y%m%d_%H%M%S}.zarr``, only works when saving one dataset, because ``name`` is
      not available for stores with several datasets. ``scn.save_datasets(writer="zarr")`` therefore
      raises for a Scene with more than one dataset unless ``filename`` or ``split_variables=True`` is
      given. Decide on a default that works for both, such as a pattern without ``name`` or
      ``split_variables=True`` by default.
    - GeoZarr: set ``include_grid_mapping=True`` by default once the GeoZarr core text is normative;
      adopt the next release of the ``proj``/``spatial`` conventions once rioxarray and geozarr-toolkit
      use the same convention names as the published schemas (v0.1 uses ``proj``/``spatial``, both tools
      use the pre-release ``proj:``/``spatial:``) and build the attributes with geozarr-toolkit then;
      ``proj:``/``spatial:`` attributes for swath data when a convention exists; follow the
      crs/geotransform/time/bands conformance classes of the GeoZarr spec.
    - Multiscales: optional pyramid levels (native resampler or dask coarsen) as child groups with the
      ``multiscales`` layout attribute; one Delayed per level.
    - Region writes: ``region={"time": slice(i, i + 1)}`` into a pre-allocated time axis for out-of-order
      or parallel appends (needs a helper that creates the store with the full time axis but no data).
    - Icechunk with ``dask.distributed``: a writable ``Session`` cannot be pickled, so only the threaded
      scheduler works. ``Session.fork()`` returns a picklable ``ForkSession`` for the workers to write to
      and ``Session.merge()`` collects their changes before the commit. ``icechunk.xarray.to_icechunk``
      does both but computes immediately, so it only fits ``compute=True``; ``compute=False`` needs the
      merge inside the returned dask graph.
    - Icechunk with concurrent writers: ``Session.commit()`` raises ``icechunk.ConflictError`` when another
      session committed to the branch after this one was created. ``commit(rebase_with=...)`` or
      ``Session.rebase()`` with ``icechunk.BasicConflictSolver()`` resolves this for region writes of
      different time steps from separate processes, but not for concurrent appends, which both resize the
      arrays.
    - Sharding: a default ``shards`` encoding, so that small chunks for fast partial reads are not stored as
      one object each; ``shards`` can currently only be set with ``encoding``.
    - obstore: URL filenames are opened with fsspec. A :class:`zarr.storage.ObjectStore` can already be
      passed as ``store``; creating one from the URL and credentials would let ``filename`` use obstore.
    - Append checks: xarray raises for appended data whose dimensions or sizes differ from the stored ones
      only when it writes their store, after the stores before it already got an appended step without
      data (see Delayed writing). Consider comparing the sizes of every store with its stored ones before
      writing any. The writer already reads the metadata of the stored arrays, for their names, so this
      costs no requests; it could also compare the stored ``x``, ``y``, and ``bands`` values, which catches
      data on another area of the same size, for one request per coordinate.
    - Time bounds: when ``append_dim="time"`` adds the ``time`` dimension, only the ``start_time`` of each
      step is kept, as its ``time`` coordinate (see Appending). A ``time_bnds`` variable from
      ``start_time``/``end_time``, as :func:`satpy.cf.coords.add_time_bounds_dimension` adds for data that
      already have a ``time`` dimension, would keep every step's period, with or without ``convert_to_cf``.

"""

from __future__ import annotations

import datetime as dt
import importlib.metadata
import json
import logging
import math
import os
import typing
import warnings
from dataclasses import dataclass, replace

import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from packaging.version import Version
from pyresample.geometry import AreaDefinition, SwathDefinition
from trollsift import compose

from satpy.cf.area import add_lonlat_coords
from satpy.cf.coords import add_xy_coords_attrs
from satpy.dataset.dataid import DataID
from satpy.dataset.metadata import combine_metadata
from satpy.utils import chunks_are_irregular, regular_chunks_from_irregular_chunks
from satpy.writers.core.base import Writer
from satpy.writers.core.compute import compute_writer_results

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Collection, Hashable, Iterable
    from os import PathLike
    from typing import Any

    from dask.delayed import Delayed

LOG = logging.getLogger(__name__)

_SATPY_KWARGS = ("overlay", "decorate", "config_files")
_NEW_GROUP_MODES = ("w", "w-")
# save_datasets arguments that are also to_zarr arguments, so they may come from xarray_kwargs, with their defaults
_TO_ZARR_ARGUMENT_DEFAULTS = {
    "store": None,
    "storage_options": None,
    "mode": None,
    "append_dim": None,
    "consolidated": None,
    "encoding": None,
    "compute": True,
}
_MIN_VERSIONS = {"zarr": Version("3.0.0"), "xarray": Version("2025.6.0")}
# Without explicit units, xarray picks "days since <first time>" and later sub-second appends
# cannot be decoded anymore.
_TIME_ENCODING = {
    "units": "microseconds since 1970-01-01 00:00:00",
    "calendar": "proleptic_gregorian",
    "dtype": "int64",
}
# Keys of DataArray.encoding that Dataset.to_zarr understands for zarr format 3; others, such as the netCDF "zlib"
# or "chunksizes" a reader may leave, or the zarr format 2 "compressor", are dropped. "fill_value" is set from
# _FillValue instead.
_ZARR_ENCODING_KEYS = frozenset(
    {
        "dtype",
        "scale_factor",
        "add_offset",
        "_FillValue",
        "missing_value",
        "units",
        "calendar",
        "chunks",
        "shards",
        "compressors",
        "filters",
        "serializer",
        "write_empty_chunks",
        "chunk_key_encoding",
    }
)
_ZARR_CONVENTIONS = {
    "proj": {
        "schema_url": "https://raw.githubusercontent.com/zarr-conventions/proj/refs/tags/v0.1/schema.json",
        "spec_url": "https://github.com/zarr-conventions/proj/blob/v0.1/README.md",
        "uuid": "f17cb550-5864-4468-aeb7-f3180cfb622f",
        "name": "proj",
        "description": "Coordinate reference system information for geospatial data",
    },
    "spatial": {
        "schema_url": "https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v0.1/schema.json",
        "spec_url": "https://github.com/zarr-conventions/spatial/blob/v0.1/README.md",
        "uuid": "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4",
        "name": "spatial",
        "description": "Spatial coordinate information",
    },
}


class ZarrWriter(Writer):
    """Write datasets to Zarr stores with :meth:`xarray.Dataset.to_zarr`.

    See the :mod:`module documentation <satpy.writers.zarr>` for usage examples.

    """

    def __init__(self, name=None, filename=None, base_dir=None, **kwargs):
        """Initialize the writer and load its YAML configuration."""
        super().__init__(
            name=name, filename=filename, base_dir=base_dir, default_config_filename="writers/zarr.yaml", **kwargs
        )

    def create_filename_parser(self, base_dir):
        """Create the filename parser, ignoring ``base_dir`` for URL patterns as it is for absolute paths."""
        if _is_url(self.file_pattern):
            base_dir = None
        return super().create_filename_parser(base_dir)

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

        See :meth:`save_datasets` for the keyword arguments. ``units`` converts the data with pint-xarray
        before writing and drops the data's ``encoding``, whose packing does not fit the converted values.

        """
        if units is not None:
            dataset = _convert_units(dataset, units)
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
        consolidated: bool | None = None,
        encoding: dict[str, dict] | None = None,
        include_lonlats: bool | None = None,
        include_grid_mapping: bool = False,
        include_geozarr: bool = True,
        convert_to_cf: bool = False,
        cf_kwargs: dict | None = None,
        xarray_kwargs: dict | None = None,
        **kwargs,
    ) -> list[Delayed] | list[Any]:
        """Save datasets to one or more zarr stores.

        Args:
            datasets: DataArrays to write.
            compute: Write the data now. If ``False``, return a list of
                :class:`~dask.delayed.Delayed` objects instead.
            filename: Local path, fsspec URL, or trollsift pattern of the store. Formatted with the
                metadata shared by the datasets of the store (see
                :func:`~satpy.dataset.metadata.combine_metadata`); ``name`` is only available for
                stores holding one dataset. Defaults to the ``filename`` pattern given when creating
                the writer.
            store: A zarr ``StoreLike`` or Icechunk ``Session`` to write to instead of ``filename``.
                With ``split_variables``, a dict of different stores keyed by dataset name.
            storage_options: fsspec options for remote URL targets.
            split_variables: Write every dataset to its own store instead of all datasets to one store.
            commit: Commit message for Icechunk sessions passed as ``store``. The commit happens after
                the data are written.
            fill_value: ``_FillValue`` encoding of all data variables, overriding their own ``encoding``,
                and the zarr ``fill_value`` of new arrays.
            append_dim: Dimension to append along. ``"time"`` is added from ``start_time`` when missing.
            mode: Passed to :meth:`xarray.Dataset.to_zarr`. ``None`` creates a new store, or appends with
                ``"a-"`` when ``append_dim`` is given, which leaves the stored variables without the append
                dimension unchanged. ``"w"`` overwrites the store, or only the group given in
                ``xarray_kwargs``.
            consolidated: Write consolidated metadata. Defaults to ``False``, or to the existing store's
                state when writing to an existing store.
            encoding: Per-variable encoding for :meth:`xarray.Dataset.to_zarr`, merged over each
                variable's own ``encoding`` and the writer's defaults. Variables with ``chunks`` in either
                are not rechunked by the writer. Each store gets the entries of its own variables, except
                for variables that are already in the store; names of no written variable raise an error.
            include_lonlats: Add 2D ``longitude`` and ``latitude`` coordinates. ``None`` adds them to swath
                data only, ``True`` to data on any area.
            include_grid_mapping: Add a CF ``spatial_ref`` grid mapping coordinate to data on an
                ``AreaDefinition``.
            include_geozarr: Add the attributes of the GeoZarr ``proj`` and ``spatial`` conventions to data
                on an ``AreaDefinition``.
            convert_to_cf: Prepare the data with Satpy's CF conversion, like the ``cf`` writer, instead of
                the writer's own preparation. ``include_lonlats``, ``include_grid_mapping``, and
                ``include_geozarr`` are not used then. Requires netCDF4 or h5netcdf.
            cf_kwargs: Keyword arguments for :func:`~satpy.cf.datasets.collect_cf_datasets` when
                ``convert_to_cf`` is ``True``. ``pretty`` defaults to ``True``.
            xarray_kwargs: Other :meth:`xarray.Dataset.to_zarr` keyword arguments, which override the
                arguments chosen by the writer. ``store``, ``storage_options``, ``mode``,
                ``append_dim``, ``consolidated``, ``encoding``, and ``compute`` are used as the arguments
                of this method; passing one both ways raises an error.
            kwargs: Satpy keyword arguments that do not apply to this writer (``overlay``,
                ``decorate``) are ignored; anything else is ignored with a warning.

        Returns:
            The computed results if ``compute`` is ``True``, else a list of Delayed objects.

        """
        _check_versions()
        _warn_unknown_kwargs(kwargs)
        xarray_kwargs = dict(xarray_kwargs or {})
        to_zarr_args = _pop_to_zarr_arguments(
            xarray_kwargs,
            store=store,
            storage_options=storage_options,
            mode=mode,
            append_dim=append_dim,
            consolidated=consolidated,
            encoding=encoding,
            compute=compute,
        )
        # The Scene passes 'filename' to __init__, where it replaces the YAML default as file_pattern. A filename
        # passed there that equals the YAML default is not detected.
        if to_zarr_args["store"] is not None and (
            filename is not None or self.file_pattern != self.info.get("filename")
        ):
            raise ValueError("'filename' and 'store' cannot be used together.")
        datasets = list(datasets)
        if not datasets:
            return []
        builder = _make_dataset_builder(
            convert_to_cf, cf_kwargs, include_lonlats, include_grid_mapping, include_geozarr
        )
        saver = _ZarrSaver(
            planner=_TargetPlanner(
                compose_filename=self._compose_filename,
                filename=filename,
                store=to_zarr_args["store"],
                storage_options=to_zarr_args["storage_options"],
                split_variables=split_variables,
                commit=commit,
            ),
            builder=builder,
            store_mode=_StoreMode(
                mode=_resolve_mode(to_zarr_args["mode"], to_zarr_args["append_dim"], xarray_kwargs.get("region")),
                append_dim=to_zarr_args["append_dim"],
                consolidated=to_zarr_args["consolidated"],
                xarray_kwargs=xarray_kwargs,
            ),
            encoding=_EncodingPolicy(builder.rename_encoding(to_zarr_args["encoding"] or {}), fill_value),
        )
        delayeds = saver.delayed_writes(datasets)
        if to_zarr_args["compute"]:
            return compute_writer_results([delayeds])
        return delayeds

    def _compose_filename(self, filename: str | PathLike | None, attrs: dict) -> str:
        # Unlike Writer.get_filename, this creates no parent directories: zarr creates them itself, and they would be
        # wrong for URLs
        self._prepare_metadata_for_filename_formatting(attrs)
        if filename is not None:
            return compose(os.fspath(filename), attrs)
        if self.filename_parser is None:
            raise RuntimeError("No filename pattern or specific filename provided")
        return self.filename_parser.compose(attrs)


def _convert_units(data_arr: xr.DataArray, units: str) -> xr.DataArray:
    import pint_xarray  # noqa: F401

    converted = data_arr.pint.quantify().pint.to(units).pint.dequantify()
    converted.encoding = {}
    return converted


@dataclass(frozen=True)
class _ZarrSaver:
    """Write the datasets of one :meth:`ZarrWriter.save_datasets` call, using one helper per group of arguments."""

    planner: _TargetPlanner
    builder: _DatasetBuilder
    store_mode: _StoreMode
    encoding: _EncodingPolicy

    def delayed_writes(self, datasets: Iterable[xr.DataArray]) -> list[Delayed]:
        """Write the store metadata of ``datasets`` and return the delayed writes of their data."""
        # to_zarr(compute=False) already writes the store metadata, so every store is prepared and checked before
        # any is written
        jobs = [self._prepare(target) for target in self.planner.plan(datasets)]
        self.encoding.check_names([job.dataset for job in jobs])
        return [job.to_delayed() for job in jobs]

    def _prepare(self, target: _Target) -> _WriteJob:
        layout = self.store_mode.resolve(target)
        ds = self.store_mode.expand(self.builder.build(target.variables), target)
        ds, encoding = self.encoding.apply(ds, layout)
        return _WriteJob(target, ds, {**self.store_mode.to_zarr_kwargs(target, layout), "encoding": encoding})


def _check_versions() -> None:
    for package, min_version in _MIN_VERSIONS.items():
        version = _package_version(package)
        if version < min_version:
            raise ImportError(
                f"The 'zarr' writer requires {package}>={min_version}, but {package} {version} is installed."
            )


def _package_version(package: str) -> Version:
    return Version(importlib.metadata.version(package))


def _resolve_mode(mode: str | None, append_dim: Hashable | None, region) -> str:
    """Get the mode that ``to_zarr`` writes with, choosing a default for ``None`` like xarray, except for appends."""
    if mode is not None:
        return mode
    if append_dim is not None:
        # Unlike xarray's default "a", "a-" leaves the stored variables without the append dimension, such as the
        # coordinates of an area, as the first write stored them
        return "a-"
    if region is not None:
        return "r+"
    return "w-"


def _warn_unknown_kwargs(kwargs: dict) -> None:
    unknown = sorted(key for key in kwargs if key not in _SATPY_KWARGS)
    if unknown:
        warnings.warn(f"The 'zarr' writer ignores these unknown keyword arguments: {unknown}", stacklevel=3)


def _pop_to_zarr_arguments(xarray_kwargs: dict, **arguments) -> dict:
    """Take the ``save_datasets`` arguments that may also be passed in ``xarray_kwargs`` out of it."""
    both = [
        key for key, value in arguments.items() if key in xarray_kwargs and value is not _TO_ZARR_ARGUMENT_DEFAULTS[key]
    ]
    if both:
        raise ValueError(f"{both} given both as arguments and in 'xarray_kwargs'; pass each only once.")
    return {key: xarray_kwargs.pop(key, value) for key, value in arguments.items()}


@dataclass(frozen=True)
class _Target:
    """One zarr store and the datasets written to it with one ``to_zarr`` call."""

    variables: dict[str, xr.DataArray]
    attrs: dict
    store: Any
    # The Icechunk session of the store, which 'commit' commits
    session: Any = None
    # Only kept for remote URLs
    storage_options: dict | None = None
    # The message the session is committed with after the data are written
    commit: str | None = None

    @property
    def name(self) -> str:
        """The names of the datasets, for messages."""
        return ", ".join(self.variables)


@dataclass(frozen=True)
class _TargetPlanner:
    """Assign the datasets to stores and find the stores from the 'filename' or 'store' argument."""

    compose_filename: Callable[[str | PathLike | None, dict], str]
    filename: str | PathLike | None
    store: Any
    storage_options: dict | None
    split_variables: bool
    # Commit message for the Icechunk sessions passed as 'store'
    commit: str | None

    def plan(self, datasets: Iterable[xr.DataArray]) -> list[_Target]:
        """Create the targets that ``datasets`` are written to."""
        variable_sets = _group_variables(datasets, self.split_variables)
        metadata = [combine_metadata(*variables.values()) for variables in variable_sets]
        if self.store is None:
            stores: list = self._filenames(variable_sets, metadata)
        else:
            stores = _assign_stores(variable_sets, self.store, self.split_variables)
        targets = [
            _make_target(variables, attrs, store, self.storage_options, self.commit)
            for variables, attrs, store in zip(variable_sets, metadata, stores)
        ]
        _require_sessions_for_commit(targets)
        return targets

    def _filenames(self, variable_sets: list[dict[str, xr.DataArray]], metadata: list[dict]) -> list[str]:
        names_by_filename: dict[str, str] = {}
        for variables, attrs in zip(variable_sets, metadata):
            names = ", ".join(variables)
            target_filename = self._filename(variables, attrs)
            if target_filename in names_by_filename:
                raise ValueError(
                    f"Datasets {names_by_filename[target_filename]!r} and {names!r} would both be "
                    f"written to {target_filename!r}. Include '{{name}}' in the filename."
                )
            names_by_filename[target_filename] = names
        return list(names_by_filename)

    def _filename(self, variables: dict[str, xr.DataArray], attrs: dict) -> str:
        attrs = dict(attrs)
        if len(variables) == 1:
            attrs["name"] = next(iter(variables))
        try:
            return self.compose_filename(self.filename, attrs)
        except KeyError as err:
            if len(variables) == 1:
                raise ValueError(f"Cannot create the zarr filename, the dataset has no {err} attribute.") from err
            raise ValueError(
                f"Cannot create the zarr filename, the datasets have no common {err} attribute. "
                "Only attributes equal in all datasets of a store, never 'name', can be used, so "
                "specify a 'filename' for multi-variable zarr stores or use split_variables=True."
            ) from err


def _variable_name(data_arr: xr.DataArray) -> str:
    name = data_arr.attrs.get("name")
    if name is None and isinstance(data_arr.name, str):
        name = data_arr.name
    if name is None:
        LOG.warning("DataArray has no 'name' attribute, writing it as 'data'.")
        name = "data"
    return str(name)


def _group_variables(datasets: Iterable[xr.DataArray], split_variables: bool) -> list[dict[str, xr.DataArray]]:
    named = [(_variable_name(data_arr), data_arr) for data_arr in datasets]
    if split_variables:
        return [{name: data_arr} for name, data_arr in named]
    variables: dict[str, xr.DataArray] = {}
    for name, data_arr in named:
        if name in variables:
            raise ValueError(f"More than one dataset named {name!r} in one store.")
        variables[name] = data_arr
    return [variables]


def _assign_stores(variable_sets: list[dict[str, xr.DataArray]], store, split_variables: bool) -> list:
    names = [name for variables in variable_sets for name in variables]
    if not isinstance(store, dict):
        if len(variable_sets) == 1:
            return [store]
        raise ValueError(
            f"With split_variables=True, a single 'store' cannot hold the {len(variable_sets)} datasets "
            f"{names}. Pass a dict of stores keyed by dataset name."
        )
    if not split_variables:
        raise ValueError(
            "A dict of stores requires split_variables=True; otherwise all datasets are written to one store."
        )
    if len(set(names)) != len(names):
        raise ValueError(f"Dataset names are not unique, cannot assign stores by name: {names}")
    for name in names:
        if name not in store:
            raise ValueError(f"No store given for dataset {name!r}.")
    stores = [store[name] for name in names]
    _require_distinct_stores(names, stores)
    return stores


def _require_distinct_stores(names: list[str], stores: list) -> None:
    # The same path or URL is the same store, like the same store object
    keys = [os.fspath(store) if isinstance(store, (str, os.PathLike)) else id(store) for store in stores]
    shared = [name for name, key in zip(names, keys) if keys.count(key) > 1]
    if shared:
        raise ValueError(
            f"Datasets {shared} are assigned the same store. Write datasets that share a store "
            "without split_variables=True."
        )


def _make_target(
    variables: dict[str, xr.DataArray], attrs: dict, store, storage_options: dict | None, commit: str | None
) -> _Target:
    session = None
    if _is_icechunk_session(store):
        session, store = store, store.store
    if storage_options is not None and not _is_remote(store):
        LOG.debug("Ignoring 'storage_options' for target %r, which is not a remote URL.", ", ".join(variables))
        storage_options = None
    return _Target(variables, attrs, store, session, storage_options, commit)


def _require_sessions_for_commit(targets: list[_Target]) -> None:
    for target in targets:
        if target.commit is not None and target.session is None:
            raise ValueError(
                f"'commit' requires an Icechunk Session as store, but {target.name!r} is not written to one."
            )


def _is_url(target) -> bool:
    return isinstance(target, (str, os.PathLike)) and "://" in os.fspath(target)


def _is_remote(target) -> bool:
    return _is_url(target) and not os.fspath(target).startswith(("file://", "memory://"))


def _is_icechunk(obj) -> bool:
    return type(obj).__module__.split(".")[0] == "icechunk"


def _is_icechunk_session(obj) -> bool:
    return _is_icechunk(obj) and hasattr(obj, "commit") and hasattr(obj, "store")


@dataclass(frozen=True)
class _StoreLayout:
    """Whether one ``to_zarr`` call consolidates the metadata, and the existing arrays that it appends to."""

    consolidated: bool = False
    # Only set when the group written to exists already, as xarray fails to append to a new one
    append_dim: Hashable | None = None
    # Names of the arrays, including coordinates, already in the group written to. They keep their stored encoding
    # because xarray refuses a new one.
    stored: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _StoreMode:
    """Decide whether ``to_zarr`` creates, overwrites, or appends to each store, and whether it consolidates."""

    # The to_zarr mode, with xarray's default already chosen for None
    mode: str
    append_dim: Hashable | None
    consolidated: bool | None
    # The other to_zarr arguments, which override the choices of the writer
    xarray_kwargs: dict

    @property
    def group(self) -> str | None:
        """The zarr group written to, if any."""
        return self.xarray_kwargs.get("group")

    @property
    def keeps_group(self) -> bool:
        """Whether the group written to, or the whole store without 'group', keeps what it has stored already."""
        return self.mode not in _NEW_GROUP_MODES

    def resolve(self, target: _Target) -> _StoreLayout:
        """Choose the consolidation of the write to ``target`` and find what it appends to."""
        # xarray fails to append to a store or group that does not exist yet
        layout = self._read_layout(target) or _StoreLayout()
        if _is_icechunk(target.store):
            # Icechunk stores do not support consolidated metadata, which xarray before 2026.7 still tries to write
            return replace(layout, consolidated=False)
        if self.consolidated is not None:
            return replace(layout, consolidated=self.consolidated)
        return layout

    def _read_layout(self, target: _Target) -> _StoreLayout | None:
        """Read the layout of the existing store of ``target``, or return ``None`` if there is none to match."""
        if self.group is None and not self.keeps_group:
            # The whole store is created or overwritten. A new group, though, is still written into the existing
            # store, whose consolidated metadata it has to keep up to date.
            return None
        import zarr

        not_found = (FileNotFoundError, zarr.errors.NodeNotFoundError)
        # Read the root even when writing a group: with consolidated=True, xarray writes the consolidated
        # metadata of the whole store only to the root.
        try:
            root = zarr.open_group(target.store, mode="r", zarr_format=3, storage_options=target.storage_options)
        except not_found:
            return None
        layout = _StoreLayout(consolidated=root.metadata.consolidated_metadata is not None)
        if not self.keeps_group:
            # The group is created or overwritten, so nothing in it is appended to
            return layout
        if self.group is None and not layout.consolidated:
            # Without consolidated metadata, the root already lists and reads its arrays from the store
            zarr_group = root
        else:
            # xarray reads the stored arrays without consolidated metadata when appending, so the writer does too
            try:
                zarr_group = zarr.open_group(
                    target.store,
                    mode="r",
                    path=self.group,
                    zarr_format=3,
                    use_consolidated=False,
                    storage_options=target.storage_options,
                )
            except not_found:
                return layout
        return replace(layout, append_dim=self.append_dim, stored=frozenset(zarr_group.array_keys()))

    def expand(self, ds: xr.Dataset, target: _Target) -> xr.Dataset:
        """Add the append dimension to ``ds`` and to its coordinates that differ for every step."""
        if self.append_dim is None:
            return ds
        ds = _maybe_expand_append_dim(ds, self.append_dim, target.attrs)
        return _expand_coords(ds, self.append_dim, target.variables.values())

    def to_zarr_kwargs(self, target: _Target, layout: _StoreLayout) -> dict:
        """Get the ``to_zarr`` keyword arguments of the write to ``target``, except ``encoding``."""
        kwargs: dict[str, Any] = {
            "mode": self.mode,
            "align_chunks": True,
            "compute": False,
            "zarr_format": 3,
            "consolidated": layout.consolidated,
        }
        if target.storage_options is not None:
            kwargs["storage_options"] = target.storage_options
        if layout.append_dim is not None:
            kwargs["append_dim"] = layout.append_dim
        kwargs.update(self.xarray_kwargs)
        return kwargs


class _DatasetBuilder(typing.Protocol):
    """Build the dataset written to one store from its DataArrays."""

    def build(self, variables: dict[str, xr.DataArray]) -> xr.Dataset:
        """Build the dataset of the DataArrays ``variables``, which are keyed by variable name."""

    def rename_encoding(self, encoding: dict[str, dict]) -> dict[str, dict]:
        """Key the entries of the 'encoding' argument by the names the variables get in the built datasets."""


def _make_dataset_builder(
    convert_to_cf: bool,
    cf_kwargs: dict | None,
    include_lonlats: bool | None,
    include_grid_mapping: bool,
    include_geozarr: bool,
) -> _DatasetBuilder:
    if convert_to_cf:
        LOG.debug("convert_to_cf=True: include_lonlats, include_grid_mapping and include_geozarr are not used.")
        return _CFDatasetBuilder(cf_kwargs)
    return _SatpyDatasetBuilder(include_lonlats, include_grid_mapping, include_geozarr)


@dataclass(frozen=True)
class _SatpyDatasetBuilder:
    """Build the dataset of one store with the writer's own preparation of the DataArrays."""

    include_lonlats: bool | None
    include_grid_mapping: bool
    include_geozarr: bool

    def build(self, variables: dict[str, xr.DataArray]) -> xr.Dataset:
        """Prepare the DataArrays and merge them into one dataset."""
        lonlats: list[tuple[Any, dict]] = []
        prepared = {name: self._prepare(data_arr, lonlats) for name, data_arr in variables.items()}
        try:
            xr.align(*prepared.values(), join="exact")
            ds = xr.Dataset(prepared)
        except ValueError as err:
            raise ValueError(
                f"Datasets {list(prepared)} cannot be written to one store because their coordinates "
                "differ. Resample them to a common area, use split_variables=True, or save them with "
                "separate save_datasets calls."
            ) from err
        ds.attrs["history"] = f"Created by pytroll/satpy on {dt.datetime.now(dt.timezone.utc)}"
        if self.include_geozarr:
            ds.attrs.update(_root_geozarr_attrs(list(prepared.values())))
        return _drop_unused_encodings(ds)

    def rename_encoding(self, encoding: dict[str, dict]) -> dict[str, dict]:
        """Return ``encoding`` unchanged, as the variables keep their names."""
        return encoding

    def _prepare(self, data_arr: xr.DataArray, lonlats: list[tuple[Any, dict]]) -> xr.DataArray:
        area = data_arr.attrs.get("area")
        data_arr = add_xy_coords_attrs(data_arr.copy(deep=False))
        if _wants_lonlats(data_arr, area, self.include_lonlats):
            data_arr = _add_shared_lonlat_coords(data_arr, area, lonlats)
        elif self.include_lonlats is None:
            _warn_if_not_georeferenced(data_arr, area)
        gridded_area = area if isinstance(area, AreaDefinition) and {"y", "x"}.issubset(data_arr.dims) else None
        if gridded_area is None and (self.include_grid_mapping or self.include_geozarr):
            LOG.debug(
                "Not adding a 'spatial_ref' coordinate or GeoZarr attributes to %r, it is not on an "
                "AreaDefinition with 'y' and 'x' dimensions.",
                data_arr.attrs.get("name"),
            )
        if self.include_grid_mapping and gridded_area is not None:
            data_arr = _add_spatial_ref(data_arr, gridded_area)
        attrs = dict(data_arr.attrs)
        if self.include_geozarr and gridded_area is not None:
            attrs.update(_geozarr_attrs(gridded_area))
        data_arr.attrs = _sanitize_attrs(attrs)
        for coord in data_arr.coords.values():
            coord.attrs = _sanitize_attrs(coord.attrs)
        return data_arr


@dataclass(frozen=True)
class _CFDatasetBuilder:
    """Build the dataset of one store with Satpy's CF conversion, like the ``cf`` writer."""

    cf_kwargs: dict | None

    def build(self, variables: dict[str, xr.DataArray]) -> xr.Dataset:
        """Convert the DataArrays to one CF dataset."""
        from satpy.cf.datasets import collect_cf_datasets

        cf_kwargs = {"pretty": True, "epoch": _TIME_ENCODING["units"], **(self.cf_kwargs or {})}
        datasets, _ = collect_cf_datasets(list_dataarrays=list(variables.values()), groups=None, **cf_kwargs)
        ds = datasets[None]
        ds.attrs = attrs_to_safe_json(ds.attrs)
        for var in ds.variables.values():
            var.attrs = attrs_to_safe_json(var.attrs)
        return _drop_unused_encodings(ds)

    def rename_encoding(self, encoding: dict[str, dict]) -> dict[str, dict]:
        """Prefix the names starting with a digit, as the CF conversion prefixes those of the variables."""
        prefix = (self.cf_kwargs or {}).get("numeric_name_prefix", "CHANNEL_")
        return {
            prefix + name if prefix and name[:1].isdigit() else name: var_encoding
            for name, var_encoding in encoding.items()
        }


def _drop_unused_encodings(ds: xr.Dataset) -> xr.Dataset:
    for name, var in ds.variables.items():
        dropped = sorted(set(var.encoding) - _ZARR_ENCODING_KEYS)
        if dropped:
            LOG.debug("Dropping encoding keys %s of %r that the zarr writer does not use.", dropped, name)
        var.encoding = {key: val for key, val in var.encoding.items() if key in _ZARR_ENCODING_KEYS}
    ds.encoding = {}
    return ds


def _warn_if_not_georeferenced(data_arr: xr.DataArray, area) -> None:
    # By default only AreaDefinitions get a grid georeference and only swaths get longitude/latitude
    if area is None or isinstance(area, (AreaDefinition, SwathDefinition)):
        return
    if {"longitude", "latitude"}.issubset(data_arr.coords):
        return
    LOG.warning(
        "%r is on a %s, which the 'zarr' writer cannot describe as one grid, so it is written without "
        "georeference. Pass include_lonlats=True to add longitude and latitude coordinates, or resample "
        "the data first.",
        data_arr.attrs.get("name"),
        type(area).__name__,
    )


def _wants_lonlats(data_arr: xr.DataArray, area, include_lonlats: bool | None) -> bool:
    if area is None or include_lonlats is False:
        return False
    if include_lonlats is None and not isinstance(area, SwathDefinition):
        return False
    if {"longitude", "latitude"}.issubset(data_arr.coords):
        return False
    if not {"y", "x"}.issubset(data_arr.dims):
        LOG.debug(
            "Not adding longitude/latitude to %r, it does not have both 'y' and 'x' dimensions.",
            data_arr.attrs.get("name"),
        )
        return False
    return True


def _add_shared_lonlat_coords(data_arr: xr.DataArray, area, lonlats: list[tuple[Any, dict]]) -> xr.DataArray:
    # Lon/lats made with the chunks of each dataset would differ in their dask names, so merging the datasets
    # would compute them to compare them
    for known_area, coords in lonlats:
        if type(known_area) is type(area) and known_area == area:
            return data_arr.assign_coords(coords)
    data_arr = add_lonlat_coords(data_arr)
    lonlats.append((area, {name: data_arr.coords[name].variable for name in ("longitude", "latitude")}))
    return data_arr


def _add_spatial_ref(data_arr: xr.DataArray, area: AreaDefinition) -> xr.DataArray:
    data_arr = data_arr.assign_coords(spatial_ref=xr.Variable((), 0, attrs=area.crs.to_cf()))
    data_arr.attrs["grid_mapping"] = "spatial_ref"
    return data_arr


def _geozarr_attrs(area: AreaDefinition) -> dict:
    # The extent follows the image's corners, which are not the minimum and maximum for flipped areas
    x_left, y_bottom, x_right, y_top = area.area_extent
    return {
        "zarr_conventions": [_ZARR_CONVENTIONS["proj"], _ZARR_CONVENTIONS["spatial"]],
        **_proj_attrs(area.crs),
        "spatial:dimensions": ["y", "x"],
        "spatial:transform": [area.pixel_size_x, 0.0, x_left, 0.0, -area.pixel_size_y, y_top],
        "spatial:bbox": [min(x_left, x_right), min(y_bottom, y_top), max(x_left, x_right), max(y_bottom, y_top)],
        "spatial:shape": list(area.shape),
        "spatial:registration": "pixel",
    }


def _proj_attrs(crs) -> dict:
    attrs = {"proj:wkt2": crs.to_wkt(), "proj:projjson": crs.to_json_dict()}
    authority = crs.to_authority()
    if authority is not None:
        attrs["proj:code"] = ":".join(authority)
    return attrs


def _root_geozarr_attrs(data_vars: list[xr.DataArray]) -> dict:
    proj_attrs = [{key: val for key, val in var.attrs.items() if key.startswith("proj:")} for var in data_vars]
    if not proj_attrs[0] or any(attrs != proj_attrs[0] for attrs in proj_attrs[1:]):
        return {}
    return {"zarr_conventions": [dict(_ZARR_CONVENTIONS["proj"])], **proj_attrs[0]}


def _sanitize_attrs(attrs: dict) -> dict:
    attrs = {
        key: val
        for key, val in attrs.items()
        if val is not None and key not in ("area", "_last_resampler") and not key.startswith("_satpy")
    }
    ancillary_variables = attrs.pop("ancillary_variables", None)
    if ancillary_variables:
        attrs["ancillary_variables"] = [
            anc.attrs.get("name", anc.name) if isinstance(anc, xr.DataArray) else anc for anc in ancillary_variables
        ]
    return attrs_to_safe_json(attrs)


def attrs_to_safe_json(attrs: dict) -> dict:
    """Convert attributes to values that zarr can store as JSON.

    The writer converts the attributes of every variable and coordinate with this function, also after the CF
    conversion of ``convert_to_cf=True``:

    - Objects with a ``to_cf`` method are replaced by its result, so wavelength ranges become strings as in
      the ``cf`` writer.
    - A ``DataID`` becomes its string.
    - Dictionary keys become strings.
    - Sets become sorted lists, and tuples become lists.
    - Numpy arrays and scalars become Python lists and numbers, and ``datetime64`` values become ISO strings.
    - NaN and infinite floats become the strings ``"NaN"``, ``"Infinity"``, and ``"-Infinity"`` that zarr
      uses for such fill values.
    - Datetimes and dates become ISO strings, and bytes are decoded as UTF-8.
    - Anything else that JSON cannot represent becomes its ``str()``.

    ``_FillValue`` is kept as it is, because xarray encodes it itself.

    Args:
        attrs: Attributes to convert.

    Returns:
        The converted attributes.

    """
    safe = json.loads(json.dumps(_to_json_types(attrs), default=_json_default, allow_nan=False))
    # xarray encodes a _FillValue attribute itself, as valid JSON also when it is NaN
    if "_FillValue" in attrs:
        safe["_FillValue"] = attrs["_FillValue"]
    return safe


def _to_json_types(obj):
    """Convert what json would encode wrongly or not at all, before ``_json_default`` sees the rest."""
    if hasattr(obj, "to_cf"):
        return _to_json_types(obj.to_cf())
    if isinstance(obj, DataID):
        return str(obj)
    if isinstance(obj, dict):
        return {_json_key(key): _to_json_types(val) for key, val in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_types(item) for item in obj]
    if isinstance(obj, (set, frozenset)):
        return [_to_json_types(item) for item in sorted(obj, key=str)]
    if isinstance(obj, (np.ndarray, np.generic)):
        return _to_json_types(_numpy_to_python(obj))
    if isinstance(obj, float):
        return _json_float(obj)
    return obj


def _json_key(key):
    # json only accepts these keys, which it writes as strings
    if key is None or isinstance(key, (str, int, float, bool)):
        return key
    return str(key)


def _numpy_to_python(obj: np.ndarray | np.generic):
    # tolist() turns datetime64 and timedelta64 values in nanoseconds into integers
    if obj.dtype.kind == "M":
        return obj.astype(str).tolist()
    if obj.dtype.kind == "m":
        return obj.astype("timedelta64[us]").tolist()
    return obj.tolist()


def _json_float(value: float) -> float | str:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def _json_default(obj):
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    return str(obj)


def _maybe_expand_append_dim(ds: xr.Dataset, append_dim: Hashable, attrs: dict) -> xr.Dataset:
    if append_dim in ds.dims:
        return ds
    if append_dim in ds.coords:
        return _expand_dims(ds, append_dim)
    if append_dim != "time":
        raise ValueError(
            f"append_dim={append_dim!r} is not a dimension of the data; only 'time' can be added by the writer."
        )
    start_time = attrs.get("start_time")
    if start_time is None:
        raise ValueError("append_dim='time' needs a 'start_time' attribute to create the time coordinate.")
    return _expand_dims(ds, {"time": [_time_value(start_time)]})


def _expand_dims(ds: xr.Dataset, dim: Hashable | dict) -> xr.Dataset:
    # 0-d data variables, such as the CF grid mapping variable, describe every step, so they stay 0-d
    scalars = {name: ds.variables[name] for name, var in ds.data_vars.items() if var.ndim == 0}
    return ds.drop_vars(list(scalars)).expand_dims(dim).assign(scalars)


def _expand_coords(ds: xr.Dataset, append_dim: Hashable, data_arrs: Iterable[xr.DataArray]) -> xr.Dataset:
    # Dataset.expand_dims only expands the data variables, but coordinates such as acq_time and the lon/lats of a
    # swath differ for every step too, while the lon/lats of an area, like its x and y, are the same for every step
    fixed = set(ds.indexes)
    if not _on_swath(data_arrs):
        fixed |= {"longitude", "latitude"}
    sizes = {append_dim: ds.sizes[append_dim]}
    return ds.assign_coords(
        {
            name: coord.variable.set_dims({**sizes, **coord.sizes})
            for name, coord in ds.coords.items()
            if coord.ndim > 0 and name not in fixed and append_dim not in coord.dims
        }
    )


def _on_swath(data_arrs: Iterable[xr.DataArray]) -> bool:
    return any(isinstance(data_arr.attrs.get("area"), SwathDefinition) for data_arr in data_arrs)


def _time_value(start_time) -> np.datetime64:
    timestamp = pd.Timestamp(start_time)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.to_datetime64().astype("datetime64[us]")


@dataclass(frozen=True)
class _EncodingPolicy:
    """Choose the chunks, encoding, and fill values of the variables of every store."""

    # The 'encoding' argument, with the entries of every store
    user_encoding: dict[str, dict]
    fill_value: float | int | None

    def apply(self, ds: xr.Dataset, layout: _StoreLayout) -> tuple[xr.Dataset, dict[Hashable, dict]]:
        """Rechunk ``ds`` and adjust its fill value attributes, and return it with its ``to_zarr`` encoding."""
        ds = _regularize_chunks(ds, self.user_encoding)
        encoding = self._encoding(ds, layout.stored)
        self._apply_fill_values(ds, encoding, layout)
        return ds, encoding

    def check_names(self, datasets: list[xr.Dataset]) -> None:
        """Raise if the 'encoding' argument has entries for variables that are in none of ``datasets``."""
        names = set().union(*(ds.variables for ds in datasets))
        unknown = sorted(set(self.user_encoding) - names)
        if unknown:
            raise ValueError(
                f"The 'encoding' argument has entries for {unknown}, which are not variables of the written data."
            )

    def _encoding(self, ds: xr.Dataset, stored: Collection[str]) -> dict[Hashable, dict]:
        ignored = sorted(name for name in self.user_encoding if name in stored)
        if ignored:
            LOG.debug("Not using the 'encoding' entries of %s, which are already in the store.", ignored)
        user_encoding: dict[Hashable, dict] = {
            name: var_encoding
            for name, var_encoding in self.user_encoding.items()
            if name in ds.variables and name not in stored
        }
        defaults = {
            name: var_encoding for name, var_encoding in self._defaults(ds, user_encoding).items() if name not in stored
        }
        # An entry passed to to_zarr replaces the variable's own encoding, so it is merged in here, into a new dict
        # that _set_zarr_fill_value can change without changing the 'encoding' argument
        names = [*defaults, *(name for name in user_encoding if name not in defaults)]
        return {name: {**ds[name].encoding, **defaults.get(name, {}), **user_encoding.get(name, {})} for name in names}

    def _defaults(self, ds: xr.Dataset, user_encoding: dict[Hashable, dict]) -> dict[Hashable, dict]:
        defaults: dict[Hashable, dict] = {}
        if _is_datetime_variable(ds, "time"):
            time_encoding = {**ds["time"].encoding, **user_encoding.get("time", {})}
            if "units" not in time_encoding:
                time_encoding = {**_TIME_ENCODING, **time_encoding}
            defaults["time"] = time_encoding
            if _is_datetime_variable(ds, "time_bnds"):
                defaults["time_bnds"] = {key: val for key, val in time_encoding.items() if key in _TIME_ENCODING}
        if self.fill_value is not None:
            for name, var in ds.data_vars.items():
                if var.ndim > 0 and not _is_datetime_variable(ds, name):
                    fill = np.asarray(self.fill_value, dtype=var.dtype).item()
                    defaults[name] = {"_FillValue": fill}
                    if "missing_value" in var.encoding:
                        defaults[name]["missing_value"] = fill
        return defaults

    def _apply_fill_values(self, ds: xr.Dataset, encoding: dict[Hashable, dict], layout: _StoreLayout) -> None:
        for name, var in ds.variables.items():
            # Variables without an entry in the to_zarr encoding are written with their own encoding
            var_encoding = encoding.get(name, var.encoding)
            # xarray refuses a _FillValue attribute next to a _FillValue encoding, and for stored variables
            if name in layout.stored or "_FillValue" in var_encoding:
                var.attrs.pop("_FillValue", None)
            if name not in layout.stored:
                _set_zarr_fill_value(var, var_encoding)


def _set_zarr_fill_value(var: xr.Variable, var_encoding: dict) -> None:
    if "fill_value" in var_encoding or var.dtype.kind in "mM":
        return
    dtype = np.dtype(var_encoding.get("dtype", var.dtype))
    fill = var_encoding.get("_FillValue")
    if "_FillValue" not in var_encoding and dtype.kind in "iu":
        fill = var.attrs.get("_FillValue")
    if fill is not None:
        var_encoding["fill_value"] = np.asarray(fill, dtype=dtype).item()


def _regularize_chunks(ds: xr.Dataset, encoding: dict | None) -> xr.Dataset:
    encoding = encoding or {}
    data_vars, coords = {}, {}
    for name, var in ds.variables.items():
        if not isinstance(var.data, da.Array) or "chunks" in {**var.encoding, **encoding.get(name, {})}:
            continue
        if not chunks_are_irregular(var.data.chunks):
            continue
        # Variable.chunk keeps the encoding
        rechunked = var.chunk(dict(zip(var.dims, regular_chunks_from_irregular_chunks(var.data.chunks))))
        if name in ds.coords:
            coords[name] = rechunked
        else:
            data_vars[name] = rechunked
    return ds.assign(data_vars).assign_coords(coords)


def _is_datetime_variable(ds: xr.Dataset, name: str) -> bool:
    return name in ds.variables and np.issubdtype(ds[name].dtype, np.datetime64)


@dataclass(frozen=True)
class _WriteJob:
    """A prepared ``to_zarr`` call for one store."""

    target: _Target
    dataset: xr.Dataset
    to_zarr_kwargs: dict

    def to_delayed(self) -> Delayed:
        """Write the store metadata and return the delayed write of the data, followed by the commit if any."""
        with warnings.catch_warnings():
            if self.to_zarr_kwargs.get("consolidated"):
                warnings.filterwarnings(
                    "ignore", message="Consolidated metadata is currently not part", category=UserWarning
                )
            delayed = self.dataset.to_zarr(self.target.store, **self.to_zarr_kwargs)
        if self.target.commit is None:
            return delayed
        return dask.delayed(_commit_after, pure=False)(self.target.session, self.target.commit, delayed)


def _commit_after(session, message: str, _write) -> str:
    return session.commit(message)
