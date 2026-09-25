"""Helpers for reading netcdf-based files."""

import inspect
import logging
import os
import warnings
from collections.abc import Mapping
from contextlib import suppress

import dask.array as da
import numpy as np
import xarray as xr

from satpy.readers.core.file_handlers import BaseFileHandler
from satpy.readers.core.remote import open_file_or_filename
from satpy.readers.core.utils import np2str
from satpy.utils import get_legacy_chunk_size

LOG = logging.getLogger(__name__)
CHUNK_SIZE = get_legacy_chunk_size()

OPEN_STRATEGIES = ("shared_store", "file_handle")
DEFAULT_OPEN_STRATEGY = "shared_store"


class NetCDF4FileHandler(BaseFileHandler):
    """Class for inspecting a NetCDF4 file and retrieving its metadata/header data.

    File information can be accessed using bracket notation. Variables are
    accessed by using:

        wrapper["var_name"]

    Or:

        wrapper["group/subgroup/var_name"]

    Attributes can be accessed by appending "/attr/attr_name" to the
    item string:

        wrapper["group/subgroup/var_name/attr/units"]

    Or for global attributes:

        wrapper["/attr/platform_short_name"]

    Or for all of global attributes:

        wrapper["/attrs"]

    The size of a dimension of the file or of one of its groups is available
    as ``wrapper["/dimension/dim_name"]`` or
    ``wrapper["group/dimension/dim_name"]``. All of this information is read
    from the file when it is requested, see :class:`NetCDF4FileContent` for
    the ``file_content`` mapping that holds it.

    Note that loading datasets requires opening the original file with
    ``xarray`` (unless those datasets are cached, see below). That dataset is
    then held open for the lifetime of this file handler; call ``close`` to
    release it sooner. To get just the shape of the dataset append "/shape" to
    the item string:

        wrapper["group/subgroup/var_name/shape"]

    If your file has many small data variables that are frequently accessed,
    you may choose to cache some of them. You can do this by passing a number,
    any variable smaller than this number in bytes will be read into RAM the
    first time it is accessed and kept there. Warning, this part of the API is
    provisional and subject to change.

    ``open_strategy`` selects how the file is opened and how variables are
    turned into ``xarray.DataArray`` objects. Every strategy holds what it
    opened for the lifetime of this file handler; call ``close`` to release it
    sooner.

    - ``"shared_store"`` (default): open the file once as an xarray backend
      store and derive a child store per accessed group. All groups share one
      file handle and one slot in xarray's global file cache
      (``xarray.set_options(file_cache_maxsize=...)``, 128 by default), and
      only the groups that are accessed are decoded by xarray. The metadata
      in ``file_content`` is read from the same (undecoded) file handle.
    - ``"file_handle"``: keep the netCDF4/h5netcdf file handle open and wrap
      each variable in a dask array "manually". xarray does not parse or
      decode anything in this case: the attributes are the raw ones from the
      file (including ``scale_factor``, ``add_offset`` and ``_FillValue``),
      the coordinates are missing and ``xarray_kwargs`` has no effect on
      variables. Masking and scaling is left to the netCDF4 library (see
      ``auto_maskandscale``), which is why this strategy can't be combined
      with ``auto_maskandscale=True`` and the h5netcdf engine. This avoids
      the overhead of xarray, which may be useful if you have a dataset
      distributed over many files, such as for FCI. Note that the arrays
      created this way can't be pickled (e.g. for dask distributed workers)
      and can't be read anymore once the file handler is closed.

    By default the file is opened when the file handler is created, so that
    an unreadable file fails right away. With ``defer_open=True`` the file
    is only opened when something is first read from it, or when the
    ``accessor`` is needed to choose between several engines, so opening
    errors are raised then instead. This saves opening files that are never
    read, but is only useful for readers whose ``start_time``, ``end_time``
    and ``available_datasets`` don't read from the file, as those are needed
    when the ``Scene`` is created.

    Args:
        filename (str): File to read.
        filename_info (dict): Dictionary with filename information.
        filetype_info (dict): Dictionary with filetype information.
        auto_maskandscale (bool): Apply mask and scale factors.
        xarray_kwargs (dict): Additional arguments to `xarray.open_dataset`. Options of the xarray backend
            store (e.g. ``lock`` or ``phony_dims``, or those in ``backend_kwargs``) are used when opening the file.
        cache_var_size (int): Keep variables smaller than this size in bytes in memory once they are read.
        cache_handle (bool): Deprecated, use ``open_strategy="file_handle"`` instead of ``cache_handle=True``.
        engine (str or list of str): The engine to use for reading, either "netcdf4" or "h5netcdf". As a list, will try
            each engine until one works.
        open_strategy (str): How to open the file and read its variables. One of "shared_store" (default)
            or "file_handle". See above.
        defer_open (bool): Only open the file when something is first read from it. See above.

    """

    # for ``close`` when ``__init__`` fails before creating the opener
    _opener = None

    def __init__(self, filename, filename_info, filetype_info,
                 auto_maskandscale=False, xarray_kwargs=None,
                 cache_var_size=0, cache_handle=None, engine="netcdf4",
                 open_strategy=None, defer_open=False):
        """Initialize object."""
        super().__init__(filename, filename_info, filetype_info)
        self.engine = engine
        self._cache_var_size = cache_var_size
        # Variables read into memory, see ``cache_var_size`` and ``get_and_cache_npxr``.
        self.cached_variables = {}
        store_open_kwargs, self._open_dataset_kwargs = _split_xarray_kwargs(xarray_kwargs, auto_maskandscale)
        open_strategy = _resolve_open_strategy(open_strategy, cache_handle)
        opener_cls = _FileHandleOpener if open_strategy == "file_handle" else _SharedStoreOpener
        self._opener = opener_cls(filename, engine, store_open_kwargs, auto_maskandscale)
        # The file content references the opener, not this file handler, so
        # deleting the file handler closes the file right away.
        self.file_content = NetCDF4FileContent(self._opener,
                                               listed_keys=_get_required_variable_names(filetype_info))
        if not defer_open:
            self._opener.open()

    @property
    def accessor(self):
        """Get the accessor for the engine of the file.

        With several engines to try, this opens the file (if it isn't yet) to
        choose one of them.

        """
        return self._opener.accessor

    def close(self):
        """Close every file object this file handler is holding open.

        Called automatically when the file handler is deleted. Call it directly
        to release the file before then, for example to write to it.

        """
        if self._opener is not None:
            self._opener.close()

    def __del__(self):
        """Delete the file handler."""
        self.close()

    def __getitem__(self, key):
        """Get item for given key."""
        if key in self.cached_variables:
            return self.cached_variables[key]
        val = self.file_content[key]
        if self.accessor.is_variable(val):
            if self._is_small_variable(val):
                return self._cache_variable(key)
            return self._opener.read_variable(key, self._open_dataset_kwargs)
        if self.accessor.is_group(val):
            return self._opener.read_group(key, self._open_dataset_kwargs)
        return val

    def _is_small_variable(self, var):
        """Tell if variable object ``var`` is smaller than ``cache_var_size``."""
        return (self._cache_var_size > 0
                and isinstance(var.dtype, np.dtype)  # vlen may be str
                and np.prod(var.shape) * var.dtype.itemsize < self._cache_var_size)

    def _cache_variable(self, key):
        """Read variable ``key`` into memory and keep it for the next reads."""
        val = self._opener.load_variable(key, self._open_dataset_kwargs)
        self.cached_variables[key] = val
        return val

    def __contains__(self, item):
        """Get item from file content."""
        return item in self.file_content

    def get(self, item, default=None):
        """Get item."""
        if item in self:
            return self[item]
        else:
            return default

    def get_and_cache_npxr(self, var_name):
        """Get and cache variable as DataArray[numpy]."""
        if var_name in self.cached_variables:
            return self.cached_variables[var_name]
        val = self.file_content[var_name]
        if self.accessor.is_variable(val):
            return self._cache_variable(var_name)
        return val


def _resolve_open_strategy(strategy, cache_handle):
    if cache_handle is not None:
        # 8< v1.0
        warnings.warn(
            "The 'cache_handle' argument is deprecated and will be removed in Satpy 1.0. "
            "Use open_strategy='file_handle' instead of cache_handle=True.",
            DeprecationWarning,
            stacklevel=3)
        # >8 v1.0
        handle_strategy = "file_handle" if cache_handle else DEFAULT_OPEN_STRATEGY
        if strategy is not None and strategy != handle_strategy:
            raise ValueError(f"cache_handle={cache_handle} conflicts with open_strategy={strategy!r}")
        strategy = handle_strategy
    if strategy is None:
        return DEFAULT_OPEN_STRATEGY
    if strategy not in OPEN_STRATEGIES:
        raise ValueError(f"Unknown open_strategy {strategy!r}, expected one of {OPEN_STRATEGIES}")
    return strategy


def _split_xarray_kwargs(xarray_kwargs, auto_maskandscale):
    """Split ``xarray_kwargs`` in options for opening the backend store and options for ``xarray.open_dataset``.

    Options of the backend store of any engine are used when opening the file,
    as the engine may not be known yet, as well as those in ``backend_kwargs``.
    The options for ``xarray.open_dataset`` default to chunks of ``CHUNK_SIZE``
    and to masking and scaling following ``auto_maskandscale``.

    """
    xarray_kwargs = xarray_kwargs or {}
    store_params = _get_store_open_params("netcdf4") | _get_store_open_params("h5netcdf")
    store_open_kwargs = dict(xarray_kwargs.get("backend_kwargs") or {})
    open_dataset_kwargs = {"chunks": CHUNK_SIZE, "mask_and_scale": auto_maskandscale}
    for key, val in xarray_kwargs.items():
        if key in store_params:
            store_open_kwargs[key] = val
        elif key not in ("engine", "backend_kwargs"):
            open_dataset_kwargs[key] = val
    return store_open_kwargs, open_dataset_kwargs


def _get_required_variable_names(filetype_info):
    """Get the ``required_netcdf_variables`` of the file type with their names composed, or None if there are none."""
    listed_variables = filetype_info.get("required_netcdf_variables")
    if not listed_variables:
        return None
    variable_name_replacements = filetype_info.get("variable_name_replacements")
    variable_names = []
    for var in listed_variables:
        if variable_name_replacements and "{" in var:
            _compose_replacement_names(variable_name_replacements, var, variable_names)
        else:
            variable_names.append(var)
    return variable_names


def _compose_replacement_names(variable_name_replacements, var, variable_names):
    for key in variable_name_replacements:
        vals = variable_name_replacements[key]
        for val in vals:
            if key in var:
                variable_names.append(var.format(**{key: val}))


class _NetCDF4Opener:
    """Open a netCDF file for one file handler, read from it and own everything opened from it.

    Subclasses implement the open strategies of the file handler: what is
    opened, what the metadata is read from and how variables are read. The
    file handler and its ``file_content`` both use this object, which doesn't
    reference either of them.

    The accessor of a single engine is known from the start. Of several
    engines, the first one that opens the file is chosen and used from then on.

    """

    def __init__(self, filename, engine, store_open_kwargs, auto_maskandscale):
        """Initialize the opener without opening anything."""
        self.filename = filename
        self._engine = engine
        self._store_open_kwargs = store_open_kwargs
        self._auto_maskandscale = auto_maskandscale
        self._accessor = choose_accessor_from_engine(engine) if isinstance(engine, str) else None
        self.file_handle = None
        self._root_store = None
        # ``xarray.Dataset`` objects held open until ``close``, keyed by group name. See ``_get_dataset``.
        self._datasets = {}

    @property
    def accessor(self):
        """Get the accessor for the engine of the file, opening the file first to choose between several engines."""
        if self._accessor is None:
            self.open()
        return self._accessor

    def open(self):
        """Open the file if it isn't open."""
        self.get_metadata_root()

    def get_metadata_root(self):
        """Get the raw netCDF4/h5netcdf root group that metadata is read from, opening the file if needed."""
        raise NotImplementedError

    def read_variable(self, key, open_dataset_kwargs):
        """Read variable ``key`` as a DataArray of a dask array."""
        raise NotImplementedError

    def load_variable(self, key, open_dataset_kwargs):
        """Read variable ``key`` into memory as a DataArray."""
        raise NotImplementedError

    def read_group(self, key, open_dataset_kwargs):
        """Read group ``key`` as a Dataset of dask arrays with xarray, whatever the open strategy."""
        # Copied so callers can modify metadata without touching the shared dataset.
        return self._get_dataset(key, open_dataset_kwargs).copy()

    def _open_with_engine(self, open_func):
        """Return ``open_func(accessor)`` with the accessor for the engine of the file.

        If the engine isn't known yet, the engines are tried in turn and the
        first one that works is chosen.

        """
        if self._accessor is not None:
            try:
                return open_func(self._accessor)
            except IOError:
                LOG.exception("Failed reading file %s. Possibly corrupted file", self.filename)
                raise

        def open_and_choose_engine(engine):
            accessor = choose_accessor_from_engine(engine)
            opened = open_func(accessor)
            self._accessor = accessor
            return opened

        return _open_with_engines(self._engine, open_and_choose_engine)

    def _get_root_store(self):
        """Get the xarray backend store of the whole file, (re)opening it if needed."""
        if self._root_store is None:
            self._root_store = self._open_with_engine(self._open_root_store)
        return self._root_store

    def _open_root_store(self, accessor):
        """Open the xarray backend store of the whole file with the engine of ``accessor``."""
        engine = accessor.engine
        # skip the options that only the backend store of the other engine has
        all_params = _get_store_open_params("netcdf4") | _get_store_open_params("h5netcdf")
        own_params = _get_store_open_params(engine)
        open_kwargs = {key: val for key, val in self._store_open_kwargs.items()
                       if key in own_params or key not in all_params}
        filename = open_file_or_filename(self.filename) if engine == "h5netcdf" else self.filename
        if isinstance(filename, os.PathLike):
            # xarray only uses its (cached) file manager for string paths
            filename = os.fspath(filename)
        return _get_store_class(engine).open(filename, mode="r", **open_kwargs)

    def _get_dataset(self, group, open_dataset_kwargs):
        """Get the dataset for ``group``, opening and remembering it if needed.

        The dataset is held open until ``close``. Opening and closing it once
        per variable instead would give every returned lazy array its own
        ``CachingFileManager``, and every manager its own entry in xarray's
        global file cache. That cache is an LRU of ``file_cache_maxsize``
        entries (128 by default), so reading more variables than that starts
        evicting entries and closing netCDF4 handles that sibling arrays of the
        same file are still reading through, which segfaults in libhdf5.

        Child stores share the file manager (and file cache slot) of the root
        store, so the file is only opened once however many groups are read.

        """
        if group not in self._datasets:
            root_store = self._get_root_store()
            store = root_store.get_child_store(group) if group else root_store
            self._datasets[group] = xr.open_dataset(store, **open_dataset_kwargs)
        return self._datasets[group]

    def close(self):
        """Close everything opened from the file. What can be reopened is reopened on the next access."""
        for nc in self._datasets.values():
            with suppress(RuntimeError):
                nc.close()
        self._datasets.clear()
        if self._root_store is not None:
            with suppress(RuntimeError):
                self._root_store.close()
        self._root_store = None
        if self.file_handle is not None:
            with suppress(RuntimeError):
                self.file_handle.close()


class _SharedStoreOpener(_NetCDF4Opener):
    """Opener for the "shared_store" open strategy: one xarray backend store for metadata and data.

    Variables are always read with xarray, never from the raw variable objects
    of the metadata: xarray disables masking and scaling on them, and its file
    cache may close and reopen the file handle they belong to.

    """

    def get_metadata_root(self):
        """Get the raw root group of the backend store, (re)opening the store if needed."""
        return self._get_root_store().ds

    def read_variable(self, key, open_dataset_kwargs):
        """Read variable ``key`` from the dataset of its group."""
        group, _, name = key.rpartition("/")
        val = self._get_dataset(group, open_dataset_kwargs)[name]
        # Even though `chunks` is specified in the kwargs, xarray
        # uses dask.arrays only for data variables that have at least
        # one dimension; for zero-dimensional data variables (scalar),
        # it uses its own lazy loading for scalars.  Loading them now keeps
        # them usable once the file handler and its datasets are gone.
        if not val.chunks:
            val.load()
        # Copied so callers can modify metadata without touching the shared dataset.
        return val.copy(deep=False)

    def load_variable(self, key, open_dataset_kwargs):
        """Read variable ``key`` from the dataset of its group into memory."""
        return self.read_variable(key, open_dataset_kwargs).load()


class _FileHandleOpener(_NetCDF4Opener):
    """Opener for the "file_handle" open strategy: a netCDF4/h5netcdf file handle kept open.

    Variables are read from the file handle without xarray decoding them. The
    file handle isn't reopened once closed. Groups are still read with xarray,
    through a backend store which is opened when a group is first read.

    """

    def get_metadata_root(self):
        """Get the file handle, opening the file if it wasn't yet."""
        if self.file_handle is None:
            self.file_handle = self._open_with_engine(self._open_file_handle)
        return self.file_handle

    def _open_file_handle(self, accessor):
        file_handle = accessor.create_file_handle(self.filename)
        if hasattr(file_handle, "set_auto_maskandscale"):
            file_handle.set_auto_maskandscale(self._auto_maskandscale)
        elif self._auto_maskandscale:
            file_handle.close()
            raise ValueError(f"open_strategy='file_handle' can't apply auto_maskandscale=True with the "
                             f"{accessor.engine} engine. Use another open_strategy or engine.")
        return file_handle

    def read_variable(self, key, open_dataset_kwargs):
        """Wrap variable ``key`` of the file handle in a dask array."""
        return self._to_dataarray(key, da.from_array)

    def load_variable(self, key, open_dataset_kwargs):
        """Read variable ``key`` of the file handle into memory.

        This is about ten times as fast for a small variable as computing the
        dask array of ``read_variable``.

        """
        return self._to_dataarray(key, _read_data)

    def _to_dataarray(self, key, get_data):
        var = self.file_handle[key]
        # Not getting coordinates as this is more work, therefore more
        # overhead, and those are not used downstream.
        return xr.DataArray(get_data(var), dims=var.dimensions, attrs=self.accessor.get_object_attrs(var),
                            name=var.name)


class NetCDF4FileContent(Mapping):
    """Lazy mapping of the variables, groups, attributes and dimensions of a netCDF file.

    The keys are the ones described in :class:`NetCDF4FileHandler`:

    - ``"group/subgroup"`` and ``"group/var_name"`` for groups and variables,
      whose values are the raw netCDF4/h5netcdf objects.
    - ``"group/var_name/dtype"``, ``".../shape"`` and ``".../dimensions"`` for
      the properties of a variable.
    - ``"group/var_name/attr/attr_name"`` and ``"group/attr/attr_name"`` for
      attributes, ``"/attr/attr_name"`` (or ``"attr/attr_name"``) for global
      attributes and ``"/attrs"`` for a dictionary of all global attributes.
    - ``"group/dimension/dim_name"`` and ``"/dimension/dim_name"`` for the size
      of a dimension (the current size for an unlimited dimension).

    A value is only read from the file when it is requested. Everything but
    the group and variable objects is remembered after that; those objects
    belong to a file handle that may be closed and reopened by xarray's file
    cache, so they are looked up again every time.

    Iterating (including ``len``, ``keys`` and ``items``) needs the full list of
    keys, which walks through the whole file the first time. If ``listed_keys``
    is given, iterating is limited to those keys, the properties and
    attributes of the listed variables. Any key can be looked up either way.

    The mapping is read-only.

    Args:
        opener: The object opening the file for the file handler, which gives the raw root group of the file
            (``get_metadata_root()``) and the ``accessor`` for its engine.
        listed_keys (list or None): The keys that iterating is limited to.

    """

    _VARIABLE_PROPERTIES = ("dtype", "shape", "dimensions")

    def __init__(self, opener, listed_keys=None):
        """Initialize the mapping without reading anything from the file."""
        self._opener = opener
        self._listed_keys = listed_keys
        self._cache = {}
        self._file_keys = None

    @property
    def _accessor(self):
        return self._opener.accessor

    def __getitem__(self, key):
        """Get the value of ``key``, reading it from the file if needed."""
        if key in self._cache:
            return self._cache[key]
        value, cacheable = self._read(key)
        if cacheable:
            self._cache[key] = value
        return value

    def __iter__(self):
        """Iterate over the keys, walking through the file the first time."""
        return iter(self._get_file_keys())

    def __len__(self):
        """Get the number of keys, walking through the file the first time."""
        return len(self._get_file_keys())

    def _get_file_keys(self):
        if self._file_keys is None:
            items = self._walk_listed_keys() if self._listed_keys is not None else self._walk_file()
            file_keys = {}
            for key, value, cacheable in items:
                file_keys[key] = None
                if cacheable:
                    self._cache.setdefault(key, value)
            self._file_keys = file_keys
        return self._file_keys

    def _read(self, key):
        """Read the value of ``key`` from the file and tell if it can be remembered."""
        root = self._opener.get_metadata_root()
        if key == "/attrs":
            return self._get_attrs(root), True
        for prefix in ("/attr/", "attr/"):
            if key.startswith(prefix):
                return self._get_attr(root, key[len(prefix):], key), True
        if key.startswith("/dimension/"):
            return self._get_dimension(root, key[len("/dimension/"):], key), True
        parts = key.split("/")
        obj = root
        for index, part in enumerate(parts):
            child = self._get_child(obj, part)
            if child is not None:
                obj = child
                continue
            return self._read_from_object(obj, part, parts[index + 1:], key), True
        return obj, False

    def _read_from_object(self, obj, part, rest, key):
        """Read the attribute, dimension or variable property of ``obj`` that the rest of ``key`` refers to."""
        if part == "attr" and rest:
            return self._get_attr(obj, "/".join(rest), key)
        is_variable = self._accessor.is_variable(obj)
        if part == "dimension" and len(rest) == 1 and not is_variable:
            return self._get_dimension(obj, rest[0], key)
        if part in self._VARIABLE_PROPERTIES and not rest and is_variable:
            return getattr(obj, part)
        raise KeyError(key)

    def _get_child(self, obj, name):
        """Get the group or variable ``name`` of group ``obj``, or None."""
        if self._accessor.is_variable(obj):
            return None
        for children in (obj.groups, obj.variables):
            if name in children:
                return children[name]
        return None

    def _get_attr(self, obj, name, key):
        if name not in self._accessor.get_object_attrs(obj):
            raise KeyError(key)
        value = self._accessor.get_attr(obj, name)
        with suppress(ValueError):
            value = np2str(value)
        return value

    def _get_attrs(self, obj):
        return {name: self._get_attr(obj, name, name) for name in self._accessor.get_object_attrs(obj)}

    @staticmethod
    def _get_dimension(obj, name, key):
        if name not in obj.dimensions:
            raise KeyError(key)
        return obj.dimensions[name].size

    def _walk_file(self):
        """Walk through the whole file, yielding every key, its value and if the value can be remembered."""
        root = self._opener.get_metadata_root()
        yield from self._walk_group("", root)
        global_attrs = self._get_attrs(root)
        for name, value in global_attrs.items():
            yield f"/attr/{name}", value, True
        yield "/attrs", global_attrs, True
        yield from self._walk_dimensions("", root)

    def _walk_group(self, name, group):
        prefix = name + "/" if name else ""
        for group_name, subgroup in group.groups.items():
            full_name = prefix + group_name
            yield full_name, subgroup, False
            yield from self._walk_attrs(full_name, subgroup)
            yield from self._walk_group(full_name, subgroup)
            yield from self._walk_dimensions(full_name, subgroup)
        for var_name, var in group.variables.items():
            full_name = prefix + var_name
            yield full_name, var, False
            yield from self._walk_variable_properties(full_name, var)
            yield from self._walk_attrs(full_name, var)

    def _walk_variable_properties(self, name, var):
        for prop in self._VARIABLE_PROPERTIES:
            yield f"{name}/{prop}", getattr(var, prop), True

    def _walk_attrs(self, name, obj):
        for attr_name, value in self._get_attrs(obj).items():
            yield f"{name}/attr/{attr_name}", value, True

    @staticmethod
    def _walk_dimensions(name, group):
        for dim_name, dim in group.dimensions.items():
            yield f"{name}/dimension/{dim_name}", dim.size, True

    def _walk_listed_keys(self):
        """Yield the listed keys that exist in the file, with the properties and attributes of listed variables."""
        for key in self._listed_keys:
            try:
                value, cacheable = self._read(key)
            except KeyError:
                continue
            yield key, value, cacheable
            if self._accessor.is_variable(value):
                yield from self._walk_variable_properties(key, value)
                yield from self._walk_attrs(key, value)


def get_data_as_xarray(variable):
    """Get data in variable as xr.DataArray."""
    try:
        attrs = variable.attrs
    except AttributeError:
        # netCDF4 backend requires usage of __dict__ to get the attributes
        attrs = variable.__dict__
    return xr.DataArray(_read_data(variable), dims=variable.dimensions, attrs=attrs, name=variable.name)


def _read_data(variable):
    """Read all the data of a netCDF4/h5netcdf variable."""
    try:
        return variable[:]
    except (ValueError, IndexError):
        # Handle scalars for h5netcdf backend
        return variable.__array__()


def choose_accessor_from_engine(engine):
    """Choose an accessor from engine."""
    if engine == "netcdf4":
        return NetCDF4Accessor()
    elif engine == "h5netcdf":
        return H5NetcdfAccessor()
    raise NotImplementedError(f"Engine {engine} not implemented.")


def _open_with_engines(engines, open_with_engine):
    """Return the result of ``open_with_engine(engine)`` for the first engine that works."""
    for engine in engines:
        try:
            LOG.debug(f"Trying reading nc file with {engine} engine…")
            return open_with_engine(engine)
        except Exception as err:
            LOG.warning(f"Cannot use {engine} engine to read nc file.")
            LOG.debug(f"The error is: {str(err)}")
            continue
    else:
        raise RuntimeError("Could not work out an appropriate engine to open netCDF4 files")


def _get_store_class(engine):
    from xarray.backends import H5NetCDFStore, NetCDF4DataStore

    return {"netcdf4": NetCDF4DataStore, "h5netcdf": H5NetCDFStore}[engine]


def _get_store_open_params(engine):
    """Get the names of the options for opening the xarray backend store of ``engine``."""
    return set(inspect.signature(_get_store_class(engine).open).parameters) - {"filename", "mode", "group"}


class NetCDF4Accessor:
    """Accessor using the netCDF4 library as engine."""
    engine = "netcdf4"

    def create_file_handle(self, filename):
        """Create a file handle."""
        import netCDF4
        return netCDF4.Dataset(filename, "r")

    @staticmethod
    def is_variable(obj):
        """Check if obj is a variable."""
        import netCDF4
        return isinstance(obj, netCDF4.Variable)

    @staticmethod
    def is_group(obj):
        """Check if obj is a group."""
        import netCDF4
        return isinstance(obj, netCDF4.Group)

    @staticmethod
    def get_attr(obj, key):
        """Get an attribute from obj."""
        return getattr(obj, key)

    @staticmethod
    def get_object_attrs(obj):
        """Get the attributes for obj."""
        try:
            return obj.__dict__
        except KeyError:
            # Maybe unrecognised datatype, retrieve recoverable attributes.
            atts = {}
            for attname in obj.ncattrs():
                try:
                    atts[attname] = obj.getncattr(attname)
                except KeyError:
                    LOG.warning(f"Warning: Cannot load object ({obj.name}) attribute ({attname}).")
            return atts


class H5NetcdfAccessor:
    """Accessor using the h5netcdf library as engine."""
    engine = "h5netcdf"

    def create_file_handle(self, filename):
        """Create a file handle."""
        import h5netcdf
        f_obj = open_file_or_filename(filename)
        return h5netcdf.File(f_obj, "r")

    @staticmethod
    def is_variable(obj):
        """Check if obj is a variable."""
        import h5netcdf
        return isinstance(obj, h5netcdf.Variable)

    @staticmethod
    def is_group(obj):
        """Check if obj is a group."""
        import h5netcdf
        return isinstance(obj, h5netcdf.Group)

    @staticmethod
    def get_object_attrs(obj):
        """Get the attributes for obj."""
        return obj.attrs

    @staticmethod
    def get_attr(obj, key):
        """Get an attribute from obj."""
        return obj.attrs[key]


class NetCDF4FsspecFileHandler(NetCDF4FileHandler):
    """NetCDF4FileHandler implementation that allows accessing files on remote filesystems by switching engines."""

    def __init__(self, *args, engine=["netcdf4", "h5netcdf"], **kwargs):
        """Set up the instance with h5netcdf if netcdf4 does not work."""
        super().__init__(*args, engine=engine, **kwargs)
