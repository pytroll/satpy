"""Helpers for reading netcdf-based files."""

import inspect
import logging
import os
import warnings
import weakref
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
    any variable smaller than this number in bytes will be read into RAM.
    Warning, this part of the API is provisional and subject to change.

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

    Args:
        filename (str): File to read.
        filename_info (dict): Dictionary with filename information.
        filetype_info (dict): Dictionary with filetype information.
        auto_maskandscale (bool): Apply mask and scale factors.
        xarray_kwargs (dict): Additional arguments to `xarray.open_dataset`. Options of the xarray backend
            store (e.g. ``lock`` or ``phony_dims``, or those in ``backend_kwargs``) are used when opening the file.
        cache_var_size (int): Cache variables smaller than this size.
        cache_handle (bool): Deprecated, use ``open_strategy="file_handle"`` instead of ``cache_handle=True``.
        engine (str or list of str): The engine to use for reading, either "netcdf4" or "h5netcdf". As a list, will try
            each engine until one works.
        open_strategy (str): How to open the file and read its variables. One of "shared_store" (default)
            or "file_handle". See above.

    """

    file_handle = None
    _open_strategy = DEFAULT_OPEN_STRATEGY
    # ``xarray.Dataset`` objects held open for the lifetime of this file
    # handler, keyed by group name. See ``_open_xr_dataset``.
    _open_datasets = None
    # Backend store the datasets above are derived from for the
    # "shared_store" open strategy.
    _root_store = None

    def __init__(self, filename, filename_info, filetype_info,
                 auto_maskandscale=False, xarray_kwargs=None,
                 cache_var_size=0, cache_handle=None, engine="netcdf4",
                 open_strategy=None):
        """Initialize object."""
        super().__init__(filename, filename_info, filetype_info)
        self.file_content = {}
        self.cached_file_content = {}
        self.engine = engine
        self._open_strategy = _resolve_open_strategy(open_strategy, cache_handle)
        self._set_xarray_kwargs(xarray_kwargs, auto_maskandscale)
        try:
            self.accessor, opened_file = self._open_file()
        except IOError:
            LOG.exception(
                "Failed reading file %s. Possibly corrupted file", self.filename)
            raise
        if self._open_strategy == "file_handle":
            self._check_file_handle_can_maskandscale(opened_file, auto_maskandscale)
            self._set_file_handle_auto_maskandscale(opened_file, auto_maskandscale)
            self.file_handle = opened_file
        else:
            self._root_store = opened_file
        self.file_content = NetCDF4FileContent(self.accessor, _weak_root_getter(self),
                                               listed_keys=self._get_listed_keys(filetype_info))
        self.collect_cache_vars(cache_var_size)

    def _open_file(self):
        """Open the file following ``open_strategy``.

        Returns:
            The accessor for the engine that could open the file and the opened
            file: the netCDF4/h5netcdf file handle for the "file_handle" open
            strategy, the xarray backend store of the whole file otherwise.

        """
        if self._open_strategy == "file_handle":
            return self.get_accessor_and_filehandle()

        def open_with_engine(engine):
            accessor = choose_accessor_from_engine(engine)
            return accessor, _open_root_store(self.filename, engine, self._store_open_kwargs)

        if isinstance(self.engine, str):
            return open_with_engine(self.engine)
        return _open_with_engines(self.engine, open_with_engine)

    def get_accessor_and_filehandle(self):
        """Choose the accessor based on the engine, and return in along with the file handle."""
        if not isinstance(self.engine, str):
            return get_accessor_and_filehandle_from_engines(self.filename, *self.engine)
        return get_accessor_and_filehandle_from_engine(self.filename, self.engine)

    def _check_file_handle_can_maskandscale(self, file_handle, auto_maskandscale):
        """Refuse reading unscaled data with the "file_handle" strategy and an engine that can't mask and scale."""
        if auto_maskandscale and not hasattr(file_handle, "set_auto_maskandscale"):
            file_handle.close()
            raise ValueError(f"open_strategy='file_handle' can't apply auto_maskandscale=True with the "
                             f"{self.accessor.engine} engine. Use another open_strategy or engine.")

    @staticmethod
    def _set_file_handle_auto_maskandscale(file_handle, auto_maskandscale):
        if hasattr(file_handle, "set_auto_maskandscale"):
            file_handle.set_auto_maskandscale(auto_maskandscale)

    def _set_xarray_kwargs(self, xarray_kwargs, auto_maskandscale):
        """Split ``xarray_kwargs`` in options for opening the backend store and for ``xarray.open_dataset``."""
        self._store_open_kwargs, self._xarray_kwargs = _split_xarray_kwargs(xarray_kwargs or {})
        self._xarray_kwargs.setdefault("chunks", CHUNK_SIZE)
        self._xarray_kwargs.setdefault("mask_and_scale", auto_maskandscale)

    def _get_listed_keys(self, filetype_info):
        """Get the keys that iterating over ``file_content`` is limited to, if any."""
        listed_variables = filetype_info.get("required_netcdf_variables")
        if not listed_variables:
            return None
        return self._get_required_variable_names(listed_variables, filetype_info.get("variable_name_replacements"))

    @staticmethod
    def _get_required_variable_names(listed_variables, variable_name_replacements):
        variable_names = []
        for var in listed_variables:
            if variable_name_replacements and "{" in var:
                _compose_replacement_names(variable_name_replacements, var, variable_names)
            else:
                variable_names.append(var)
        return variable_names

    def close(self):
        """Close every file object this file handler is holding open.

        Called automatically when the file handler is deleted. Call it directly
        to release the file before then, for example to write to it.

        """
        if self.file_handle is not None:
            with suppress(RuntimeError):
                self.file_handle.close()
        self._close_open_datasets()

    def __del__(self):
        """Delete the file handler."""
        self.close()

    def _close_open_datasets(self):
        """Close the datasets held open by ``_open_xr_dataset`` and what they were derived from."""
        for nc in (self._open_datasets or {}).values():
            with suppress(RuntimeError):
                nc.close()
        if self._open_datasets:
            self._open_datasets.clear()
        if self._root_store is not None:
            with suppress(RuntimeError):
                self._root_store.close()
        self._root_store = None

    def collect_cache_vars(self, cache_var_size):
        """Collect data variables for caching.

        This method will collect some data variables and store them in RAM.
        This may be useful if some small variables are frequently accessed,
        to prevent needlessly frequently opening and closing the file, which
        in case of xarray is associated with some overhead.

        Args:
            cache_var_size (int): Maximum size of the collected variables in bytes

        """
        if cache_var_size == 0:
            return

        cache_vars = self._collect_cache_var_names(cache_var_size)
        for var_name in cache_vars:
            self.get_and_cache_npxr(var_name)

    def _collect_cache_var_names(self, cache_var_size):
        return [varname for (varname, var)
                in self.file_content.items()
                if self.accessor.is_variable(var)
                and isinstance(var.dtype, np.dtype)  # vlen may be str
                and np.prod(var.shape) * var.dtype.itemsize < cache_var_size]

    def __getitem__(self, key):
        """Get item for given key."""
        val = self.file_content[key]
        if self.accessor.is_variable(val):
            return self._get_variable(key, val)
        if self.accessor.is_group(val):
            return self._get_group(key, val)
        return val

    def _get_variable(self, key, val):
        """Get a variable from the netcdf file."""
        if key in self.cached_file_content:
            return self.cached_file_content[key]
        # these datasets are closed and inaccessible when the file is
        # closed, need to reopen
        # TODO: Handle HDF4 versus NetCDF3 versus NetCDF4
        parts = key.rsplit("/", 1)
        if len(parts) == 2:
            group, key = parts
        else:
            group = None
        if self._open_strategy == "file_handle":
            val = self._get_var_from_filehandle(group, key)
        else:
            val = self._get_var_from_xr(group, key)
        return val

    def _open_xr_dataset(self, group):
        """Get the dataset for ``group``, opening and remembering it if needed.

        The dataset is held open for the lifetime of this file handler. Opening
        and closing it once per variable instead would give every returned lazy
        array its own ``CachingFileManager``, and every manager its own entry in
        xarray's global file cache. That cache is an LRU of ``file_cache_maxsize``
        entries (128 by default), so reading more variables than that starts
        evicting entries and closing netCDF4 handles that sibling arrays of the
        same file are still reading through, which segfaults in libhdf5.

        """
        if self._open_datasets is None:
            self._open_datasets = {}
        if group not in self._open_datasets:
            self._open_datasets[group] = self._open_xr_dataset_for_group(group)
        return self._open_datasets[group]

    def _open_xr_dataset_for_group(self, group):
        """Open the dataset for ``group`` from the backend store of the whole file.

        Child stores share the file manager (and file cache slot) of the root
        store, so the file is only opened once however many groups are read.

        """
        root_store = self._get_root_store()
        store = root_store.get_child_store(group) if group else root_store
        return xr.open_dataset(store, **self._xarray_kwargs)

    def _get_root_store(self):
        """Get the xarray backend store of the whole file.

        It is reopened after ``close`` or, for the "file_handle" open strategy,
        opened on the first access.

        """
        if self._root_store is None:
            self._root_store = _open_root_store(self.filename, self.accessor.engine, self._store_open_kwargs)
        return self._root_store

    def _get_metadata_root(self):
        """Get the (raw) root group of the file that ``file_content`` reads from."""
        if self._open_strategy == "file_handle":
            return self.file_handle
        # xarray reopens the file if its file cache closed it in the meantime
        return self._get_root_store().ds

    def _get_group(self, key, val):
        """Get a group from the netcdf file."""
        # Full groups are conveniently read with xr even if file_handle is available
        # Copied so callers can modify metadata without touching the shared dataset.
        return self._open_xr_dataset(key).copy()

    def _get_var_from_xr(self, group, key):
        nc = self._open_xr_dataset(group)
        val = nc[key]
        # Even though `chunks` is specified in the kwargs, xarray
        # uses dask.arrays only for data variables that have at least
        # one dimension; for zero-dimensional data variables (scalar),
        # it uses its own lazy loading for scalars.  Loading them now keeps
        # them usable once this file handler and its datasets are gone.
        if not val.chunks:
            val.load()
        # Copied so callers can modify metadata without touching the shared dataset.
        return val.copy(deep=False)

    def _get_var_from_filehandle(self, group, key):
        # Not getting coordinates as this is more work, therefore more
        # overhead, and those are not used downstream.
        if group is None:
            g = self.file_handle
        else:
            g = self.file_handle[group]
        v = g[key]
        attrs = self.accessor.get_object_attrs(v)
        x = xr.DataArray(
                da.from_array(v), dims=v.dimensions, attrs=attrs,
                name=v.name)
        return x

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
        if var_name in self.cached_file_content:
            return self.cached_file_content[var_name]
        v = self.file_content[var_name]
        if isinstance(v, xr.DataArray):
            val = v
        elif self._open_strategy != "file_handle" and self.accessor.is_variable(v):
            # The variable object belongs to the file handle of the xarray
            # backend store, which may have been closed (and reopened) by
            # xarray's file cache since and has masking and scaling disabled
            # by xarray. Read the data through xarray instead.
            val = self[var_name].load()
        else:
            try:
                val = get_data_as_xarray(v)
            except AttributeError:
                # Handle strings
                val = v
        self.cached_file_content[var_name] = val
        return self.cached_file_content[var_name]

    def _get_attr(self, obj, key):
        return self.accessor.get_attr(obj, key)

    def _get_object_attrs(self, obj):
        return self.accessor.get_object_attrs(obj)

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


def _weak_root_getter(file_handler):
    """Get a function returning the root group of ``file_handler`` that doesn't keep the file handler alive.

    Referencing the file handler itself from its ``file_content`` would be a
    reference cycle, which delays closing the file until the garbage
    collector runs instead of when the file handler is deleted.

    """
    weak_get_root = weakref.WeakMethod(file_handler._get_metadata_root)

    def _get_root():
        get_root = weak_get_root()
        if get_root is None:
            raise ReferenceError("The file handler of this file content doesn't exist anymore.")
        return get_root()
    return _get_root


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
      of a dimension.

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
        accessor: The ``NetCDF4Accessor``/``H5NetcdfAccessor`` for the engine of the file.
        get_root (callable): Function returning the raw root group of the file.
        listed_keys (list or None): The keys that iterating is limited to.

    """

    _VARIABLE_PROPERTIES = ("dtype", "shape", "dimensions")

    def __init__(self, accessor, get_root, listed_keys=None):
        """Initialize the mapping without reading anything from the file."""
        self._accessor = accessor
        self._get_root = get_root
        self._listed_keys = listed_keys
        self._cache = {}
        self._file_keys = None

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
        root = self._get_root()
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
        return len(obj.dimensions[name])

    def _walk_file(self):
        """Walk through the whole file, yielding every key, its value and if the value can be remembered."""
        root = self._get_root()
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
            yield f"{name}/dimension/{dim_name}", len(dim), True

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


def _compose_replacement_names(variable_name_replacements, var, variable_names):
    for key in variable_name_replacements:
        vals = variable_name_replacements[key]
        for val in vals:
            if key in var:
                variable_names.append(var.format(**{key: val}))


def get_data_as_xarray(variable):
    """Get data in variable as xr.DataArray."""
    try:
        attrs = variable.attrs
    except AttributeError:
        # netCDF4 backend requires usage of __dict__ to get the attributes
        attrs = variable.__dict__
    try:
        data = variable[:]
    except (ValueError, IndexError):
        # Handle scalars for h5netcdf backend
        data = variable.__array__()

    arr = xr.DataArray(data, dims=variable.dimensions, attrs=attrs, name=variable.name)

    return arr


def choose_accessor_from_engine(engine):
    """Choose an accessor from engine."""
    if engine == "netcdf4":
        return NetCDF4Accessor()
    elif engine == "h5netcdf":
        return H5NetcdfAccessor()
    raise NotImplementedError(f"Engine {engine} not implemented.")


def get_accessor_and_filehandle_from_engine(filename, engine):
    """Choose an accessor from engine, and return in along with the file handle."""
    accessor = choose_accessor_from_engine(engine)
    file_handle = accessor.create_file_handle(filename)
    return accessor, file_handle


def get_accessor_and_filehandle_from_engines(filename, *engines):
    """Choose an accessor from the first possible engine, and return in along with the file handle."""
    return _open_with_engines(engines, lambda engine: get_accessor_and_filehandle_from_engine(filename, engine))


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


def _split_xarray_kwargs(xarray_kwargs):
    """Split ``xarray_kwargs`` in options for opening the backend store and options for ``xarray.open_dataset``.

    Options of the backend store of any engine are used when opening the file,
    as the engine may not be known yet, as well as those in ``backend_kwargs``.

    """
    store_params = _get_store_open_params("netcdf4") | _get_store_open_params("h5netcdf")
    open_kwargs = dict(xarray_kwargs.get("backend_kwargs") or {})
    decode_kwargs = {}
    for key, val in xarray_kwargs.items():
        if key in store_params:
            open_kwargs[key] = val
        elif key not in ("engine", "backend_kwargs"):
            decode_kwargs[key] = val
    return open_kwargs, decode_kwargs


def _open_root_store(filename, engine, store_open_kwargs):
    """Open the xarray backend store of the whole file with ``engine``."""
    store_cls = _get_store_class(engine)
    # skip the options that only the backend store of the other engine has
    all_params = _get_store_open_params("netcdf4") | _get_store_open_params("h5netcdf")
    own_params = _get_store_open_params(engine)
    open_kwargs = {key: val for key, val in store_open_kwargs.items()
                   if key in own_params or key not in all_params}
    if engine == "h5netcdf":
        filename = open_file_or_filename(filename)
    if isinstance(filename, os.PathLike):
        # xarray only uses its (cached) file manager for string paths
        filename = os.fspath(filename)
    return store_cls.open(filename, mode="r", **open_kwargs)


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
