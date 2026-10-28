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
from dask.array.core import normalize_chunks
from dask.base import tokenize
from xarray.backends import AbstractDataStore

from satpy.readers.core.file_handlers import BaseFileHandler
from satpy.readers.core.remote import open_file_or_filename
from satpy.readers.core.utils import np2str
from satpy.utils import get_legacy_chunk_size

LOG = logging.getLogger(__name__)
CHUNK_SIZE = get_legacy_chunk_size()

OPEN_STRATEGIES = ("shared_store", "file_handle")
DEFAULT_OPEN_STRATEGY = "shared_store"


class NetCDF4FileHandler(BaseFileHandler):
    """Class for inspecting a NetCDF4 file and retrieving its contents.

    Access Content
    ^^^^^^^^^^^^^^

    File information can be accessed using bracket notation. Variables are
    accessed by using:

        wrapper["var_name"]

    Or:

        wrapper["group/subgroup/var_name"]

    Attributes can be accessed by appending "/attr/attr_name" to the
    item string:

        wrapper["group/subgroup/var_name/attr/units"]

    This also works for dimensions using "dimension":

        wrapper["group/subgroup/var_name/dimension/dim_name"]

    Or the overall shape of a variable:

        wrapper["group/subgroup/var_name/shape"]

    For global attributes use:

        wrapper["/attr/platform_short_name"]

    Or for all of global attributes:

        wrapper["/attrs"]

    All of this information is read from the file lazily when it is requested,
    see :class:`NetCDF4FileContent` for the ``file_content`` mapping that
    holds it.

    Small Variable Caching
    ^^^^^^^^^^^^^^^^^^^^^^

    If your file has many small data variables that are frequently accessed,
    you may choose to cache some of them. You can do this by passing a size in
    bytes using the ``cache_var_size`` keyword argument (see below). Any item
    read that is smaller than this size will be cached on first access. Use the
    :meth:`get_and_cache_npxr` method to explicitly cache a variable regardless
    of size.

    Data Consistency
    ^^^^^^^^^^^^^^^^

    When data are ready into ``xarray.DataArray`` or ``xarray.Dataset`` objects
    the underlying data may be shared with later reads for that same item. It
    is important not to modify these objects in place including: their dask
    arrays, scalar variables, which are loaded
    into memory, and the arrays in their attributes. Modifying in place would
    be using patterns like ``data[...] = ...`` or assigning using ``+=`` on
    a scalar or by changing an array in ``attrs``. Always create new ones instead
    (``data = data * factor``). Adding, removing, or replacing attributes,
    coordinates, or the ``data`` of a returned object doesn't change later
    reads, except for cached variables, which are the same object every time.

    Opening and Closing
    ^^^^^^^^^^^^^^^^^^^

    The ``open_strategy`` keyword argument selects how the file is opened and how variables are
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
      and can't be read anymore once the file handler is closed or deleted.

    By default the file is opened when the file handler is created, so that
    an unreadable file fails right away. With ``defer_open=True`` the file
    is only opened when something is first read from it or opening would
    help determine what what engine to use so errors are only raised then.
    This saves opening files that are never
    read, but is only useful for readers whose ``start_time``, ``end_time``
    and ``available_datasets`` don't read from the file, as those are needed
    when the ``Scene`` is created.

    By default when the handler is deleted it uses the :meth:`release` method
    to release direct references to the files it has opened. The files will be
    closed once all dask arrays that reference
    them are deleted. This does not apply to the "file_handle" ``open_strategy``
    in which case files are closed immediately. In this case the file handler
    and therefore the reader that references it and the Scene that references
    the reader must not be deleted until all computations are done. This could
    result in an error or in the worst case could provide data from another
    file that was opened.

    Args:
        filename (str): File to read.
        filename_info (dict): Dictionary with filename information.
        filetype_info (dict): Dictionary with filetype information.
        auto_maskandscale (bool): Apply mask and scale factors.
        xarray_kwargs (dict): Additional arguments to `xarray.open_dataset` and xarray store object.
            Options of the xarray backend store (e.g. ``lock`` or ``phony_dims``, or those in ``backend_kwargs``)
            are used when opening the file.
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
        open_strategy = _resolve_open_strategy(open_strategy, cache_handle)
        opener_cls = _FileHandleOpener if open_strategy == "file_handle" else _SharedStoreOpener
        self._opener = opener_cls(filename, engine, xarray_kwargs, auto_maskandscale)
        # The file content references the opener, not this file handler, so
        # deleting the file handler releases the file right away (see ``__del__``).
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

        See the class docstring for more details.
        """
        if self._opener is not None:
            self._opener.close()

    def __del__(self):
        """Delete the file handler leaving open files open until all dask arrays using it are deleted.

        See the class docstring for more details.

        """
        if self._opener is not None:
            self._opener.release()

    def __getitem__(self, key):
        """Get item for given key.

        Don't modify a returned variable or group in place, see the class docstring.

        """
        return self._get_item(key, always_cache=False)

    def _get_item(self, key, always_cache):
        """Get item for given key, reading a variable into memory and keeping it if it is small or ``always_cache``."""
        if key in self.cached_variables:
            return self.cached_variables[key]
        val = self.file_content[key]
        if self.accessor.is_variable(val):
            if always_cache or self._is_small_variable(val):
                return self._cache_variable(key)
            return self._opener.read_variable(key)
        if self.accessor.is_group(val):
            return self._opener.read_group(key)
        return val

    def _is_small_variable(self, var):
        """Tell if variable object ``var`` is smaller than ``cache_var_size``."""
        return (self._cache_var_size > 0
                and isinstance(var.dtype, np.dtype)  # vlen may be str
                and np.prod(var.shape) * var.dtype.itemsize < self._cache_var_size)

    def _cache_variable(self, key):
        """Read variable ``key`` into memory and keep it for the next reads."""
        val = self._opener.load_variable(key)
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
        return self._get_item(var_name, always_cache=True)


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
    store_open_kwargs = dict(xarray_kwargs.get("backend_kwargs") or {})
    open_dataset_kwargs = {"chunks": CHUNK_SIZE, "mask_and_scale": auto_maskandscale}
    for key, val in xarray_kwargs.items():
        if key in _ALL_STORE_OPEN_PARAMS:
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

    The accessor of a single engine is known from the start. If several
    engines are provided the first one that opens the file successfully is
    used.

    The options for ``xarray.open_dataset`` are in ``open_dataset_kwargs``.
    A file handler can change them until it first reads a variable or group
    with xarray, as the datasets opened with them are kept until ``close``.

    """

    def __init__(self, filename, engine, xarray_kwargs, auto_maskandscale):
        """Initialize the opener without opening anything."""
        self.filename = filename
        self._engine = engine
        self._store_open_kwargs, self.open_dataset_kwargs = _split_xarray_kwargs(xarray_kwargs, auto_maskandscale)
        self._accessor = choose_accessor_from_engine(engine) if isinstance(engine, str) else None
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

    def read_variable(self, key):
        """Read variable ``key`` as a DataArray of a dask array."""
        raise NotImplementedError

    def load_variable(self, key):
        """Read variable ``key`` into memory as a DataArray."""
        raise NotImplementedError

    def read_group(self, key):
        """Read group ``key`` as a Dataset of dask arrays with xarray, whatever the open strategy."""
        # Copied so callers can modify metadata without touching the shared dataset. The data is still shared,
        # which is why callers must not modify it in place (see ``NetCDF4FileHandler``).
        return self._get_dataset(key).copy()

    def _open_with_engine(self, open_func):
        """Return ``open_func(accessor)`` with the accessor for the engine of the file."""
        if self._accessor is not None:
            try:
                return open_func(self._accessor)
            except IOError:
                LOG.exception("Failed reading file %s. Possibly corrupted file", self.filename)
                raise
        for engine in self._engine:
            LOG.debug(f"Trying reading nc file with {engine} engine…")
            try:
                accessor = choose_accessor_from_engine(engine)
                opened = open_func(accessor)
            except Exception as err:
                LOG.warning(f"Cannot use {engine} engine to read nc file.")
                LOG.debug(f"The error is: {str(err)}")
                continue
            self._accessor = accessor
            return opened
        raise RuntimeError("Could not work out an appropriate engine to open netCDF4 files")

    def _get_root_store(self):
        """Get the xarray backend store of the whole file, (re)opening it if needed."""
        if self._root_store is None:
            self._root_store = self._open_with_engine(self._open_root_store)
        return self._root_store

    def _open_root_store(self, accessor):
        """Open the xarray backend store of the whole file with the engine of ``accessor``."""
        engine = accessor.engine
        # skip the options that only the backend store of the other engine has
        other_params = _ALL_STORE_OPEN_PARAMS - _STORE_OPEN_PARAMS[engine]
        open_kwargs = {key: val for key, val in self._store_open_kwargs.items() if key not in other_params}
        filename = open_file_or_filename(self.filename) if engine == "h5netcdf" else self.filename
        if isinstance(filename, os.PathLike):
            # xarray only uses its (cached) file manager for string paths
            filename = os.fspath(filename)
        return _get_store_class(engine).open(filename, mode="r", **open_kwargs)

    def _get_dataset(self, group):
        """Get the dataset for ``group``, opening and remembering it if needed.

        The dataset is held open until ``close`` and reused to avoid filling
        xarray's LRU file cache (128 files by default using ``file_cache_maxsize``).
        Future updates to xarray's caching
        (see https://github.com/pydata/xarray/issues/11581) may change the need for this.

        """
        if group not in self._datasets:
            root_store = self._get_root_store()
            store = root_store.get_child_store(group) if group else root_store
            store = _TokenizedStore(store, self._tokenize(self._store_open_kwargs, group))
            self._datasets[group] = xr.open_dataset(store, **self.open_dataset_kwargs)
        return self._datasets[group]

    def _tokenize(self, *args):
        """Get a dask token of the file, its engine and ``args``."""
        return tokenize(self.filename, _get_mtime(self.filename), self.accessor.engine, *args)

    def close(self):
        """Close everything opened from the file. It is opened again when it is next read from."""
        for nc in self._datasets.values():
            with suppress(RuntimeError):
                nc.close()
        self._datasets.clear()
        if self._root_store is not None:
            with suppress(RuntimeError):
                self._root_store.close()
        self._root_store = None

    def release(self):
        """Stop holding what was opened from the file, leaving it to xarray to close the file."""
        self._datasets.clear()
        self._root_store = None


class _TokenizedStore(AbstractDataStore):
    """Wrapper of an xarray backend store that dask tokenizes as ``token``.

    ``xarray.open_dataset`` names the dask arrays it creates from a store after
    a dask token of the store. For the stores of xarray this token is random:
    it includes the id of their file manager and the token of their lock,
    which change for every store and every process respectively (see
    https://github.com/pydata/xarray/issues/11814). Wrapping them gives
    arrays read from the same file with the same options the same names, as
    ``xarray.open_dataset`` does for a file opened by its name, so that dask
    and the caches keyed by those names can reuse them. For example, the
    ``cache_dir`` of the nearest neighbour and bilinear resamplers is keyed
    by the hash of a swath definition, which pyresample computes from the
    names of the dask arrays of its longitudes and latitudes.

    Everything else is the wrapped store's, whatever xarray uses of it: other
    attributes are looked up on the wrapped store by ``__getattr__``, and the
    methods of ``AbstractDataStore``, which are found before ``__getattr__`` is
    tried, call those of the wrapped store (see ``_call_store_method``).

    """

    __slots__ = ("_store", "_token")

    def __init__(self, store, token):
        """Wrap ``store``."""
        self._store = store
        self._token = token

    def __getattr__(self, name):
        """Get attribute ``name`` of the wrapped store."""
        # Not ``self._store``, which calls this again and again when it isn't set yet (e.g. while unpickling).
        return getattr(object.__getattribute__(self, "_store"), name)

    def __dask_tokenize__(self):
        """Get the dask token of the store."""
        return self._token


def _call_store_method(name):
    """Create a ``_TokenizedStore`` method calling method ``name`` of the wrapped store."""
    def method(self, *args, **kwargs):
        return getattr(self._store, name)(*args, **kwargs)

    method.__name__ = name
    return method


for _name, _attr in vars(AbstractDataStore).items():
    if callable(_attr) and not _name.startswith("_"):
        setattr(_TokenizedStore, _name, _call_store_method(_name))


def _get_mtime(filename):
    """Get the modification time of local file ``filename``, or None for a remote file."""
    try:
        return os.path.getmtime(filename)
    except (TypeError, OSError):
        return None


class _SharedStoreOpener(_NetCDF4Opener):
    """Opener for the "shared_store" open strategy: one xarray backend store for metadata and data.

    Variables are always read with xarray including the decoding done for
    attributes (e.g. masking, scaling, coordinates, times).
    Xarray's file cache may close and reopen the file handle underneath and will
    be reopened as needed (ex. dask array computation).

    """

    def get_metadata_root(self):
        """Get the raw root group of the backend store, (re)opening the store if needed."""
        return self._get_root_store().ds

    def read_variable(self, key):
        """Read variable ``key`` from the dataset of its group."""
        group, _, name = key.rpartition("/")
        val = self._get_dataset(group)[name]
        # Even though `chunks` is specified in the kwargs, xarray
        # uses dask.arrays only for data variables that have at least
        # one dimension; for zero-dimensional data variables (scalar),
        # it uses its own lazy loading for scalars.  Loading them now keeps
        # xarray from reopening the file (and keeping it open) when they are
        # read after the file handler was closed.
        if not val.chunks:
            val.load()
        # Copied so callers can modify metadata without touching the shared dataset, see ``read_group``.
        return val.copy(deep=False)

    def load_variable(self, key):
        """Read variable ``key`` from the dataset of its group into memory."""
        return self.read_variable(key).load()


class _FileHandleOpener(_NetCDF4Opener):
    """Opener for the "file_handle" open strategy: a netCDF4/h5netcdf file handle kept open.

    Variables are read from the file handle without xarray decoding them. The
    file handle is reopened when something is read after ``close``, but the
    arrays read from it before can't be read anymore. Groups are still read
    with xarray, through a backend store which is opened when a group is first
    read.

    """

    def __init__(self, filename, engine, xarray_kwargs, auto_maskandscale):
        """Initialize the opener without opening anything."""
        super().__init__(filename, engine, xarray_kwargs, auto_maskandscale)
        self._auto_maskandscale = auto_maskandscale
        self.file_handle = None

    def close(self):
        """Close everything opened from the file, including the file handle."""
        super().close()
        self._close_file_handle()

    def release(self):
        """Stop holding what was opened from the file and close the file handle.

        The arrays read from the file handle shouldn't be read after it is closed.

        """
        super().release()
        self._close_file_handle()

    def _close_file_handle(self):
        if self.file_handle is not None:
            with suppress(RuntimeError):
                self.file_handle.close()
        # netCDF-C reuses the ids of closed files, so reading through the closed
        # file handle could silently read from another file opened since.
        self.file_handle = None

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

    def read_variable(self, key):
        """Wrap variable ``key`` of the file handle in a dask array.

        Dask can't tokenize the variable, so it would give the array a random
        name. It is named after the file, the options it is opened with and
        the chunks instead, see ``_TokenizedStore`` for why. The chunks are
        resolved like ``dask.array.from_array`` does for ``chunks="auto"``.

        """
        var = self.get_metadata_root()[key]
        chunks = normalize_chunks("auto", var.shape, dtype=var.dtype, previous_chunks=getattr(var, "chunks", None))
        name = f"{var.name}-{self._tokenize(self._auto_maskandscale, key, chunks)}"
        return self._to_dataarray(var, da.from_array(var, chunks=chunks, name=name))

    def load_variable(self, key):
        """Read variable ``key`` of the file handle into memory.

        This is about ten times as fast for a small variable as computing the
        dask array of ``read_variable``.

        """
        var = self.get_metadata_root()[key]
        return self._to_dataarray(var, _read_data(var))

    def _to_dataarray(self, var, data):
        # Don't get coordinates for historical reasons.
        # They weren't used downstream and take more work to gather.
        return xr.DataArray(data, dims=var.dimensions, attrs=self.accessor.get_object_attrs(var), name=var.name)


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
        value = self._read(key)
        self._remember(key, value)
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
            for key, value in items:
                file_keys[key] = None
                self._remember(key, value)
            self._file_keys = file_keys
        return self._file_keys

    def _remember(self, key, value):
        """Remember ``value`` for the next lookups of ``key``, unless it is a group or variable object."""
        if not (self._accessor.is_variable(value) or self._accessor.is_group(value)):
            self._cache.setdefault(key, value)

    def _read(self, key):
        """Read the value of ``key`` from the file."""
        root = self._opener.get_metadata_root()
        if key == "/attrs":
            return self._get_attrs(root)
        parts = key.split("/")
        if key.startswith("/"):
            # "/attr/..." and "/dimension/..." of the root group itself, even if it has a child named like that
            return self._read_from_object(root, parts[1], parts[2:], key)
        obj = root
        for index, part in enumerate(parts):
            child = self._get_child(obj, part)
            if child is None:
                return self._read_from_object(obj, part, parts[index + 1:], key)
            if obj is not root and part in ("attr", "dimension"):
                # The attributes and dimensions of a group come before its children named "attr" or "dimension", as
                # when walking through the file (those of the root group are under "/attr" and "/dimension").
                with suppress(KeyError):
                    return self._read_from_object(obj, part, parts[index + 1:], key)
            obj = child
        return obj

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
        try:
            value = self._accessor.get_attr(obj, name)
        except (AttributeError, KeyError):
            raise KeyError(key) from None
        return _decode_attr(value)

    def _get_attrs(self, obj):
        # The values come from a single read of all attributes. With netCDF4, every such
        # read gets each attribute from the file, so reading it per attribute is quadratic.
        return {name: _decode_attr(value) for name, value in self._accessor.get_object_attrs(obj).items()}

    @staticmethod
    def _get_dimension(obj, name, key):
        if name not in obj.dimensions:
            raise KeyError(key)
        return obj.dimensions[name].size

    def _walk_file(self):
        """Walk through the whole file, yielding every key and its value."""
        root = self._opener.get_metadata_root()
        yield from self._walk_group("", root)
        global_attrs = self._get_attrs(root)
        for name, value in global_attrs.items():
            yield f"/attr/{name}", value
        yield "/attrs", global_attrs
        yield from self._walk_dimensions("", root)

    def _walk_group(self, name, group):
        prefix = name + "/" if name else ""
        for group_name, subgroup in group.groups.items():
            full_name = prefix + group_name
            yield full_name, subgroup
            # walk attrs and dims before children in case some children are called "attr" or "dimension"
            yield from self._walk_attrs(full_name, subgroup)
            yield from self._walk_dimensions(full_name, subgroup)
            yield from self._walk_group(full_name, subgroup)
        for var_name, var in group.variables.items():
            full_name = prefix + var_name
            yield full_name, var
            yield from self._walk_variable_properties(full_name, var)
            yield from self._walk_attrs(full_name, var)

    def _walk_variable_properties(self, name, var):
        for prop in self._VARIABLE_PROPERTIES:
            yield f"{name}/{prop}", getattr(var, prop)

    def _walk_attrs(self, name, obj):
        for attr_name, value in self._get_attrs(obj).items():
            yield f"{name}/attr/{attr_name}", value

    @staticmethod
    def _walk_dimensions(name, group):
        for dim_name, dim in group.dimensions.items():
            yield f"{name}/dimension/{dim_name}", dim.size

    def _walk_listed_keys(self):
        """Yield the listed keys that exist in the file, with the properties and attributes of listed variables."""
        for key in self._listed_keys:
            try:
                value = self._read(key)
            except KeyError:
                continue
            yield key, value
            if self._accessor.is_variable(value):
                yield from self._walk_variable_properties(key, value)
                yield from self._walk_attrs(key, value)


def _decode_attr(value):
    """Convert attribute ``value`` to str if it is a single numpy string or bytes value, see :func:`np2str`."""
    with suppress(ValueError):
        value = np2str(value)
    return value


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


def _get_store_class(engine):
    from xarray.backends import H5NetCDFStore, NetCDF4DataStore

    return {"netcdf4": NetCDF4DataStore, "h5netcdf": H5NetCDFStore}[engine]


def _get_store_open_params(engine):
    """Get the names of the options for opening the xarray backend store of ``engine``."""
    return frozenset(inspect.signature(_get_store_class(engine).open).parameters) - {"filename", "mode", "group"}


# Looked up once, when this module is imported, so that replacing the ``open``
# method of a store later (e.g. with a mock in tests) doesn't change them.
_STORE_OPEN_PARAMS = {engine: _get_store_open_params(engine) for engine in ("netcdf4", "h5netcdf")}
_ALL_STORE_OPEN_PARAMS = _STORE_OPEN_PARAMS["netcdf4"] | _STORE_OPEN_PARAMS["h5netcdf"]


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
        return obj.getncattr(key)

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
