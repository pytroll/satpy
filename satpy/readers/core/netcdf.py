"""Helpers for reading netcdf-based files."""

import logging
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

# HDF5 dimension-scale bookkeeping attrs that h5netcdf hides from users.
# When reading attrs directly via h5py (bypassing the h5netcdf Attributes proxy)
# we must filter these out to match h5netcdf's view.
_H5_HIDDEN_ATTRS = frozenset([
    "REFERENCE_LIST", "CLASS", "DIMENSION_LIST", "NAME",
    "_Netcdf4Dimid", "_Netcdf4Coordinates", "_nc3_strict", "_NCProperties",
])


def _h5attrs_to_dict(h5attrs):
    """Convert an h5py AttributeManager to a plain dict, matching h5netcdf behaviour.

    Filters out internal HDF5/NetCDF4 bookkeeping attributes and squeezes
    single-element arrays to scalars (matching what h5netcdf.Attributes does).
    """
    result = {}
    for k, v in h5attrs.items():
        if k in _H5_HIDDEN_ATTRS:
            continue
        # h5netcdf squeezes length-1 arrays to scalars (core.py Attributes.__getitem__)
        if not np.isscalar(v) and hasattr(v, "__len__") and len(v) == 1:
            v = v[0]
        result[k] = v
    return result


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

    Note that loading datasets requires reopening the original file
    (unless those datasets are cached, see below), but to get just the
    shape of the dataset append "/shape" to the item string:

        wrapper["group/subgroup/var_name/shape"]

    If your file has many small data variables that are frequently accessed,
    you may choose to cache some of them. You can do this by passing a number,
    any variable smaller than this number in bytes will be read into RAM.
    Warning, this part of the API is provisional and subject to change.

    You may get an additional speedup by passing ``cache_handle=True``. This
    will keep the netCDF4 dataset handles open throughout the lifetime of the
    object, and instead of using `xarray.open_dataset` to open every data
    variable, a dask array will be created "manually". This may be useful if
    you have a dataset distributed over many files, such as for FCI. Note
    that the coordinates will be missing in this case. If you use this option,
    ``xarray_kwargs`` will have no effect.

    Args:
        filename (str): File to read.
        filename_info (dict): Dictionary with filename information.
        filetype_info (dict): Dictionary with filetype information.
        auto_maskandscale (bool): Apply mask and scale factors.
        xarray_kwargs (dict): Addition arguments to `xarray.open_dataset`.
        cache_var_size (int): Cache variables smaller than this size.
        cache_handle (bool): Keep files open for lifetime of filehandler.
        engine (str or list of str): The engine to use for reading, either "netcdf4" or "h5netcdf". As a list, will try
            each engine until one works.

    """

    file_handle = None

    def __init__(
        self,
        filename,
        filename_info,
        filetype_info,
        auto_maskandscale=False,
        xarray_kwargs=None,
        cache_var_size=0,
        cache_handle=False,
        engine="netcdf4",
    ):
        """Initialize object."""
        super().__init__(filename, filename_info, filetype_info)
        self.file_content = {}
        self.cached_file_content = {}
        self.engine = engine
        self._auto_maskandscale = auto_maskandscale
        self._cache_var_size = cache_var_size
        self._cache_handle = cache_handle
        self._xarray_kwargs_input = xarray_kwargs
        self._initialized = False

    def _ensure_initialized(self):
        """Lazily initialize the file handler by opening the file and collecting metadata."""
        if self._initialized:
            return

        try:
            self.accessor, file_handle = self.get_accessor_and_filehandle()
        except IOError:
            LOG.exception("Failed reading file %s. Possibly corrupted file", self.filename)
            raise
        self._set_file_handle_auto_maskandscale(file_handle, self._auto_maskandscale)
        self._set_xarray_kwargs(self._xarray_kwargs_input, self._auto_maskandscale)

        filetype_info = self.filetype_info
        listed_variables = filetype_info.get("required_netcdf_variables")
        if listed_variables:
            if self._cache_handle:
                # Lazy mode: keep file handle open; variables are navigated on demand.
                pass
            else:
                self._collect_listed_variables(file_handle, listed_variables)
                self.collect_cache_vars(self._cache_var_size)
        else:
            self.collect_metadata("", file_handle)
            self.collect_dimensions("", file_handle)
            self.collect_cache_vars(self._cache_var_size)

        if self._cache_handle:
            self.file_handle = file_handle
        else:
            file_handle.close()

        self._initialized = True

    def get_accessor_and_filehandle(self):
        """Choose the accessor based on the engine, and return in along with the file handle."""
        if not isinstance(self.engine, str):
            return get_accessor_and_filehandle_from_engines(self.filename, *self.engine)
        return get_accessor_and_filehandle_from_engine(self.filename, self.engine)

    @staticmethod
    def _set_file_handle_auto_maskandscale(file_handle, auto_maskandscale):
        if hasattr(file_handle, "set_auto_maskandscale"):
            file_handle.set_auto_maskandscale(auto_maskandscale)

    def _set_xarray_kwargs(self, xarray_kwargs, auto_maskandscale):
        self._xarray_kwargs = xarray_kwargs or {}
        self._xarray_kwargs.setdefault("chunks", CHUNK_SIZE)
        self._xarray_kwargs.setdefault("mask_and_scale", auto_maskandscale)
        self._xarray_kwargs["engine"] = self.accessor.engine

    def collect_metadata(self, name, obj):
        """Collect all file variables and attributes for the provided file object.

        This method also iterates through subgroups of the provided object.
        """
        # Look through each subgroup
        base_name = name + "/" if name else ""
        self._collect_groups_info(base_name, obj)
        self._collect_variables_info(base_name, obj)
        if not name:
            self._collect_global_attrs(obj)
        else:
            self._collect_attrs(name, obj)

    def _collect_groups_info(self, base_name, obj):
        for group_name, group_obj in obj.groups.items():
            full_group_name = base_name + group_name
            self.file_content[full_group_name] = group_obj
            self._collect_attrs(full_group_name, group_obj)
            self.collect_metadata(full_group_name, group_obj)

    def _collect_variables_info(self, base_name, obj):
        for var_name, var_obj in obj.variables.items():
            var_name = base_name + var_name
            self._collect_variable_info(var_name, var_obj)

    def _collect_variable_info(self, var_name, var_obj):
        self.file_content[var_name] = var_obj
        self.file_content[var_name + "/dtype"] = var_obj.dtype
        self.file_content[var_name + "/shape"] = var_obj.shape
        self.file_content[var_name + "/dimensions"] = var_obj.dimensions
        self._collect_attrs(var_name, var_obj)

    def _collect_listed_variables(self, file_handle, listed_variables):
        variable_name_replacements = self.filetype_info.get("variable_name_replacements")
        for itm in self._get_required_variable_names(listed_variables, variable_name_replacements):
            parts = itm.split("/")
            grp = file_handle
            for p in parts[:-1]:
                if p == "attr":
                    n = "/".join(parts)
                    self.file_content[n] = self._get_attr_value(grp, parts[-1])
                    break
                grp = grp[p]
            if p != "attr":
                var_obj = grp[parts[-1]]
                self._collect_variable_info(itm, var_obj)
                self.collect_dimensions(itm, grp)

    @staticmethod
    def _get_required_variable_names(listed_variables, variable_name_replacements):
        variable_names = []
        for var in listed_variables:
            if variable_name_replacements and "{" in var:
                _compose_replacement_names(variable_name_replacements, var, variable_names)
            else:
                variable_names.append(var)
        return variable_names

    def __del__(self):
        """Delete the file handler."""
        if self.file_handle is not None:
            with suppress(RuntimeError):
                self.file_handle.close()

    def _collect_global_attrs(self, obj):
        """Collect all the global attributes for the provided file object."""
        global_attrs = {}
        for key in self.accessor.get_object_attrs(obj):
            fc_key = f"/attr/{key}"
            value = self._get_attr_value(obj, key)
            self.file_content[fc_key] = global_attrs[key] = value
        self.file_content["/attrs"] = global_attrs

    def _collect_attrs(self, name, obj):
        """Collect all the attributes for the provided file object."""
        for key in self.accessor.get_object_attrs(obj):
            fc_key = f"{name}/attr/{key}"
            value = self._get_attr_value(obj, key)
            self.file_content[fc_key] = value

    def _get_attr_value(self, obj, key):
        value = self.accessor.get_attr(obj, key)
        try:
            value = np2str(value)
        except ValueError:
            pass
        return value

    def collect_dimensions(self, name, obj):
        """Collect dimensions."""
        for dim_name, dim_obj in obj.dimensions.items():
            dim_name = "{}/dimension/{}".format(name, dim_name)
            self.file_content[dim_name] = len(dim_obj)

    def collect_cache_vars(self, cache_var_size):
        """Collect data variables for caching.

        This method will collect some data variables and store them in RAM.
        This may be useful if some small variables are frequently accessed,
        to prevent needlessly frequently opening and closing the file, which
        in case of xarray is associated with some overhead.

        Should be called later than `collect_metadata`.

        Args:
            cache_var_size (int): Maximum size of the collected variables in bytes

        """
        if cache_var_size == 0:
            return

        cache_vars = self._collect_cache_var_names(cache_var_size)
        for var_name in cache_vars:
            v = self.file_content[var_name]
            arr = get_data_as_xarray(v)
            self.cached_file_content[var_name] = arr

    def _collect_cache_var_names(self, cache_var_size):
        return [
            varname
            for (varname, var) in self.file_content.items()
            if self.accessor.is_variable(var)
            and isinstance(var.dtype, np.dtype)  # vlen may be str
            and var.size * var.dtype.itemsize < cache_var_size
        ]

    def _lazily_navigate(self, key):
        """Navigate the open file handle to find *key* and populate file_content.

        Only active when cache_handle=True (file_handle is open). Navigates only
        the single path requested; results are cached so repeated access is free.
        """
        if self.file_handle is None:
            return

        # Normalise away any leading slash
        normalized = key.lstrip("/")
        if not normalized:
            return

        # For /shape, /dtype, /dimensions sub-keys, navigate to the parent variable first
        for suffix in ("/shape", "/dtype", "/dimensions"):
            if normalized.endswith(suffix):
                parent = normalized[: -len(suffix)]
                if parent not in self.file_content:
                    self._lazily_navigate(parent)
                return

        parts = normalized.split("/")

        # Detect attribute access: "some/group/attr/name"
        if "attr" in parts:
            self._lazily_navigate_attribute(parts, parts.index("attr"))
        else:
            self._lazily_navigate_node(normalized, parts)

    def _walk_file_handle(self, group_parts):
        """Walk *group_parts* down from the file handle, returning the object reached or None if not found."""
        obj = self.file_handle
        for p in group_parts:
            try:
                obj = obj[p]
            except (KeyError, AttributeError, IndexError):
                return None
        return obj

    def _lazily_navigate_attribute(self, parts, attr_idx):
        group_parts = [p for p in parts[:attr_idx] if p]
        obj = self._walk_file_handle(group_parts)
        if obj is None:
            return
        parent_path = "/".join(group_parts)
        if attr_idx + 1 < len(parts):
            attr_name = parts[attr_idx + 1]
            fc_key = f"{parent_path}/attr/{attr_name}" if parent_path else f"attr/{attr_name}"
            with suppress(Exception):
                self.file_content[fc_key] = self._get_attr_value(obj, attr_name)
        elif parent_path:
            self._collect_attrs(parent_path, obj)
        else:
            self._collect_global_attrs(obj)

    def _lazily_navigate_node(self, normalized, parts):
        obj = self._walk_file_handle([p for p in parts[:-1] if p])
        if obj is None:
            return
        try:
            child = obj[parts[-1]]
        except (KeyError, AttributeError, IndexError):
            return

        if self.accessor.is_variable(child):
            self._collect_variable_info(normalized, child)
            self.collect_dimensions(normalized, obj)
        elif self.accessor.is_group(child):
            self.file_content[normalized] = child
            self._collect_attrs(normalized, child)
        else:
            self.file_content[normalized] = child

    def __getitem__(self, key):
        """Get item for given key."""
        self._ensure_initialized()
        if key not in self.file_content and self.file_handle is not None:
            self._lazily_navigate(key)
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
        if self.file_handle is not None:
            val = self._get_var_from_filehandle(group, key)
        else:
            val = self._get_var_from_xr(group, key)
        return val

    def _get_group(self, key, val):
        """Get a group from the netcdf file."""
        # Full groups are conveniently read with xr even if file_handle is available
        with xr.open_dataset(self.filename, group=key, **self._xarray_kwargs) as nc:
            val = nc
        return val

    def _get_var_from_xr(self, group, key):
        with xr.open_dataset(self.filename, group=group, **self._xarray_kwargs) as nc:
            val = nc[key]
            # Even though `chunks` is specified in the kwargs, xarray
            # uses dask.arrays only for data variables that have at least
            # one dimension; for zero-dimensional data variables (scalar),
            # it uses its own lazy loading for scalars.  When those are
            # accessed after file closure, xarray reopens the file without
            # closing it again.  This will leave potentially many open file
            # objects (which may in turn trigger a Segmentation Fault:
            # https://github.com/pydata/xarray/issues/2954#issuecomment-491221266
            if not val.chunks:
                val.load()
        return val

    def _get_var_from_filehandle(self, group, key):
        # Not getting coordinates as this is more work, therefore more
        # overhead, and those are not used downstream.
        if group is None:
            g = self.file_handle
        else:
            g = self.file_handle[group]
        v = g[key]
        # Fast attrs path: reading attrs via the h5py Dataset directly is ~2× faster
        # than iterating through h5netcdf's Attributes proxy, which goes through
        # h5py's low-level API one attribute at a time.
        if hasattr(v, "_h5ds"):
            attrs = _h5attrs_to_dict(v._h5ds.attrs)
        else:
            attrs = self.accessor.get_object_attrs(v)
        x = xr.DataArray(da.from_array(v), dims=v.dimensions, attrs=attrs, name=v.name)
        return x

    def __contains__(self, item):
        """Get item from file content."""
        self._ensure_initialized()
        if item not in self.file_content and self.file_handle is not None:
            self._lazily_navigate(item)
        return item in self.file_content

    def get(self, item, default=None):
        """Get item."""
        self._ensure_initialized()
        if item in self:
            return self[item]
        else:
            return default

    def get_and_cache_npxr(self, var_name):
        """Get and cache variable as DataArray[numpy]."""
        self._ensure_initialized()
        if var_name in self.cached_file_content:
            return self.cached_file_content[var_name]

        # Fast path for h5netcdf engine: bypass h5netcdf Group navigation and
        # _collect_attrs overhead by reading the HDF5 dataset directly via h5py.
        # This avoids creating intermediate Group objects (Group.__init__ iterates
        # all group children) and redundant attribute reads.
        # Attr-path requests (containing "/attr/") are handled by the fallback below.
        if "/attr/" not in var_name and self.file_handle is not None and hasattr(self.file_handle, "_h5file"):
            val = self._read_npxr_via_h5py(var_name)
            if val is not None:
                self.cached_file_content[var_name] = val
                return val

        if var_name not in self.file_content and self.file_handle is not None:
            self._lazily_navigate(var_name)
        v = self.file_content[var_name]
        if isinstance(v, xr.DataArray):
            val = v
        else:
            try:
                val = get_data_as_xarray(v)
            except AttributeError:
                # Handle strings
                val = v
        self.cached_file_content[var_name] = val
        return self.cached_file_content[var_name]

    def _read_npxr_via_h5py(self, var_name):
        """Read a variable directly via h5py, bypassing h5netcdf Group navigation.

        Returns an xr.DataArray backed by numpy data, or None if the fast path
        cannot be used (e.g. the path is not found or not a Dataset).
        """
        import h5py
        h5file = self.file_handle._h5file
        try:
            obj = h5file[var_name]
        except KeyError:
            return None
        if not isinstance(obj, h5py.Dataset):
            return None
        try:
            # Use [()] for scalars (0-D); use [:] for arrays.  Both return numpy.
            data = obj[()] if obj.ndim == 0 else obj[:]
        except Exception:
            return None
        attrs = _h5attrs_to_dict(obj.attrs)
        name = var_name.rsplit("/", 1)[-1]
        # Resolve dimension names from the cache built during file initialisation.
        # Scalars have no dimensions; for arrays we fall back to generic names if
        # the variable is not in the cache (rare; affects only unindexed LUTs).
        if obj.ndim == 0:
            dims = ()
        else:
            dims = self.file_handle._dim_ref_cache.get(obj.name)
            if dims is None:
                dims = tuple(f"dim_{i}" for i in range(obj.ndim))
        return xr.DataArray(data, dims=dims, attrs=attrs, name=name)

    def _get_attr(self, obj, key):
        return self.accessor.get_attr(obj, key)

    def _get_object_attrs(self, obj):
        return self.accessor.get_object_attrs(obj)


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
    for engine in engines:
        try:
            LOG.info(f"Trying reading nc file with {engine} engine…")
            return get_accessor_and_filehandle_from_engine(filename, engine)
        except Exception as err:
            LOG.warning(f"Cannot use {engine} engine to read nc file.")
            LOG.debug(f"The error is: {str(err)}")
            continue
    else:
        raise RuntimeError("Could not work out an appropriate engine to open netCDF4 files")


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
