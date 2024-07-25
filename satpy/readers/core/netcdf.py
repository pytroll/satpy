# Copyright (c) 2016-2024 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Helpers for reading netcdf-based files."""

import functools
import glob
import logging
import os
import pickle  # nosec
import time
import warnings

import dask.array as da
import dask.distributed
import netCDF4
import numpy as np
import xarray as xr

import satpy
from satpy.readers.core.file_handlers import BaseFileHandler
from satpy.readers.core.remote import open_file_or_filename
from satpy.readers.core.utils import get_serializable_dask_array, np2str
from satpy.utils import get_legacy_chunk_size

LOG = logging.getLogger(__name__)
CHUNK_SIZE = get_legacy_chunk_size()


class NetCDF4FileHandler(BaseFileHandler):
    """Small class for inspecting a NetCDF4 file and retrieving its metadata/header data.

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
    you may choose to cache some of them.  You can do this by passing a number,
    any variable smaller than this number in bytes will be read into RAM.
    Warning, this part of the API is provisional and subject to change.

    You may get an additional speedup by passing ``cache_handle=True``.  This
    will keep the netCDF4 dataset handles open throughout the lifetime of the
    object, and instead of using `xarray.open_dataset` to open every data
    variable, a dask array will be created "manually".  This may be useful if
    you have a dataset distributed over many files, such as for FCI.  Note
    that the coordinates will be missing in this case.  If you use this option,
    ``xarray_kwargs`` will have no effect.

    Args:
        filename (str): File to read
        filename_info (dict): Dictionary with filename information
        filetype_info (dict): Dictionary with filetype information
        auto_maskandscale (bool): Apply mask and scale factors
        xarray_kwargs (dict): Addition arguments to `xarray.open_dataset`
        cache_var_size (int): Cache variables smaller than this size.
        cache_handle (bool): Keep files open for lifetime of filehandler.
            Uses xarray.backends.CachingFileManager, which uses a least
            recently used cache.

    """

    manager = None

    def __init__(self, filename, filename_info, filetype_info,
                 auto_maskandscale=False, xarray_kwargs=None,
                 cache_var_size=0, cache_handle=False):
        """Initialize object."""
        super(NetCDF4FileHandler, self).__init__(
            filename, filename_info, filetype_info)
        self.file_content = {}
        self.cached_file_content = {}
        self._use_h5netcdf = False
        self._auto_maskandscale = auto_maskandscale
        try:
            file_handle = self._get_file_handle()
        except IOError:
            LOG.exception(
                "Failed reading file %s. Possibly corrupted file", self.filename)
            raise

        self._set_file_handle_auto_maskandscale(file_handle, auto_maskandscale)
        self._set_xarray_kwargs(xarray_kwargs, auto_maskandscale)

        listed_variables = filetype_info.get("required_netcdf_variables")
        if listed_variables is not None:
            self._collect_listed_variables(file_handle, listed_variables)
        else:
            self.collect_metadata("", file_handle)
            self.collect_dimensions("", file_handle)
        self.collect_cache_vars(cache_var_size)

        if cache_handle:
            self.manager = xr.backends.CachingFileManager(
                    functools.partial(_NCDatasetWrapper,
                                      auto_maskandscale=auto_maskandscale),
                    self.filename, mode="r")
        else:
            file_handle.close()

    def _get_file_handle(self):
        return netCDF4.Dataset(self.filename, "r")

    @property
    def file_handle(self):
        """Backward-compatible way for file handle caching."""
        warnings.warn(
                "attribute .file_handle is deprecated, use .manager instead",
                DeprecationWarning)
        if self.manager is None:
            return None
        return self.manager.acquire()

    @staticmethod
    def _set_file_handle_auto_maskandscale(file_handle, auto_maskandscale):
        if hasattr(file_handle, "set_auto_maskandscale"):
            file_handle.set_auto_maskandscale(auto_maskandscale)

    def _set_xarray_kwargs(self, xarray_kwargs, auto_maskandscale):
        self._xarray_kwargs = xarray_kwargs or {}
        self._xarray_kwargs.setdefault("chunks", CHUNK_SIZE)
        self._xarray_kwargs.setdefault("mask_and_scale", auto_maskandscale)

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
        if self.manager is not None:
            self.manager.close()

    def _collect_global_attrs(self, obj):
        """Collect all the global attributes for the provided file object."""
        global_attrs = {}
        for key in self._get_object_attrs(obj):
            fc_key = f"/attr/{key}"
            value = self._get_attr_value(obj, key)
            self.file_content[fc_key] = global_attrs[key] = value
        self.file_content["/attrs"] = global_attrs

    @staticmethod
    def _get_object_attrs(obj):
        """Get object attributes using __dict__ but retrieve recoverable attributes on failure."""
        try:
            return obj.__dict__
        except KeyError:
            # Maybe unrecognised datatype.
            atts = {}
            for attname in obj.ncattrs():
                try:
                    atts[attname] = obj.getncattr(attname)
                except KeyError:
                    LOG.warning(f"Warning: Cannot load object ({obj.name}) attribute ({attname}).")
            return atts

    def _collect_attrs(self, name, obj):
        """Collect all the attributes for the provided file object."""
        for key in self._get_object_attrs(obj):
            fc_key = f"{name}/attr/{key}"
            value = self._get_attr_value(obj, key)
            self.file_content[fc_key] = value

    def _get_attr_value(self, obj, key):
        value = self._get_attr(obj, key)
        try:
            value = np2str(value)
        except ValueError:
            pass
        return value

    def _get_attr(self, obj, key):
        return getattr(obj, key)

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
        return [varname for (varname, var)
                in self.file_content.items()
                if isinstance(var, netCDF4.Variable)
                and isinstance(var.dtype, np.dtype)  # vlen may be str
                and var.size * var.dtype.itemsize < cache_var_size]

    def __getitem__(self, key):
        """Get item for given key."""
        val = self.file_content[key]
        if isinstance(val, netCDF4.Variable):
            return self._get_variable(key, val)
        if isinstance(val, netCDF4.Group):
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
        if self.manager is not None:
            val = self._get_var_from_manager(group, key)
        else:
            val = self._get_var_from_xr(group, key)
        return val

    def _get_group(self, key, val):
        """Get a group from the netcdf file."""
        # Full groups are conveniently read with xr even if file_handle is available
        with xr.open_dataset(self.filename, group=key,
                             **self._xarray_kwargs) as nc:
            val = nc
        return val

    def _get_var_from_xr(self, group, key):
        with xr.open_dataset(self.filename, group=group,
                             **self._xarray_kwargs) as nc:
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

    def _get_var_from_manager(self, group, key):
        # Not getting coordinates as this is more work, therefore more
        # overhead, and those are not used downstream.

        with self.manager.acquire_context() as ds:
            if group is not None:
                v = ds[group][key]
            else:
                v = ds[key]
        if group is None:
            dv = get_serializable_dask_array(
                    self.manager, key,
                    chunks=v.shape, dtype=v.dtype)
        else:
            dv = get_serializable_dask_array(
                    self.manager, "/".join([group, key]),
                    chunks=v.shape, dtype=v.dtype)
        attrs = self._get_object_attrs(v)
        x = xr.DataArray(
                dv,
                dims=v.dimensions, attrs=attrs, name=v.name)
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
        else:
            try:
                val = v[:]
                val = xr.DataArray(val, dims=v.dimensions, attrs=self._get_object_attrs(v), name=v.name)
            except IndexError:
                # Handle scalars
                val = v.__array__().item()
                val = xr.DataArray(val, dims=(), attrs={}, name=var_name)
            except AttributeError:
                # Handle strings
                val = v
        self.cached_file_content[var_name] = val
        return self.cached_file_content[var_name]


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


class NetCDF4FsspecFileHandler(NetCDF4FileHandler):
    """NetCDF4 file handler using fsspec to read files remotely."""

    def _get_file_handle(self):
        try:
            # Default to using NetCDF4 backend for local files
            return super()._get_file_handle()
        except OSError:
            # The netCDF4 lib raises either FileNotFoundError or OSError for remote files. OSError catches both.
            import h5netcdf
            f_obj = open_file_or_filename(self.filename)
            self._use_h5netcdf = True
            return h5netcdf.File(f_obj, "r")

    def __getitem__(self, key):
        """Get item for given key."""
        if self._use_h5netcdf:
            return self._getitem_h5netcdf(key)
        return super().__getitem__(key)

    def _getitem_h5netcdf(self, key):
        from h5netcdf import Group, Variable
        val = self.file_content[key]
        if isinstance(val, Variable):
            return self._get_variable(key, val)
        if isinstance(val, Group):
            return self._get_group(key, val)
        return val

    def _collect_cache_var_names(self, cache_var_size):
        if self._use_h5netcdf:
            return self._collect_cache_var_names_h5netcdf(cache_var_size)
        return super()._collect_cache_var_names(cache_var_size)

    def _collect_cache_var_names_h5netcdf(self, cache_var_size):
        from h5netcdf import Variable
        return [varname for (varname, var)
                in self.file_content.items()
                if isinstance(var, Variable)
                and isinstance(var.dtype, np.dtype)  # vlen may be str
                and np.prod(var.shape) * var.dtype.itemsize < cache_var_size]

    def _get_object_attrs(self, obj):
        if self._use_h5netcdf:
            return obj.attrs
        return super()._get_object_attrs(obj)

    def _get_attr(self, obj, key):
        if self._use_h5netcdf:
            return obj.attrs[key]
        return super()._get_attr(obj, key)


class _NCDatasetWrapper(netCDF4.Dataset):
    """Wrap netcdf4.Dataset setting auto_maskandscale globally.

    Helper class that wraps netcdf4.Dataset while setting extra parameters.
    By encapsulating this in a helper class, we can
    pass it to CachingFileManager directly.  Currently sets
    auto_maskandscale globally (for all variables).
    """

    def __init__(self, *args, auto_maskandscale=False, **kwargs):
        """Initialise object."""
        super().__init__(*args, **kwargs)
        self._set_extra_settings(auto_maskandscale=auto_maskandscale)

    def _set_extra_settings(self, auto_maskandscale):
        """Set our own custom settings.

        Currently only applies set_auto_maskandscale.
        """
        self.set_auto_maskandscale(auto_maskandscale)


class PreloadableSegments:
    """Mixin class for pre-loading segments.

    This is a mixin class designed to use with file handlers that support
    preloading segments.  Subclasses deriving from this mixin are expected
    to be geo-segmentable file handlers, and can be created based on
    a glob pattern for a non-existing file (as well as a path or glob
    pattern to an existing file).  The first segment of a repeat cycle
    is always processed only after the file exists.  For the remaining
    segments, metadata are collected as much as possible before the file
    exists, from two possible sources:

    - From the first segment.  For some file formats, many pieces of metadata
      can be expected to be identical between segments.
    - From the same segment number for a previous repeat cycle.  In this case,
      caching is done on disk.

    To implement a filehandler using this, make sure it derives from both
    :class:`NetCDF4FileHandler` and this class, and make sure to pass keyword
    arguments in ``__init__`` to the superclass.  In the YAML file,
    ``required_netcdf_variables`` must be defined as a dictionary.  The keys
    are variable names, and values are a list of strings describing what
    assumptions can be made about caching, where ``"rc"`` means the value is
    expected to be constant between repeat cycles, and ``"segment"`` means it is
    expected to be constant between the segments within a repeat cycle.  For
    variables that are actually variable, define the empty list.

    Attributes (variable, global, and group), shapes, dtypes, and dimension names
    are assumed to be always shareable between repeat cycles.

    To use preloading, set the satpy configuration variable
    ``readers.preload_segments`` to True.  The initialisation parameter
    for this filehandler might still be False, because the first segment (the
    reference segment) of a repeat cycle is loaded normally.

    This feature is experimental.

    .. versionadded:: 0.50
    """

    def __init__(self, *args, preload=False, ref_fh=None, rc_cache=None, **kwargs):
        """Store attributes needed for preloading to work."""
        self.preload = preload
        self.preload_tries = satpy.config.get("readers.preload_tries")
        self.preload_step = satpy.config.get("readers.preload_step")
        self.use_distributed = satpy.config.get("readers.preload_dask_distributed")
        if preload:
            if not isinstance(ref_fh, BaseFileHandler):
                raise TypeError(
                    "Expect reference filehandler when preloading, got "
                    f"{type(ref_fh)!s}")
            self.ref_fh = ref_fh
            if not isinstance(rc_cache, (str, bytes, os.PathLike)):
                raise TypeError(
                    "Expected cache file when preloading, got "
                    f"{type(rc_cache)!s}")
            self.rc_cache = rc_cache
        super().__init__(*args, **kwargs)
        if "required_netcdf_variables" not in self.filetype_info:
            raise ValueError("For preloadable filehandlers, "
                             "required_netcdf_variables is mandatory.")

    def _collect_listed_variables(self, file_handle, listed_variables):
        """Collect listed variables, either preloaded or regular."""
        if self.preload:
            self._preload_listed_variables(listed_variables)
        else:
            super()._collect_listed_variables(file_handle, listed_variables)

    def _preload_listed_variables(self, listed_variables):
        """Preload listed variables, either from RC cache or segment cache."""
        variable_name_replacements = self.filetype_info.get("variable_name_replacements")
        with open(self.rc_cache, "rb") as fp:
            self.file_content.update(pickle.load(fp))  # nosec
        for raw_name in listed_variables:
            for subst_name in self._get_required_variable_names(
                    [raw_name], variable_name_replacements):
                if self._can_get_from_other_segment(raw_name):
                    self.file_content[subst_name] = self.ref_fh[subst_name]
                elif not self._can_get_from_other_rc(raw_name):
                    self._collect_variable_delayed(subst_name)

    def _can_get_from_other_segment(self, itm):
        """Return true if variable is segment-cachable."""
        return "segment" in self.filetype_info["required_netcdf_variables"][itm]

    def _can_get_from_other_rc(self, itm):
        """Return true if variable is rc-cachable."""
        # it's always safe to store variable attributes, shapes, dtype, dimensions
        # between repeat cycles
        if self._is_safely_shareable_metadata(itm):
            return True
        # need an inverted mapping so I can tell which variables I can store
        invmap = self._get_inv_name_map()
        if "rc" in self.filetype_info["required_netcdf_variables"][invmap.get(itm, itm)]:
            return True
        return False

    def _is_safely_shareable_metadata(self, itm):
        """Check if item refers to safely shareable metadata."""
        for meta in ("/attr/", "/shape", "/dtype", "/dimension/", "/dimensions"):
            if meta in itm:
                return True

    def _get_inv_name_map(self):
        """Get inverted mapping for variable name replacements."""
        listed_variables = self.filetype_info.get("required_netcdf_variables")
        variable_name_replacements = self.filetype_info.get("variable_name_replacements")
        invmap = {}
        for raw_name in listed_variables:
            for subst_name in self._get_required_variable_names([raw_name],
                                                                variable_name_replacements):
                invmap[subst_name] = raw_name
        return invmap

    def _collect_variable_delayed(self, subst_name):
        md = self.ref_fh[subst_name]  # some metadata from reference segment
        fn_matched = _wait_for_file(self.filename, self.preload_tries,
                                    self.preload_step,
                                    self.use_distributed)
        dade = _get_delayed_value_from_nc(fn_matched, subst_name)
        array = da.from_delayed(
                dade,
                shape=self[subst_name + "/shape"],  # not safe from reference segment!
                dtype=md.dtype)

        var_obj = xr.DataArray(
                array,
                dims=md.dims,  # dimension /names/ should be safe
                attrs=md.attrs,  # FIXME: is this safe?  More involved to gather otherwise
                name=md.name)
        self.file_content[subst_name] = var_obj

    def _get_file_handle(self):
        if self.preload:
            return open(os.devnull, "r")
        return super()._get_file_handle()

    def store_cache(self, filename=None):
        """Store RC-cachable data to cache."""
        if self.preload:
            raise ValueError("Cannot store cache with pre-loaded handler")
        to_store = {}
        for key in self.file_content.keys():
            if self._can_get_from_other_rc(key):
                try:
                    to_store[key] = self[key].compute()
                except AttributeError:  # not a dask array
                    to_store[key] = self[key]

        filename = filename or self.rc_cache
        LOG.info(f"Storing cache to {filename!s}")
        with open(filename, "wb") as fp:
            # FIXME: potential security risk?  Acceptable?
            pickle.dump(to_store, fp)


@dask.delayed
def _wait_for_file(pat, max_tries=300, wait=2, use_distributed=False):
    """Wait for file to appear."""
    for i in range(max_tries):
        if (match := _check_for_matching_file(i, pat, wait, use_distributed)):
            if i > 0:  # only log if not found immediately
                LOG.debug(f"Found {match!s} matching {pat!s}!")
            return match
    else:
        raise TimeoutError(f"File matching {pat!s} failed to materialise")


def _check_for_matching_file(i, pat, wait, use_distributed=False):
    fns = glob.glob(pat)
    if len(fns) == 0:
        _log_and_wait(i, pat, wait, use_distributed)
        return
    if len(fns) > 1:
        raise ValueError(f"Expected one matching file, found {len(fns):d}")
    return fns[0]


def _log_and_wait(i, pat, wait, use_distributed=False):
    """Maybe log that we're waiting for pat, then wait."""
    if i == 0:  # only log if not yet present
        LOG.debug(f"Waiting for {pat!s} to appear.")
    if i % 60 == 30:
        LOG.debug(f"Still waiting for {pat!s}")
    if use_distributed:
        dask.distributed.secede()
    time.sleep(wait)
    if use_distributed:
        dask.distributed.rejoin()


@dask.delayed
def _get_delayed_value_from_nc(fn, var, auto_maskandscale=False):
    if "/" in var:
        (grp, var) = var.rsplit("/", maxsplit=1)
    else:
        (grp, var) = (None, var)
    with xr.open_dataset(fn, group=grp, mask_and_scale=auto_maskandscale, engine="h5netcdf") as nc:
        return nc[var][:]
