#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016-2020 Satpy developers
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

import netCDF4
import logging
import numpy as np
import xarray as xr
import dask.array as da

from satpy import CHUNK_SIZE
from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.utils import np2str

LOG = logging.getLogger(__name__)


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

    """

    file_handle = None

    def __init__(self, filename, filename_info, filetype_info,
                 auto_maskandscale=False, xarray_kwargs=None,
                 cache_var_size=0, cache_handle=False):
        """Initialize object."""
        super(NetCDF4FileHandler, self).__init__(
            filename, filename_info, filetype_info)
        self.file_content = {}
        self.cached_file_content = {}
        try:
            file_handle = netCDF4.Dataset(self.filename, 'r')
        except IOError:
            LOG.exception(
                'Failed reading file %s. Possibly corrupted file', self.filename)
            raise

        self.auto_maskandscale = auto_maskandscale
        if hasattr(file_handle, "set_auto_maskandscale"):
            file_handle.set_auto_maskandscale(auto_maskandscale)

        self.collect_metadata("", file_handle)
        self.collect_dimensions("", file_handle)
        if cache_var_size > 0:
            self.collect_cache_vars(
                    [varname for (varname, var)
                        in self.file_content.items()
                        if isinstance(var, netCDF4.Variable)
                        and isinstance(var.dtype, np.dtype)  # vlen may be str
                        and var.size * var.dtype.itemsize < cache_var_size],
                    file_handle)
        if cache_handle:
            self.file_handle = file_handle
        else:
            file_handle.close()
        self._xarray_kwargs = xarray_kwargs or {}
        self._xarray_kwargs.setdefault('chunks', CHUNK_SIZE)
        self._xarray_kwargs.setdefault('mask_and_scale', self.auto_maskandscale)

    def __del__(self):
        """Delete object."""
        if self.file_handle is not None:
            try:
                self.file_handle.close()
            except RuntimeError:  # presumably closed already
                pass

    def _collect_attrs(self, name, obj):
        """Collect all the attributes for the provided file object."""
        for key in obj.ncattrs():
            value = getattr(obj, key)
            fc_key = "{}/attr/{}".format(name, key)
            try:
                self.file_content[fc_key] = np2str(value)
            except ValueError:
                self.file_content[fc_key] = value

    def collect_metadata(self, name, obj):
        """Collect all file variables and attributes for the provided file object.

        This method also iterates through subgroups of the provided object.
        """
        # Look through each subgroup
        base_name = name + "/" if name else ""
        for group_name, group_obj in obj.groups.items():
            full_group_name = base_name + group_name
            self.file_content[full_group_name] = group_obj
            self._collect_attrs(full_group_name, group_obj)
            self.collect_metadata(full_group_name, group_obj)
        for var_name, var_obj in obj.variables.items():
            var_name = base_name + var_name
            self.file_content[var_name] = var_obj
            self.file_content[var_name + "/dtype"] = var_obj.dtype
            self.file_content[var_name + "/shape"] = var_obj.shape
            self.file_content[var_name + "/dimensions"] = var_obj.dimensions
            self._collect_attrs(var_name, var_obj)
        self._collect_attrs(name, obj)

    def collect_dimensions(self, name, obj):
        """Collect dimensions."""
        for dim_name, dim_obj in obj.dimensions.items():
            dim_name = "{}/dimension/{}".format(name, dim_name)
            self.file_content[dim_name] = len(dim_obj)

    def collect_cache_vars(self, cache_vars, obj):
        """Collect data variables for caching.

        This method will collect some data variables and store them in RAM.
        This may be useful if some small variables are frequently accessed,
        to prevent needlessly frequently opening and closing the file, which
        in case of xarray is associated with some overhead.

        Should be called later than `collect_metadata`.

        Args:
            cache_vars (List[str]): Names of data variables to be cached.
            obj (netCDF4.Dataset): Dataset object from which to read them.

        """
        for var_name in cache_vars:
            v = self.file_content[var_name]
            self.cached_file_content[var_name] = xr.DataArray(
                    v[:], dims=v.dimensions, attrs=v.__dict__, name=v.name)

    def __getitem__(self, key):
        """Get item for given key."""
        val = self.file_content[key]
        if isinstance(val, netCDF4.Variable):
            if key in self.cached_file_content:
                return self.cached_file_content[key]
            # these datasets are closed and inaccessible when the file is
            # closed, need to reopen
            # TODO: Handle HDF4 versus NetCDF3 versus NetCDF4
            parts = key.rsplit('/', 1)
            if len(parts) == 2:
                group, key = parts
            else:
                group = None
            if self.file_handle is not None:
                val = self._get_var_from_filehandle(group, key)
            else:
                val = self._get_var_from_xr(group, key)
        elif isinstance(val, netCDF4.Group):
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

    def _get_var_from_filehandle(self, group, key):
        # Not getting coordinates as this is more work, therefore more
        # overhead, and those are not used downstream.
        if group is None:
            g = self.file_handle
        else:
            g = self.file_handle[group]
        v = g[key]
        x = xr.DataArray(
                da.from_array(v), dims=v.dimensions, attrs=v.__dict__,
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
