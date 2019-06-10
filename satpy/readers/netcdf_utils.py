#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016-2017 Satpy developers
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
"""Helpers for reading netcdf-based files.

"""
import netCDF4
import logging
import xarray as xr

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

    Note that loading datasets requires reopening the original file, but to
    get just the shape of the dataset append "/shape" to the item string:

        wrapper["group/subgroup/var_name/shape"]

    """

    def __init__(self, filename, filename_info, filetype_info,
                 auto_maskandscale=False, xarray_kwargs=None):
        super(NetCDF4FileHandler, self).__init__(
            filename, filename_info, filetype_info)
        self.file_content = {}
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
        file_handle.close()
        self._xarray_kwargs = xarray_kwargs or {}
        self._xarray_kwargs.setdefault('chunks', CHUNK_SIZE)
        self._xarray_kwargs.setdefault('mask_and_scale', self.auto_maskandscale)

    def _collect_attrs(self, name, obj):
        """Collect all the attributes for the provided file object.
        """
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
            self.collect_metadata(base_name + group_name, group_obj)
        for var_name, var_obj in obj.variables.items():
            var_name = base_name + var_name
            self.file_content[var_name] = var_obj
            self.file_content[var_name + "/dtype"] = var_obj.dtype
            self.file_content[var_name + "/shape"] = var_obj.shape
            self._collect_attrs(var_name, var_obj)
        self._collect_attrs(name, obj)

    def collect_dimensions(self, name, obj):
        for dim_name, dim_obj in obj.dimensions.items():
            dim_name = "{}/dimension/{}".format(name, dim_name)
            self.file_content[dim_name] = len(dim_obj)

    def __getitem__(self, key):
        val = self.file_content[key]
        if isinstance(val, netCDF4.Variable):
            # these datasets are closed and inaccessible when the file is
            # closed, need to reopen
            # TODO: Handle HDF4 versus NetCDF3 versus NetCDF4
            parts = key.rsplit('/', 1)
            if len(parts) == 2:
                group, key = parts
            else:
                group = None
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

    def __contains__(self, item):
        return item in self.file_content

    def get(self, item, default=None):
        if item in self:
            return self[item]
        else:
            return default
