#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016-2017.

# Author(s):

#
#   David Hoese <david.hoese@ssec.wisc.edu>
#

# This file is part of satpy.

# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.

"""Helpers for reading hdf5-based files.

"""
import logging
import h5py
import numpy as np
import six
import xarray as xr
import dask.array as da

from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.utils import np2str
from satpy import CHUNK_SIZE

LOG = logging.getLogger(__name__)


class HDF5FileHandler(BaseFileHandler):
    """Small class for inspecting a HDF5 file and retrieve its metadata/header data.
    """

    def __init__(self, filename, filename_info, filetype_info):
        super(HDF5FileHandler, self).__init__(
            filename, filename_info, filetype_info)
        self.file_content = {}
        try:
            file_handle = h5py.File(self.filename, 'r')
        except IOError:
            LOG.exception(
                'Failed reading file %s. Possibly corrupted file', self.filename)
            raise

        file_handle.visititems(self.collect_metadata)
        self._collect_attrs('', file_handle.attrs)
        file_handle.close()

    def _collect_attrs(self, name, attrs):
        for key, value in six.iteritems(attrs):
            value = np.squeeze(value)
            fc_key = "{}/attr/{}".format(name, key)
            try:
                self.file_content[fc_key] = np2str(value)
            except ValueError:
                self.file_content[fc_key] = value

    def collect_metadata(self, name, obj):
        if isinstance(obj, h5py.Dataset):
            self.file_content[name] = obj
            self.file_content[name + "/dtype"] = obj.dtype
            self.file_content[name + "/shape"] = obj.shape
        self._collect_attrs(name, obj.attrs)

    def __getitem__(self, key):
        val = self.file_content[key]
        if isinstance(val, h5py.Dataset):
            # these datasets are closed and inaccessible when the file is closed, need to reopen
            dset = h5py.File(self.filename, 'r')[key]
            dset_data = da.from_array(dset, chunks=CHUNK_SIZE)
            if dset.ndim == 2:
                return xr.DataArray(dset_data, dims=['y', 'x'], attrs=dset.attrs)
            return xr.DataArray(dset_data)

        return val

    def __contains__(self, item):
        return item in self.file_content

    def get(self, item, default=None):
        if item in self:
            return self[item]
        else:
            return default
