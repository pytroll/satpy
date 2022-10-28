#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016-2017, 2019 Satpy developers
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
"""Helpers for reading hdf5-based files."""

import logging

import dask.array as da
import h5py
import numpy as np
import xarray as xr

from satpy import CHUNK_SIZE
from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.utils import np2str

LOG = logging.getLogger(__name__)


class HDF5FileHandler(BaseFileHandler):
    """Small class for inspecting a HDF5 file and retrieve its metadata/header data."""

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize file handler."""
        super(HDF5FileHandler, self).__init__(
            filename, filename_info, filetype_info)
        self.file_content = {}
        self._attrs_cache = {}

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
        attrs_cache = self._attrs_cache.setdefault(name, {})
        for key, value in attrs.items():
            value = np.squeeze(value)
            fc_key = "{}/attr/{}".format(name, key)
            try:
                value = np2str(value)
            except ValueError:
                # use the original value
                pass
            except AttributeError:
                # A HDF5 reference ?
                value = self.get_reference(name, key)
                if value is None:
                    LOG.warning("Value cannot be converted - skip setting attribute %s", fc_key)
                    continue
            self.file_content[fc_key] = attrs_cache[key] = value

    def get_reference(self, name, key):
        """Get reference."""
        with h5py.File(self.filename, 'r') as hf:
            return self._get_reference(hf, hf[name].attrs[key])

    def _get_reference(self, hf, ref):
        try:
            return [self._get_reference(hf, elt) for elt in ref]
        except TypeError:
            if isinstance(ref, h5py.h5r.Reference):
                ref_name = h5py.h5r.get_name(ref, hf.id)
                return hf[ref_name][()]

    def collect_metadata(self, name, obj):
        """Collect metadata."""
        if isinstance(obj, h5py.Dataset):
            self.file_content[name] = obj
            self.file_content[name + "/dtype"] = obj.dtype
            self.file_content[name + "/shape"] = obj.shape
        self._collect_attrs(name, obj.attrs)

    def __getitem__(self, key):
        """Get item for given key."""
        val = self.file_content[key]
        if isinstance(val, h5py.Dataset):
            # these datasets are closed and inaccessible when the file is closed, need to reopen
            dset = h5py.File(self.filename, 'r')[key]
            dset_data = da.from_array(dset, chunks=CHUNK_SIZE)
            attrs = self._attrs_cache.get(key, dset.attrs)
            if dset.ndim == 2:
                return xr.DataArray(dset_data, dims=['y', 'x'], attrs=attrs)
            return xr.DataArray(dset_data, attrs=attrs)

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
