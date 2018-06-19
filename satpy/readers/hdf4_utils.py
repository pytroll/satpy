#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017.
#
# Author(s):
#
#
#   David Hoese <david.hoese@ssec.wisc.edu>
#
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
"""Helpers for reading hdf4-based files.

"""
import logging

from pyhdf.SD import SD, SDC, SDS
import dask.array as da
import xarray as xr
import numpy as np
import six

from satpy import CHUNK_SIZE
from satpy.readers.file_handlers import BaseFileHandler

LOG = logging.getLogger(__name__)


HTYPE_TO_DTYPE = {
    SDC.INT8: np.int8,
    SDC.UCHAR: np.uint8,
    SDC.CHAR: np.int8,
    SDC.INT32: np.int32,
    SDC.INT16: np.int16,
    SDC.UINT8: np.uint8,
    SDC.UINT16: np.uint16,
    SDC.UINT32: np.uint32,
    SDC.FLOAT32: np.float32,
    SDC.FLOAT64: np.float64,
}


def from_sds(var, *args, **kwargs):
    """Create a dask array from a SD dataset."""
    var.__dict__['dtype'] = HTYPE_TO_DTYPE[var.info()[3]]
    shape = var.info()[2]
    var.__dict__['shape'] = shape if isinstance(shape, (tuple, list)) else tuple(shape)
    return da.from_array(var, *args, **kwargs)


class HDF4FileHandler(BaseFileHandler):
    """Small class for inspecting a HDF5 file and retrieve its metadata/header data.
    """

    def __init__(self, filename, filename_info, filetype_info):
        super(HDF4FileHandler, self).__init__(filename, filename_info, filetype_info)
        self.file_content = {}
        file_handle = SD(self.filename, SDC.READ)
        self._collect_attrs('', file_handle.attributes())
        for k, v in file_handle.datasets().items():
            self.collect_metadata(k, file_handle.select(k))
        del file_handle

    def _collect_attrs(self, name, attrs):
        for key, value in six.iteritems(attrs):
            value = np.squeeze(value)
            if issubclass(value.dtype.type, np.string_) and not value.shape:
                value = np.asscalar(value)
                if not isinstance(value, str):
                    # python 3 - was scalar numpy array of bytes
                    # otherwise python 2 - scalar numpy array of 'str'
                    value = value.decode()
                self.file_content["{}/attr/{}".format(name, key)] = value
            else:
                self.file_content["{}/attr/{}".format(name, key)] = value

    def collect_metadata(self, name, obj):
        if isinstance(obj, SDS):
            self.file_content[name] = obj
            info = obj.info()
            self.file_content[name + "/dtype"] = HTYPE_TO_DTYPE.get(info[3])
            self.file_content[name + "/shape"] = info[2] if isinstance(info[2], (int, float)) else tuple(info[2])

    def _open_xarray_dataset(self, val, chunks=CHUNK_SIZE):
        """Read the band in blocks."""
        dask_arr = from_sds(val, chunks=chunks)
        attrs = val.attributes()
        return xr.DataArray(dask_arr, dims=('y', 'x'),
                            attrs=attrs)

    def __getitem__(self, key):
        val = self.file_content[key]
        if isinstance(val, SDS):
            # these datasets are closed and inaccessible when the file is closed, need to reopen
            return self._open_xarray_dataset(val)
        return val

    def __contains__(self, item):
        return item in self.file_content

    def get(self, item, default=None):
        if item in self:
            return self[item]
        else:
            return default
