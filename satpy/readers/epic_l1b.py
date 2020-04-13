#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 Satpy developers
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
"""EPIC L1B hdf5 file reader.
Format: https://eosweb.larc.nasa.gov/project/dscovr/EPIC_Data_Format_Control_Book_2016-07-01.pdf

calibration factors from counts/send to reflectance:
https://eosweb.larc.nasa.gov/project/dscovr/DSCOVR_EPIC_Calibration_Factors_V02.pdf
"""

import h5py
import numpy as np
import xarray as xr
import dask.array as da
import datetime as dt
import logging

from satpy.readers.hdf5_utils import HDF5FileHandler
from satpy import CHUNK_SIZE

LOGGER = logging.getLogger(__name__)

DSET_NAMES = {'Band317nm': 'Band'}

GEO_NAMES = {'latitude': 'Latitude',
             'longitude': 'Longitude'}

REFLECTANCE_CALIBRATION_FACTORS_V02 = {'Band317nm': 1.216E-04,
                                       'Band325nm': 1.111E-04,
                                       'Band340nm': 1.975E-05,
                                       'Band388nm': 2.685E-05,
                                       'Band443nm': 8.340E-06,
                                       'Band551nm': 6.660E-06,
                                       'Band680nm': 9.300E-06,
                                       'Band688nm': 2.020E-05,
                                       'Band764nm': 2.360E-05,
                                       'Band780nm': 1.435E-05}


class EPICL1BReader(HDF5FileHandler):

    """File handler for EPIC L1B HDF5 files."""

    def __init__(self, filename, filename_info, filetype_info, **kwargs):
        super(EPICL1BReader, self).__init__(filename, filename_info, filetype_info)

        self.finfo = filename_info
        self.lons = None
        self.lats = None
        self.sensor = filename_info['sensor']

        self.mda = {}
        self.mda['platform_name'] = 'DISCOVR'
        self.mda['sensor'] = self.sensor

    @property
    def start_time(self):
        return self.finfo['start_time']

    def get_dataset(self, key, info):
        """Load a dataset"""
        with h5py.File(self.filename, 'r') as fid:
            LOGGER.debug('Reading %s.', key.name)
            if 'Band' in key.name:  # in DSET_NAMES:
                m_data = read_dataset(fid, key)
            else:
                m_data = read_geo(fid, key)
        m_data.attrs.update(info)
        m_data.attrs['sensor'] = self.sensor

        return m_data


def read_dataset(fid, key):
    """Read dataset"""
    dset = fid["/" + key.name + "/Image"]
    dims = ['y', 'x']
    data = xr.DataArray(da.from_array(dset.value, chunks=CHUNK_SIZE),
                        name=key.name, dims=dims).astype(np.float32)
    mask = xr.DataArray(da.from_array(fid["/" + key.name + "/Geolocation/Earth/Mask"].value,
                                      chunks=CHUNK_SIZE),
                        dims=dims).astype(np.byte)
    data = xr.where(mask == 0, np.nan, data)
    data = xr.where(data == 0, np.nan, data)
    data *= REFLECTANCE_CALIBRATION_FACTORS_V02[key.name]
    data *= 100.

    dset_attrs = dict(dset.attrs)
    data.attrs.update(dset_attrs)

    return data


def read_geo(fid, key):
    """Read geolocation and related datasets."""
    dsid = GEO_NAMES[key.name]
    data = fid["/Band317nm/Geolocation/Earth/" + dsid].value
    dtype = np.float32
    data = xr.DataArray(da.from_array(data, chunks=CHUNK_SIZE),
                        name=key.name, dims=['y', 'x']).astype(dtype)

    return data
