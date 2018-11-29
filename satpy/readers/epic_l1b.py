#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017.

# Author(s):

#   edited from original autors 
#   inspired in maia.py and iasi_l2.py readers

# This file is part of satpy.

# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Reader for EPIC instrument from DSCOVR

"""
import logging
from datetime import datetime

import h5py
import numpy as np
from xarray import DataArray
import dask.array as da

from satpy.readers.file_handlers import BaseFileHandler
from satpy import CHUNK_SIZE

logger = logging.getLogger(__name__)

DSET_NAMES = {'B01':'Band317nm',
              'B02':'Band325nm',
              'B03':'Band340nm',
              'B04':'Band388nm',
              'B05':'Band443nm',
              'B06':'Band551nm',
              'B07':'Band680nm',
              'B08':'Band688nm',
              'B09':'Band764nm',
              'B10':'Band780nm',}

GEO_NAMES = {'latitude':'Latitude',
             'longitude':'Longitude',
             'satellite_azimuth_angle':'ViewAngleAzimuth',
             'satellite_zenith_angle':'ViewAngleZenith',
             'solar_azimuth_angle':'SunAngleAzimuth',
             'solar_zenith_angle':'SunAngleZenith'
             }


class EPIC_L1B(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(EPIC_L1B, self).__init__(filename, filename_info, filetype_info)
        
        self.h5 = h5py.File(self.filename, 'r')
        
        self.finfo = filename_info
        self.selected = None
        
        self.finfo = filename_info
        self.lons = None
        self.lats = None
        self.sensor = self.h5.attrs['keywords'].split(',')[1].strip().lower()
        
        self.ds = {}
        self.ds['platform_name'] = self.h5.attrs['keywords'].split(',')[0]
        self.ds['sensor'] = self.sensor

    @property
    def start_time(self):
        dtstr = self.h5.attrs['begin_time']
        return datetime.strptime(dtstr, '%Y-%m-%d %H:%M:%S')

    @property
    def end_time(self):
        dtstr = self.h5.attrs['end_time']
        return datetime.strptime(dtstr, '%Y-%m-%d %H:%M:%S')

    def get_dataset(self, key, info):
        """Load a dataset"""
        fid = self.h5
        logger.debug('Reading %s.', key.name)
        if key.name in DSET_NAMES.keys():
            ds = read_dataset(fid, key)
        else:
            ds = read_geo(fid, key)

        ds.attrs.update(info)
        
        ds.attrs.update({'platform_name': self.h5.attrs['keywords'].split(',')[0],
                  'sensor': self.h5.attrs['keywords'].split(',')[1].strip().lower(),
                  'satellite_latitude': float(self.h5.attrs['centroid_mean_latitude']),
                  'satellite_longitude': float(self.h5.attrs['centroid_mean_longitude']),
                  'satellite_altitude': float(1500000)}) #aprox no real

        return ds


def read_dataset(fid, key):
    """Read dataset"""

    dsid = DSET_NAMES[key.name]

    dset = fid[dsid]['Image']
    values = da.from_array(dset, chunks=CHUNK_SIZE)
    values /= 1000.
    
    fill_value = dset.attrs['_FillValue'][0]

    data = DataArray(values, name=key.name, dims=['y', 'x']).astype(np.float32)
    data = data.where(data != fill_value)
    dset_attrs = dict(dset.attrs)
    data.attrs.update(dset_attrs)

    return data


def read_geo(fid, key):
    """Read geolocation and related datasets."""

    dsig = GEO_NAMES[key.name]
    
    gset = fid["Band317nm"]['Geolocation']['Earth'][dsig]
    values = da.from_array(gset, chunks=CHUNK_SIZE)
    
    fill_value = gset.attrs['_FillValue'][0]
    
    data = DataArray(values, dims=['y', 'x']).astype(np.float32)
    data = data.where(data != fill_value)
    
    return data
