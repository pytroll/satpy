#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Martin Raspaud

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Sentinel-3 OLCI reader
"""

import logging
from datetime import datetime

import dask.array as da
import numpy as np
import xarray as xr

from satpy.readers.file_handlers import BaseFileHandler
from satpy.utils import angle2xyz, xyz2angle
from satpy import CHUNK_SIZE
from functools import reduce

logger = logging.getLogger(__name__)

PLATFORM_NAMES = {'S3A': 'Sentinel-3A',
                  'S3B': 'Sentinel-3B'}


class BitFlags(object):

    """Manipulate flags stored bitwise.
    """
    flag_list = ['INVALID', 'WATER', 'LAND', 'CLOUD', 'SNOW_ICE',
                 'INLAND_WATER', 'TIDAL', 'COSMETIC', 'SUSPECT',
                 'HISOLZEN', 'SATURATED', 'MEGLINT', 'HIGHGLINT',
                 'WHITECAPS', 'ADJAC', 'WV_FAIL', 'PAR_FAIL',
                 'AC_FAIL', 'OC4ME_FAIL', 'OCNN_FAIL',
                 'Extra_1',
                 'KDM_FAIL',
                 'Extra_2',
                 'CLOUD_AMBIGUOUS', 'CLOUD_MARGIN', 'BPAC_ON', 'WHITE_SCATT',
                 'LOWRW', 'HIGHRW']

    meaning = {f: i for i, f in enumerate(flag_list)}

    def __init__(self, value):

        self._value = value

    def __getitem__(self, item):
        pos = self.meaning[item]
        return ((self._value >> pos) % 2).astype(np.bool)


class NCOLCIBase(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(NCOLCIBase, self).__init__(filename, filename_info,
                                         filetype_info)
        self.nc = xr.open_dataset(self.filename,
                                  decode_cf=True,
                                  mask_and_scale=True,
                                  engine='h5netcdf',
                                  chunks={'columns': CHUNK_SIZE,
                                          'rows': CHUNK_SIZE})

        self.nc = self.nc.rename({'columns': 'x', 'rows': 'y'})

        # TODO: get metadata from the manifest file (xfdumanifest.xml)
        self.platform_name = PLATFORM_NAMES[filename_info['mission_id']]
        self.sensor = 'olci'

    @property
    def start_time(self):
        return datetime.strptime(self.nc.attrs['start_time'],
                                 '%Y-%m-%dT%H:%M:%S.%fZ')

    @property
    def end_time(self):
        return datetime.strptime(self.nc.attrs['stop_time'],
                                 '%Y-%m-%dT%H:%M:%S.%fZ')

    def get_dataset(self, key, info):
        """Load a dataset."""
        logger.debug('Reading %s.', key.name)
        variable = self.nc[key.name]

        return variable


class NCOLCICal(NCOLCIBase):
    pass


class NCOLCIGeo(NCOLCIBase):
    pass


class NCOLCIChannelBase(NCOLCIBase):

    def __init__(self, filename, filename_info, filetype_info):
        super(NCOLCIChannelBase, self).__init__(filename, filename_info,
                                                filetype_info)

        self.channel = filename_info.get('dataset_name')


class NCOLCI1B(NCOLCIChannelBase):

    def __init__(self, filename, filename_info, filetype_info, cal):
        super(NCOLCI1B, self).__init__(filename, filename_info,
                                       filetype_info)
        self.cal = cal.nc

    def _get_solar_flux_old(self, band):
        # TODO: this could be replaced with vectorized indexing in the future.
        from dask.base import tokenize
        blocksize = CHUNK_SIZE

        solar_flux = self.cal['solar_flux'].isel(bands=band).values
        d_index = self.cal['detector_index'].fillna(0).astype(int)

        shape = d_index.shape
        vchunks = range(0, shape[0], blocksize)
        hchunks = range(0, shape[1], blocksize)

        token = tokenize(band, d_index, solar_flux)
        name = 'solar_flux_' + token

        def get_items(array, slices):
            return solar_flux[d_index[slices].values]

        dsk = {(name, i, j): (get_items,
                              d_index,
                              (slice(vcs, min(vcs + blocksize, shape[0])),
                               slice(hcs, min(hcs + blocksize, shape[1]))))
               for i, vcs in enumerate(vchunks)
               for j, hcs in enumerate(hchunks)
               }

        res = da.Array(dsk, name, shape=shape,
                       chunks=(blocksize, blocksize),
                       dtype=solar_flux.dtype)
        return res

    @staticmethod
    def _get_items(idx, solar_flux):
        return solar_flux[idx]

    def _get_solar_flux(self, band):
        """Get the solar flux for the band."""
        solar_flux = self.cal['solar_flux'].isel(bands=band).values
        d_index = self.cal['detector_index'].fillna(0).astype(int)

        return da.map_blocks(self._get_items, d_index.data, solar_flux=solar_flux, dtype=solar_flux.dtype)

    def get_dataset(self, key, info):
        """Load a dataset."""
        if self.channel != key.name:
            return
        logger.debug('Reading %s.', key.name)

        radiances = self.nc[self.channel + '_radiance']

        if key.calibration == 'reflectance':
            idx = int(key.name[2:]) - 1
            sflux = self._get_solar_flux(idx)
            radiances = radiances / sflux * np.pi * 100
            radiances.attrs['units'] = '%'

        radiances.attrs['platform_name'] = self.platform_name
        radiances.attrs['sensor'] = self.sensor
        radiances.attrs.update(key.to_dict())
        return radiances


class NCOLCI2(NCOLCIChannelBase):

    def get_dataset(self, key, info):
        """Load a dataset
        """
        if self.channel is not None and self.channel != key.name:
            return
        logger.debug('Reading %s.', key.name)

        if self.channel is not None and self.channel.startswith('Oa'):
            dataset = self.nc[self.channel + '_reflectance']
        else:
            dataset = self.nc[info['nc_key']]

        if key.name == 'wqsf':
            dataset.attrs['_FillValue'] = 1
        elif key.name == 'mask':
            mask = self.getbitmask(dataset.to_masked_array().data)
            dataset = dataset * np.nan
            dataset = dataset.where(~ mask, True)

        dataset.attrs['platform_name'] = self.platform_name
        dataset.attrs['sensor'] = self.sensor
        dataset.attrs.update(key.to_dict())
        return dataset

    def getbitmask(self, wqsf, items=[]):
        """ """
        items = ["INVALID", "SNOW_ICE", "INLAND_WATER", "SUSPECT",
                 "AC_FAIL", "CLOUD", "HISOLZEN", "OCNN_FAIL",
                 "CLOUD_MARGIN", "CLOUD_AMBIGUOUS", "LOWRW", "LAND"]
        bflags = BitFlags(wqsf)
        return reduce(np.logical_or, [bflags[item] for item in items])


class NCOLCIAngles(BaseFileHandler):

    datasets = {'satellite_azimuth_angle': 'OAA',
                'satellite_zenith_angle': 'OZA',
                'solar_azimuth_angle': 'SAA',
                'solar_zenith_angle': 'SZA'}

    def __init__(self, filename, filename_info, filetype_info):
        super(NCOLCIAngles, self).__init__(filename, filename_info,
                                           filetype_info)
        self.nc = None
        # TODO: get metadata from the manifest file (xfdumanifest.xml)
        self.platform_name = PLATFORM_NAMES[filename_info['mission_id']]
        self.sensor = 'olci'
        self.cache = {}
        self._start_time = filename_info['start_time']
        self._end_time = filename_info['end_time']

    def get_dataset(self, key, info):
        """Load a dataset."""
        if key.name not in self.datasets:
            return

        if self.nc is None:
            self.nc = xr.open_dataset(self.filename,
                                      decode_cf=True,
                                      mask_and_scale=True,
                                      engine='h5netcdf',
                                      chunks={'tie_columns': CHUNK_SIZE,
                                              'tie_rows': CHUNK_SIZE})

            self.nc = self.nc.rename({'tie_columns': 'x', 'tie_rows': 'y'})
        logger.debug('Reading %s.', key.name)

        l_step = self.nc.attrs['al_subsampling_factor']
        c_step = self.nc.attrs['ac_subsampling_factor']

        if (c_step != 1 or l_step != 1) and self.cache.get(key.name) is None:

            if key.name.startswith('satellite'):
                zen = self.nc[self.datasets['satellite_zenith_angle']]
                zattrs = zen.attrs
                azi = self.nc[self.datasets['satellite_azimuth_angle']]
                aattrs = azi.attrs
            elif key.name.startswith('solar'):
                zen = self.nc[self.datasets['solar_zenith_angle']]
                zattrs = zen.attrs
                azi = self.nc[self.datasets['solar_azimuth_angle']]
                aattrs = azi.attrs
            else:
                raise NotImplementedError("Don't know how to read " + key.name)

            x, y, z = angle2xyz(azi, zen)
            shape = x.shape

            from geotiepoints.interpolator import Interpolator
            tie_lines = np.arange(
                0, (shape[0] - 1) * l_step + 1, l_step)
            tie_cols = np.arange(0, (shape[1] - 1) * c_step + 1, c_step)
            lines = np.arange((shape[0] - 1) * l_step + 1)
            cols = np.arange((shape[1] - 1) * c_step + 1)
            along_track_order = 1
            cross_track_order = 3
            satint = Interpolator([x.values, y.values, z.values],
                                  (tie_lines, tie_cols),
                                  (lines, cols),
                                  along_track_order,
                                  cross_track_order)
            (x, y, z, ) = satint.interpolate()
            del satint
            x = xr.DataArray(da.from_array(x, chunks=(CHUNK_SIZE, CHUNK_SIZE)),
                             dims=['y', 'x'])
            y = xr.DataArray(da.from_array(y, chunks=(CHUNK_SIZE, CHUNK_SIZE)),
                             dims=['y', 'x'])
            z = xr.DataArray(da.from_array(z, chunks=(CHUNK_SIZE, CHUNK_SIZE)),
                             dims=['y', 'x'])

            azi, zen = xyz2angle(x, y, z)
            azi.attrs = aattrs
            zen.attrs = zattrs

            if 'zenith' in key.name:
                values = zen
            elif 'azimuth' in key.name:
                values = azi
            else:
                raise NotImplementedError("Don't know how to read " + key.name)

            if key.name.startswith('satellite'):
                self.cache['satellite_zenith_angle'] = zen
                self.cache['satellite_azimuth_angle'] = azi
            elif key.name.startswith('solar'):
                self.cache['solar_zenith_angle'] = zen
                self.cache['solar_azimuth_angle'] = azi

        elif key.name in self.cache:
            values = self.cache[key.name]
        else:
            values = self.nc[self.datasets[key.name]]

        values.attrs['platform_name'] = self.platform_name
        values.attrs['sensor'] = self.sensor

        values.attrs.update(key.to_dict())
        return values

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time
