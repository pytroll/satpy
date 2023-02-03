#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016 Satpy developers
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
"""Sentinel-3 OLCI reader.

This reader supports an optional argument to choose the 'engine' for reading
OLCI netCDF4 files. By default, this reader uses the default xarray choice of
engine, as defined in the :func:`xarray.open_dataset` documentation`.

As an alternative, the user may wish to use the 'h5netcdf' engine, but that is
not default as it typically prints many non-fatal but confusing error messages
to the terminal.
To choose between engines the user can  do as follows for the default::

    scn = Scene(filenames=my_files, reader='olci_l1b')

or as follows for the h5netcdf engine::

    scn = Scene(filenames=my_files,
                reader='olci_l1b', reader_kwargs={'engine': 'h5netcdf'})

References:
    - :func:`xarray.open_dataset`

"""


import logging
from functools import reduce

import dask.array as da
import numpy as np
import xarray as xr

from satpy import CHUNK_SIZE
from satpy._compat import cached_property
from satpy.readers import open_file_or_filename
from satpy.readers.file_handlers import BaseFileHandler
from satpy.utils import angle2xyz, xyz2angle

logger = logging.getLogger(__name__)

PLATFORM_NAMES = {'S3A': 'Sentinel-3A',
                  'S3B': 'Sentinel-3B',
                  'ENV': 'Environmental Satellite'}


class BitFlags:
    """Manipulate flags stored bitwise."""

    def __init__(self, value, flag_list=None):
        """Init the flags."""
        self._value = value
        flag_list = flag_list or ['INVALID', 'WATER', 'LAND', 'CLOUD', 'SNOW_ICE',
                                  'INLAND_WATER', 'TIDAL', 'COSMETIC', 'SUSPECT',
                                  'HISOLZEN', 'SATURATED', 'MEGLINT', 'HIGHGLINT',
                                  'WHITECAPS', 'ADJAC', 'WV_FAIL', 'PAR_FAIL',
                                  'AC_FAIL', 'OC4ME_FAIL', 'OCNN_FAIL',
                                  'Extra_1',
                                  'KDM_FAIL',
                                  'Extra_2',
                                  'CLOUD_AMBIGUOUS', 'CLOUD_MARGIN', 'BPAC_ON', 'WHITE_SCATT',
                                  'LOWRW', 'HIGHRW']
        self.meaning = {f: i for i, f in enumerate(flag_list)}

    def __getitem__(self, item):
        """Get the item."""
        pos = self.meaning[item]
        data = self._value
        if isinstance(data, xr.DataArray):
            data = data.data
            res = ((data >> pos) % 2).astype(bool)
            res = xr.DataArray(res, coords=self._value.coords,
                               attrs=self._value.attrs,
                               dims=self._value.dims)
        else:
            res = ((data >> pos) % 2).astype(bool)
        return res


class NCOLCIBase(BaseFileHandler):
    """The OLCI reader base."""

    rows_name = "rows"
    cols_name = "columns"

    def __init__(self, filename, filename_info, filetype_info,
                 engine=None):
        """Init the olci reader base."""
        super().__init__(filename, filename_info, filetype_info)
        self._engine = engine
        self._start_time = filename_info['start_time']
        self._end_time = filename_info['end_time']
        # TODO: get metadata from the manifest file (xfdumanifest.xml)
        self.platform_name = PLATFORM_NAMES[filename_info['mission_id']]
        self.sensor = 'olci'

    @cached_property
    def nc(self):
        """Get the nc xr dataset."""
        f_obj = open_file_or_filename(self.filename)
        dataset = xr.open_dataset(f_obj,
                                  decode_cf=True,
                                  mask_and_scale=True,
                                  engine=self._engine,
                                  chunks={self.cols_name: CHUNK_SIZE,
                                          self.rows_name: CHUNK_SIZE})
        return dataset.rename({self.cols_name: 'x', self.rows_name: 'y'})

    @property
    def start_time(self):
        """Start time property."""
        return self._start_time

    @property
    def end_time(self):
        """End time property."""
        return self._end_time

    def get_dataset(self, key, info):
        """Load a dataset."""
        logger.debug('Reading %s.', key['name'])
        variable = self.nc[key['name']]

        return variable


class NCOLCICal(NCOLCIBase):
    """Dummy class for calibration."""


class NCOLCIGeo(NCOLCIBase):
    """Dummy class for navigation."""


class NCOLCIChannelBase(NCOLCIBase):
    """Base class for channel reading."""

    def __init__(self, filename, filename_info, filetype_info, engine=None):
        """Init the file handler."""
        super().__init__(filename, filename_info, filetype_info, engine)
        self.channel = filename_info.get('dataset_name')
        self.reflectance_prefix = 'Oa'
        self.reflectance_suffix = '_reflectance'


class NCOLCI1B(NCOLCIChannelBase):
    """File handler for OLCI l1b."""

    def __init__(self, filename, filename_info, filetype_info, cal, engine=None):
        """Init the file handler."""
        super().__init__(filename, filename_info, filetype_info, engine)
        self.cal = cal.nc

    @staticmethod
    def _get_items(idx, solar_flux):
        """Get items."""
        return solar_flux[idx]

    def _get_solar_flux(self, band):
        """Get the solar flux for the band."""
        solar_flux = self.cal['solar_flux'].isel(bands=band).values
        d_index = self.cal['detector_index'].fillna(0).astype(int)

        return da.map_blocks(self._get_items, d_index.data,
                             solar_flux=solar_flux, dtype=solar_flux.dtype)

    def get_dataset(self, key, info):
        """Load a dataset."""
        if self.channel != key['name']:
            return
        logger.debug('Reading %s.', key['name'])

        radiances = self.nc[self.channel + '_radiance']

        if key['calibration'] == 'reflectance':
            idx = int(key['name'][2:]) - 1
            sflux = self._get_solar_flux(idx)
            radiances = radiances / sflux * np.pi * 100
            radiances.attrs['units'] = '%'

        radiances.attrs['platform_name'] = self.platform_name
        radiances.attrs['sensor'] = self.sensor
        radiances.attrs.update(key.to_dict())
        return radiances


class NCOLCI2(NCOLCIChannelBase):
    """File handler for OLCI l2."""

    def get_dataset(self, key, info):
        """Load a dataset."""
        if self.channel is not None and self.channel != key['name']:
            return
        logger.debug('Reading %s.', key['name'])
        if self.channel is not None and self.channel.startswith(self.reflectance_prefix):
            dataset = self.nc[self.channel + self.reflectance_suffix]
        else:
            dataset = self.nc[info['nc_key']]

        if key['name'] == 'wqsf':
            dataset.attrs['_FillValue'] = 1
        elif key['name'] == 'mask':
            dataset = self.getbitmask(dataset)

        dataset.attrs['platform_name'] = self.platform_name
        dataset.attrs['sensor'] = self.sensor
        dataset.attrs.update(key.to_dict())
        return dataset

    def getbitmask(self, wqsf, items=None):
        """Get the bitmask."""
        if items is None:
            items = ["INVALID", "SNOW_ICE", "INLAND_WATER", "SUSPECT",
                     "AC_FAIL", "CLOUD", "HISOLZEN", "OCNN_FAIL",
                     "CLOUD_MARGIN", "CLOUD_AMBIGUOUS", "LOWRW", "LAND"]
        bflags = BitFlags(wqsf)
        return reduce(np.logical_or, [bflags[item] for item in items])


class NCOLCILowResData(NCOLCIBase):
    """Handler for low resolution data."""

    rows_name = "tie_rows"
    cols_name = "tie_columns"

    def __init__(self, filename, filename_info, filetype_info,
                 engine=None):
        """Init the file handler."""
        super().__init__(filename, filename_info, filetype_info, engine)
        self.l_step = self.nc.attrs['al_subsampling_factor']
        self.c_step = self.nc.attrs['ac_subsampling_factor']

    def _do_interpolate(self, data):

        if not isinstance(data, tuple):
            data = (data,)

        shape = data[0].shape

        from geotiepoints.interpolator import Interpolator
        tie_lines = np.arange(0, (shape[0] - 1) * self.l_step + 1, self.l_step)
        tie_cols = np.arange(0, (shape[1] - 1) * self.c_step + 1, self.c_step)
        lines = np.arange((shape[0] - 1) * self.l_step + 1)
        cols = np.arange((shape[1] - 1) * self.c_step + 1)
        along_track_order = 1
        cross_track_order = 3
        satint = Interpolator([x.values for x in data],
                              (tie_lines, tie_cols),
                              (lines, cols),
                              along_track_order,
                              cross_track_order)
        int_data = satint.interpolate()

        return [xr.DataArray(da.from_array(x, chunks=(CHUNK_SIZE, CHUNK_SIZE)),
                             dims=['y', 'x']) for x in int_data]

    @property
    def _need_interpolation(self):
        return (self.c_step != 1 or self.l_step != 1)


class NCOLCIAngles(NCOLCILowResData):
    """File handler for the OLCI angles."""

    datasets = {'satellite_azimuth_angle': 'OAA',
                'satellite_zenith_angle': 'OZA',
                'solar_azimuth_angle': 'SAA',
                'solar_zenith_angle': 'SZA'}

    def get_dataset(self, key, info):
        """Load a dataset."""
        if key['name'] not in self.datasets:
            return

        logger.debug('Reading %s.', key['name'])

        if self._need_interpolation:
            if key['name'].startswith('satellite'):
                azi, zen = self.satellite_angles
            elif key['name'].startswith('solar'):
                azi, zen = self.sun_angles
            else:
                raise NotImplementedError("Don't know how to read " + key['name'])

            if 'zenith' in key['name']:
                values = zen
            elif 'azimuth' in key['name']:
                values = azi
            else:
                raise NotImplementedError("Don't know how to read " + key['name'])
        else:
            values = self.nc[self.datasets[key['name']]]

        values.attrs['platform_name'] = self.platform_name
        values.attrs['sensor'] = self.sensor

        values.attrs.update(key.to_dict())
        return values

    @cached_property
    def sun_angles(self):
        """Return the sun angles."""
        zen = self.nc[self.datasets['solar_zenith_angle']]
        azi = self.nc[self.datasets['solar_azimuth_angle']]
        azi, zen = self._interpolate_angles(azi, zen)
        return azi, zen

    @cached_property
    def satellite_angles(self):
        """Return the satellite angles."""
        zen = self.nc[self.datasets['satellite_zenith_angle']]
        azi = self.nc[self.datasets['satellite_azimuth_angle']]
        azi, zen = self._interpolate_angles(azi, zen)
        return azi, zen

    def _interpolate_angles(self, azi, zen):
        aattrs = azi.attrs
        zattrs = zen.attrs
        x, y, z = angle2xyz(azi, zen)
        x, y, z = self._do_interpolate((x, y, z))
        azi, zen = xyz2angle(x, y, z)
        azi.attrs = aattrs
        zen.attrs = zattrs
        return azi, zen


class NCOLCIMeteo(NCOLCILowResData):
    """File handler for the OLCI meteo data."""

    datasets = ['humidity', 'sea_level_pressure', 'total_columnar_water_vapour', 'total_ozone']

    def __init__(self, filename, filename_info, filetype_info,
                 engine=None):
        """Init the file handler."""
        super().__init__(filename, filename_info, filetype_info, engine)
        self.cache = {}

    # TODO: the following depends on more than columns, rows
    # float atmospheric_temperature_profile(tie_rows, tie_columns, tie_pressure_levels) ;
    # float horizontal_wind(tie_rows, tie_columns, wind_vectors) ;
    # float reference_pressure_level(tie_pressure_levels) ;

    def get_dataset(self, key, info):
        """Load a dataset."""
        if key['name'] not in self.datasets:
            return

        logger.debug('Reading %s.', key['name'])

        if self._need_interpolation and self.cache.get(key['name']) is None:

            data = self.nc[key['name']]

            values, = self._do_interpolate(data)
            values.attrs = data.attrs

            self.cache[key['name']] = values

        elif key['name'] in self.cache:
            values = self.cache[key['name']]
        else:
            values = self.nc[key['name']]

        values.attrs['platform_name'] = self.platform_name
        values.attrs['sensor'] = self.sensor

        values.attrs.update(key.to_dict())
        return values
