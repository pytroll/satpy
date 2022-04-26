#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016-2021 Satpy developers
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
from contextlib import suppress
from functools import reduce, lru_cache

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
                  'S3B': 'Sentinel-3B'}


class BitFlags:
    """Manipulate flags stored bitwise."""

    def __init__(self, masks, meanings):
        """Init the flags."""
        self._masks = masks
        self._meanings = meanings
        self._map = dict(zip(meanings, masks))

    def match_item(self, item, data):
        """Match any of the item."""
        mask = self._map[item]
        return np.bitwise_and(data, mask).astype(np.bool)

    def match_any(self, items, data):
        """Match any of the items in data."""
        mask = reduce(np.bitwise_or, [self._map[item] for item in items])
        return np.bitwise_and(data, mask).astype(np.bool)

    def __eq__(self, other):
        """Check equality."""
        return all(self._masks == other._masks) and self._meanings == other._meanings


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
        self.open_file = None

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

    def __del__(self):
        """Close the NetCDF file that may still be open."""
        with suppress(IOError, OSError, AttributeError, TypeError):
            self.nc.close()

    def _fill_dataarray_attrs(self, data, key, info=None):
        """Fill the dataarray with relevant attributes."""
        data.attrs['platform_name'] = self.platform_name
        data.attrs['sensor'] = self.sensor
        data.attrs.update(key.to_dict())
        if info is not None:
            info = info.copy()
            for key in ["nc_key", "coordinates", "file_type", "name"]:
                info.pop(key, None)
            data.attrs.update(info)


class NCOLCIGeo(NCOLCIBase):
    """Dummy class for navigation."""


class NCOLCIChannelBase(NCOLCIBase):
    """Base class for channel reading."""

    def __init__(self, filename, filename_info, filetype_info, engine=None):
        """Init the file handler."""
        super().__init__(filename, filename_info, filetype_info, engine)
        self.channel = filename_info.get('dataset_name')


class NCOLCI1B(NCOLCIChannelBase):
    """File handler for OLCI l1b."""

    def __init__(self, filename, filename_info, filetype_info, cal,
                 engine=None):
        """Init the file handler."""
        super().__init__(filename, filename_info, filetype_info, engine)
        self.cal = cal.nc

    def _get_solar_flux(self, band):
        """Get the solar flux for the band."""
        solar_flux = self.cal['solar_flux'].isel(bands=band).values
        d_index = self.cal['detector_index'].fillna(0).astype(int)

        return da.map_blocks(_take_indices, d_index.data,
                             data=solar_flux, dtype=solar_flux.dtype)

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

        self._fill_dataarray_attrs(radiances, key)
        return radiances


def _take_indices(idx, data):
    """Take values from data using idx."""
    return data[idx]


class NCOLCI2(NCOLCIChannelBase):
    """File handler for OLCI l2."""

    def get_dataset(self, key, info):
        """Load a dataset."""
        if self.channel is not None and self.channel != key['name']:
            return
        logger.debug('Reading %s.', key['name'])
        if self.channel is not None and self.channel.startswith('Oa'):
            dataset = self.nc[self.channel + '_reflectance']
        else:
            dataset = self.nc[info['nc_key']]

        self._fill_dataarray_attrs(dataset, key, info)
        return dataset


class NCOLCI2Flags(NCOLCIChannelBase):
    """File handler for OLCI l2 flag files.

    A correctly-initialized BitFlags instance is added to the "bitflags"
    attribute in case the masked items are not defined (eg for wqsf).
    """

    def get_dataset(self, key, info):
        """Load a dataset."""
        logger.debug('Reading %s.', key['name'])
        dataset = self.nc[info['nc_key']]
        self.create_bitflags(dataset)

        if key['name'] == 'wqsf':
            dataset.attrs['_FillValue'] = 1
        elif "masked_items" in info:
            dataset = self.getbitmask(dataset, info["masked_items"])

        self._fill_dataarray_attrs(dataset, key)
        return dataset

    def create_bitflags(self, dataset):
        """Create the bitflags attribute."""
        bflags = BitFlags(dataset.attrs['flag_masks'],
                          dataset.attrs['flag_meanings'].split())
        dataset.attrs["bitflags"] = bflags

    def getbitmask(self, dataset, items=None):
        """Generate the bitmask."""
        if items is None:
            items = ["INVALID", "SNOW_ICE", "INLAND_WATER", "SUSPECT",
                     "AC_FAIL", "CLOUD", "HISOLZEN", "OCNN_FAIL",
                     "CLOUD_MARGIN", "CLOUD_AMBIGUOUS", "LOWRW", "LAND"]
        return dataset.attrs["bitflags"].match_any(items, dataset)


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
        """Do the interpolation."""
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
        key_name = key['name']
        if key_name not in self.datasets:
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
            data = self.nc[self.datasets[key_name]]

        self._fill_dataarray_attrs(data, key)
        return data

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
        """Interpolate angles."""
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

        values = self._get_full_resolution_dataset(key['name'])
        self._fill_dataarray_attrs(values, key)
        return values

    @lru_cache(None)
    def _get_full_resolution_dataset(self, key_name):
        """Get the full resolution dataset."""
        if self._needs_interpolation():

            data = self.nc[key_name]

            values, = self._do_interpolate(data)
            values.attrs = data.attrs

            return values

        return self.nc[key_name]
