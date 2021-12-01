#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 Satpy developers
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
"""GCOM-W1 AMSR2 Level 2 files from the GAASP software.

GAASP output files are in the NetCDF4 format. Software is provided by NOAA
and is also distributed by the CSPP group. More information on the products
supported by this reader can be found here:
https://www.star.nesdis.noaa.gov/jpss/gcom.php for more information.

GAASP includes both swath/granule products and gridded products. Swath
products are provided in files with "MBT", "OCEAN", "SNOW", or "SOIL" in the
filename. Gridded products are in files with "SEAICE-SH" or "SEAICE-NH" in the
filename where SH stands for South Hemisphere and NH stands for North
Hemisphere. These gridded products are on the EASE2 North pole and South pole
grids. See https://nsidc.org/ease/ease-grid-projection-gt for more details.

Note that since SEAICE products can be on both the northern or
southern hemisphere or both depending on what files are provided to Satpy, this
reader appends a `_NH` and `_SH` suffix to all variable names that are
dynamically discovered from the provided files.

"""

import logging
from datetime import datetime
from typing import Tuple

import numpy as np
import xarray as xr
from pyproj import CRS
from pyresample.geometry import AreaDefinition

from satpy import CHUNK_SIZE
from satpy._compat import cached_property
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)


class GAASPFileHandler(BaseFileHandler):
    """Generic file handler for GAASP output files."""

    y_dims: Tuple[str, ...] = (
        'Number_of_Scans',
    )
    x_dims: Tuple[str, ...] = (
        'Number_of_hi_rez_FOVs',
        'Number_of_low_rez_FOVs',
    )
    time_dims = (
        'Time_Dimension',
    )
    is_gridded = False
    dim_resolutions = {
        'Number_of_hi_rez_FOVs': 5000,
        'Number_of_low_rez_FOVs': 10000,
    }

    @cached_property
    def nc(self):
        """Get the xarray dataset for this file."""
        chunks = {dim_name: CHUNK_SIZE for dim_name in
                  self.y_dims + self.x_dims + self.time_dims}
        nc = xr.open_dataset(self.filename,
                             decode_cf=True,
                             mask_and_scale=False,
                             chunks=chunks)

        if len(self.time_dims) == 1:
            nc = nc.rename({self.time_dims[0]: 'time'})
        return nc

    @property
    def start_time(self):
        """Get start time of observation."""
        try:
            return self.filename_info['start_time']
        except KeyError:
            time_str = self.nc.attrs['time_coverage_start']
            return datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%fZ")

    @property
    def end_time(self):
        """Get end time of observation."""
        try:
            return self.filename_info['end_time']
        except KeyError:
            time_str = self.nc.attrs['time_coverage_end']
            return datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%fZ")

    @property
    def sensor_names(self):
        """Sensors who have data in this file."""
        return {self.nc.attrs['instrument_name'].lower()}

    @property
    def platform_name(self):
        """Name of the platform whose data is stored in this file."""
        return self.nc.attrs['platform_name']

    def _get_var_name_without_suffix(self, var_name):
        var_suffix = self.filetype_info.get('var_suffix', "")
        if var_suffix:
            var_name = var_name[:-len(var_suffix)]
        return var_name

    def _scale_data(self, data_arr, attrs):
        # handle scaling
        # take special care for integer/category fields
        scale_factor = attrs.pop('scale_factor', 1.)
        add_offset = attrs.pop('add_offset', 0.)
        scaling_needed = not (scale_factor == 1 and add_offset == 0)
        if scaling_needed:
            data_arr = data_arr * scale_factor + add_offset
        return data_arr, attrs

    @staticmethod
    def _nan_for_dtype(data_arr_dtype):
        # don't force the conversion from 32-bit float to 64-bit float
        # if we don't have to
        if data_arr_dtype.type == np.float32:
            return np.float32(np.nan)
        if np.issubdtype(data_arr_dtype, np.timedelta64):
            return np.timedelta64('NaT')
        if np.issubdtype(data_arr_dtype, np.datetime64):
            return np.datetime64('NaT')
        return np.nan

    def _fill_data(self, data_arr, attrs):
        fill_value = attrs.pop('_FillValue', None)
        is_int = np.issubdtype(data_arr.dtype, np.integer)
        has_flag_comment = 'comment' in attrs
        if is_int and has_flag_comment:
            # category product
            fill_out = fill_value
            attrs['_FillValue'] = fill_out
        else:
            fill_out = self._nan_for_dtype(data_arr.dtype)
        if fill_value is not None:
            data_arr = data_arr.where(data_arr != fill_value, fill_out)
        return data_arr, attrs

    def get_dataset(self, dataid, ds_info):
        """Load, scale, and collect metadata for the specified DataID."""
        orig_var_name = self._get_var_name_without_suffix(dataid['name'])
        data_arr = self.nc[orig_var_name].copy()
        attrs = data_arr.attrs.copy()
        data_arr, attrs = self._scale_data(data_arr, attrs)
        data_arr, attrs = self._fill_data(data_arr, attrs)

        attrs.update({
            'platform_name': self.platform_name,
            'sensor': sorted(self.sensor_names)[0],
            'start_time': self.start_time,
            'end_time': self.end_time,
        })
        dim_map = dict(zip(data_arr.dims, ('y', 'x')))
        # rename dims
        data_arr = data_arr.rename(**dim_map)
        # drop coords, the base reader will recreate these
        data_arr = data_arr.reset_coords(drop=True)
        data_arr.attrs = attrs
        return data_arr

    def _available_if_this_file_type(self, configured_datasets):
        for is_avail, ds_info in (configured_datasets or []):
            if is_avail is not None:
                # some other file handler said it has this dataset
                # we don't know any more information than the previous
                # file handler so let's yield early
                yield is_avail, ds_info
                continue
            yield self.file_type_matches(ds_info['file_type']), ds_info

    def _add_lonlat_coords(self, data_arr, ds_info):
        lat_coord = None
        lon_coord = None
        for coord_name in data_arr.coords:
            if 'longitude' in coord_name.lower():
                lon_coord = coord_name
            if 'latitude' in coord_name.lower():
                lat_coord = coord_name
        ds_info['coordinates'] = [lon_coord, lat_coord]

    def _get_ds_info_for_data_arr(self, var_name, data_arr):
        var_suffix = self.filetype_info.get('var_suffix', "")
        ds_info = {
            'file_type': self.filetype_info['file_type'],
            'name': var_name + var_suffix,
        }
        x_dim_name = data_arr.dims[1]
        if x_dim_name in self.dim_resolutions:
            ds_info['resolution'] = self.dim_resolutions[x_dim_name]
        if not self.is_gridded and data_arr.coords:
            self._add_lonlat_coords(data_arr, ds_info)
        return ds_info

    def _is_2d_yx_data_array(self, data_arr):
        has_y_dim = data_arr.dims[0] in self.y_dims
        has_x_dim = data_arr.dims[1] in self.x_dims
        return has_y_dim and has_x_dim

    def _available_new_datasets(self):
        possible_vars = list(self.nc.data_vars.items()) + list(self.nc.coords.items())
        for var_name, data_arr in possible_vars:
            if data_arr.ndim != 2:
                # we don't currently handle non-2D variables
                continue
            if not self._is_2d_yx_data_array(data_arr):
                # we need 'traditional' y/x dimensions currently
                continue

            ds_info = self._get_ds_info_for_data_arr(var_name, data_arr)
            yield True, ds_info

    def available_datasets(self, configured_datasets=None):
        """Dynamically discover what variables can be loaded from this file.

        See :meth:`satpy.readers.file_handlers.BaseHandler.available_datasets`
        for more information.

        """
        yield from self._available_if_this_file_type(configured_datasets)
        yield from self._available_new_datasets()


class GAASPGriddedFileHandler(GAASPFileHandler):
    """GAASP file handler for gridded products like SEAICE."""

    y_dims = (
        'Number_of_Y_Dimension',
    )
    x_dims = (
        'Number_of_X_Dimension',
    )
    dim_resolutions = {
        'Number_of_X_Dimension': 10000,
    }
    is_gridded = True

    @staticmethod
    def _get_extents(data_shape, res):
        # assume data is centered at projection center
        x_min = -(data_shape[1] / 2.0) * res
        x_max = (data_shape[1] / 2.0) * res
        y_min = -(data_shape[0] / 2.0) * res
        y_max = (data_shape[0] / 2.0) * res
        return x_min, y_min, x_max, y_max

    def get_area_def(self, dataid):
        """Create area definition for equirectangular projected data."""
        var_suffix = self.filetype_info.get('var_suffix', '')
        area_name = 'gaasp{}'.format(var_suffix)
        orig_var_name = self._get_var_name_without_suffix(dataid['name'])
        data_shape = self.nc[orig_var_name].shape
        crs = CRS(self.filetype_info['grid_epsg'])
        res = dataid['resolution']
        extent = self._get_extents(data_shape, res)
        area_def = AreaDefinition(
            area_name,
            area_name,
            area_name,
            crs,
            data_shape[1],
            data_shape[0],
            extent
        )
        return area_def


class GAASPLowResFileHandler(GAASPFileHandler):
    """GAASP file handler for files that only have low resolution products."""

    x_dims = (
        'Number_of_low_rez_FOVs',
    )
    dim_resolutions = {
        'Number_of_low_rez_FOVs': 10000,
    }
