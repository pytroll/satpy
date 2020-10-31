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

"""

import logging
import numpy as np
import xarray as xr
from satpy.readers.file_handlers import BaseFileHandler
from satpy import CHUNK_SIZE
from pyresample.geometry import AreaDefinition
from pyproj import CRS
from datetime import datetime

logger = logging.getLogger(__name__)


class GAASPFileHandler(BaseFileHandler):
    """Generic file handler for GAASP output files."""

    y_dims = (
        'Number_of_Scans',
    )
    x_dims = (
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

    def __init__(self, filename, filename_info, filetype_info):
        """Open the NetCDF file with xarray and prepare the Dataset for reading."""
        super().__init__(filename, filename_info, filetype_info)
        chunks = {dim_name: CHUNK_SIZE for dim_name in
                  self.y_dims + self.x_dims + self.time_dims}
        self.nc = xr.open_dataset(self.filename,
                                  decode_cf=True,
                                  mask_and_scale=False,
                                  chunks=chunks)

        if len(self.time_dims) == 1:
            self.nc = self.nc.rename({self.time_dims[0]: 'time'})

    @property
    def start_time(self):
        """Get start time of observation."""
        time_str = self.nc.attrs['time_coverage_start']
        return datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%fZ")

    @property
    def end_time(self):
        """Get end time of observation."""
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

    def get_dataset(self, dataid, ds_info):
        """Load, scale, and collect metadata for the specified DataID."""
        orig_var_name = self._get_var_name_without_suffix(dataid['name'])
        data_arr = self.nc[orig_var_name].copy()
        attrs = data_arr.attrs.copy()

        # handle scaling
        # take special care for integer/category fields
        scale_factor = data_arr.attrs.pop('scale_factor', 1.)
        add_offset = data_arr.attrs.pop('add_offset', 0.)
        fill_value = data_arr.attrs.pop('_FillValue', None)

        scaling_needed = not (scale_factor == 1 and add_offset == 0)
        if scaling_needed:
            data_arr = data_arr * scale_factor + add_offset

        is_int = np.issubdtype(data_arr.dtype, np.integer)
        has_flag_comment = 'comment' in data_arr.attrs
        if is_int and has_flag_comment:
            # category product
            fill_out = fill_value
            attrs['_FillValue'] = fill_out
        else:
            fill_out = np.nan
        if fill_value is not None:
            data_arr = data_arr.where(data_arr != fill_value, fill_out)

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

    def available_datasets(self, configured_datasets=None):
        """Dynamically discover what variables can be loaded from this file.

        See :meth:`satpy.readers.file_handlers.BaseHandler.available_datasets`
        for more information.

        """
        handled_variables = set()
        for is_avail, ds_info in (configured_datasets or []):
            if is_avail is not None:
                # some other file handler said it has this dataset
                # we don't know any more information than the previous
                # file handler so let's yield early
                yield is_avail, ds_info
                continue
            yield self.file_type_matches(ds_info['file_type']), ds_info

        # Provide new datasets
        var_suffix = self.filetype_info.get('var_suffix', "")
        possible_vars = list(self.nc.data_vars.items()) + list(self.nc.coords.items())
        for var_name, data_arr in possible_vars:
            if var_name in handled_variables:
                continue
            if data_arr.ndim != 2:
                # we don't currently handle non-2D variables
                continue
            has_y_dim = data_arr.dims[0] in self.y_dims
            has_x_dim = data_arr.dims[1] in self.x_dims
            if not has_y_dim or not has_x_dim:
                # we need 'traditional' y/x dimensions currently
                continue

            ds_info = {
                'file_type': self.filetype_info['file_type'],
                'name': var_name + var_suffix,
            }
            x_dim_name = data_arr.dims[1]
            if x_dim_name in self.dim_resolutions:
                ds_info['resolution'] = self.dim_resolutions[x_dim_name]
            if not self.is_gridded and data_arr.coords:
                lat_coord = None
                lon_coord = None
                for coord_name in data_arr.coords:
                    if 'longitude' in coord_name.lower():
                        lon_coord = coord_name
                    if 'latitude' in coord_name.lower():
                        lat_coord = coord_name
                ds_info['coordinates'] = [lon_coord, lat_coord]
            handled_variables.add(var_name)
            yield True, ds_info


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

    def get_area_def(self, dataid):
        """Create area definition for equirectangular projected data."""
        var_suffix = self.filetype_info.get('var_suffix', '')
        area_name = 'gaasp{}'.format(var_suffix)
        orig_var_name = self._get_var_name_without_suffix(dataid['name'])
        data_shape = self.nc[orig_var_name].shape
        crs = CRS(self.filetype_info['grid_epsg'])
        res = dataid['resolution']
        # assume data is centered at projection center
        x_min = -(data_shape[1] / 2.0) * res
        x_max = (data_shape[1] / 2.0) * res
        y_min = -(data_shape[0] / 2.0) * res
        y_max = (data_shape[0] / 2.0) * res
        extent = (x_min, y_min, x_max, y_max)
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


class GAASPLowRezFileHandler(GAASPFileHandler):
    """GAASP file handler for files that only have low resolution products."""

    x_dims = (
        'Number_of_low_rez_FOVs',
    )
    dim_resolutions = {
        'Number_of_low_rez_FOVs': 10000,
    }
