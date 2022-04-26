#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2021 Satpy developers
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
"""Interface to CLAVR-X HDF4 products."""

import logging
import os
from glob import glob
from pathlib import Path
from typing import Optional

import netCDF4
import numpy as np
import xarray as xr
from pyresample import geometry

from satpy import CHUNK_SIZE
from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.hdf4_utils import SDS, HDF4FileHandler

LOG = logging.getLogger(__name__)


CF_UNITS = {
    'none': '1',
}

SENSORS = {
    'MODIS': 'modis',
    'VIIRS': 'viirs',
    'AVHRR': 'avhrr',
    'AHI': 'ahi',
    'ABI': 'abi',
}
PLATFORMS = {
    'SNPP': 'npp',
    'HIM8': 'himawari8',
    'HIM9': 'himawari9',
    'H08': 'himawari8',
    'H09': 'himawari9',
    'G16': 'GOES-16',
    'G17': 'GOES-17'
}
ROWS_PER_SCAN = {
    'viirs': 16,
    'modis': 10,
}
NADIR_RESOLUTION = {
    'viirs': 742,
    'modis': 1000,
    'avhrr': 1050,
    'ahi': 2000,
    'abi': 2004,
}


def _get_sensor(sensor: str) -> str:
    """Get the sensor."""
    for k, v in SENSORS.items():
        if k in sensor:
            return v
    raise ValueError("Unknown sensor '{}'".format(sensor))


def _get_platform(platform: str) -> str:
    """Get the platform."""
    for k, v in PLATFORMS.items():
        if k in platform:
            return v
    return platform


def _get_rows_per_scan(sensor: str) -> Optional[int]:
    """Get number of rows per scan."""
    for k, v in ROWS_PER_SCAN.items():
        if sensor.startswith(k):
            return v
    return None


def _remove_attributes(attrs: dict) -> dict:
    """Remove attributes that described data before scaling."""
    old_attrs = ['unscaled_missing', 'SCALED_MIN', 'SCALED_MAX',
                 'SCALED_MISSING']

    for attr_key in old_attrs:
        attrs.pop(attr_key, None)
    return attrs


class _CLAVRxHelper:
    """A base class for the CLAVRx File Handlers."""

    @staticmethod
    def _scale_data(data_arr: xr.DataArray, scale_factor: float, add_offset: float) -> xr.DataArray:
        """Scale data, if needed."""
        scaling_needed = not (scale_factor == 1.0 and add_offset == 0.0)
        if scaling_needed:
            data_arr = data_arr * scale_factor + add_offset
        return data_arr

    @staticmethod
    def _get_data(data: xr.DataArray, dataset_id: dict) -> xr.DataArray:
        """Get a dataset."""
        if dataset_id.get('resolution'):
            data.attrs['resolution'] = dataset_id['resolution']

        attrs = data.attrs.copy()

        fill = attrs.get('_FillValue')
        factor = attrs.pop('scale_factor', (np.ones(1, dtype=data.dtype))[0])
        offset = attrs.pop('add_offset', (np.zeros(1, dtype=data.dtype))[0])
        valid_range = attrs.get('valid_range', [None])
        if isinstance(valid_range, np.ndarray):
            attrs["valid_range"] = valid_range.tolist()

        flags = not data.attrs.get("SCALED", 1) and any(data.attrs.get("flag_values", [None]))
        if not flags:
            data = data.where(data != fill)
            data = _CLAVRxHelper._scale_data(data, factor, offset)
            # don't need _FillValue if it has been applied.
            attrs.pop('_FillValue', None)

        if all(valid_range):
            valid_min = _CLAVRxHelper._scale_data(valid_range[0], factor, offset)
            valid_max = _CLAVRxHelper._scale_data(valid_range[1], factor, offset)
            if flags:
                data = data.where((data >= valid_min) & (data <= valid_max), fill)
            else:
                data = data.where((data >= valid_min) & (data <= valid_max))
            attrs['valid_range'] = [valid_min, valid_max]

        data.attrs = _remove_attributes(attrs)

        return data

    @staticmethod
    def _area_extent(x, y, h):
        x_l = h * x[0]
        x_r = h * x[-1]
        y_l = h * y[-1]
        y_u = h * y[0]
        ncols = x.shape[0]
        nlines = y.shape[0]
        x_half = (x_r - x_l) / (ncols - 1) / 2.
        y_half = (y_u - y_l) / (nlines - 1) / 2.
        area_extent = (x_l - x_half, y_l - y_half, x_r + x_half, y_u + y_half)
        return area_extent, ncols, nlines

    @staticmethod
    def _read_pug_fixed_grid(projection, distance_multiplier=1.0) -> dict:
        """Read from recent PUG format, where axes are in meters."""
        a = projection.semi_major_axis
        h = projection.perspective_point_height
        b = projection.semi_minor_axis

        lon_0 = projection.longitude_of_projection_origin
        sweep_axis = projection.sweep_angle_axis[0]

        proj_dict = {'a': float(a) * distance_multiplier,
                     'b': float(b) * distance_multiplier,
                     'lon_0': float(lon_0),
                     'h': float(h) * distance_multiplier,
                     'proj': 'geos',
                     'units': 'm',
                     'sweep': sweep_axis}
        return proj_dict

    @staticmethod
    def _find_input_nc(filename: str, l1b_base: str) -> str:
        file_path = Path(filename)
        dirname = file_path.parent
        l1b_filename = dirname.joinpath(l1b_base + '.nc')
        if l1b_filename.exists():
            return str(l1b_filename)

        glob_pat = os.path.join(dirname, l1b_base + '*R20*.nc')
        LOG.debug("searching for {0}".format(glob_pat))
        l1b_filenames = list(glob(glob_pat))
        if not l1b_filenames:
            raise IOError("Could not find navigation donor for {0}"
                          " in same directory as CLAVR-x data".format(l1b_base))
        LOG.debug('Candidate nav donors: {0}'.format(repr(l1b_filenames)))
        return l1b_filenames[0]

    @staticmethod
    def _read_axi_fixed_grid(filename: str, l1b_attr) -> geometry.AreaDefinition:
        """Read a fixed grid.

        CLAVR-x does not transcribe fixed grid parameters to its output
        We have to recover that information from the original input file,
        which is partially named as L1B attribute

        example attributes found in L2 CLAVR-x files:
        sensor = "AHI" ;
        platform = "HIM8" ;
        FILENAME = "clavrx_H08_20180719_1300.level2.hdf" ;
        L1B = "clavrx_H08_20180719_1300" ;
        """
        LOG.debug("looking for corresponding input file for {0}"
                  " to act as fixed grid navigation donor".format(l1b_attr))
        l1b_path = _CLAVRxHelper._find_input_nc(filename, l1b_attr)
        LOG.info("Since CLAVR-x does not include fixed-grid parameters,"
                 " using input file {0} as donor".format(l1b_path))
        l1b = netCDF4.Dataset(l1b_path)
        proj = None
        proj_var = l1b.variables.get("Projection", None)
        if proj_var is not None:
            # hsd2nc input typically used by CLAVR-x uses old-form km for axes/height
            LOG.debug("found hsd2nc-style draft PUG fixed grid specification")
            proj = _CLAVRxHelper._read_pug_fixed_grid(proj_var, 1000.0)
        if proj is None:  # most likely to come into play for ABI cases
            proj_var = l1b.variables.get("goes_imager_projection", None)
            if proj_var is not None:
                LOG.debug("found cmip-style final PUG fixed grid specification")
                proj = _CLAVRxHelper._read_pug_fixed_grid(proj_var)
        if not proj:
            raise ValueError("Unable to recover projection information"
                             " for {0}".format(filename))

        h = float(proj['h'])
        x, y = l1b['x'], l1b['y']
        area_extent, ncols, nlines = _CLAVRxHelper._area_extent(x, y, h)

        # LOG.debug(repr(proj))
        # LOG.debug(repr(area_extent))

        area = geometry.AreaDefinition(
            'ahi_geos',
            "AHI L2 file area",
            'ahi_geos',
            proj,
            ncols,
            nlines,
            np.asarray(area_extent))

        return area

    @staticmethod
    def get_metadata(sensor, platform, attrs: dict, ds_info: dict) -> dict:
        """Get metadata."""
        i = {}
        i.update(attrs)
        i.update(ds_info)

        flag_meanings = i.get('flag_meanings', None)
        if not i.get('SCALED', 1) and not flag_meanings:
            i['flag_meanings'] = '<flag_meanings_unknown>'
            i.setdefault('flag_values', [None])
        u = i.get('units')
        if u in CF_UNITS:
            # CF compliance
            i['units'] = CF_UNITS[u]
            if u.lower() == "none":
                i['units'] = "1"
        i['sensor'] = sensor
        i['platform_name'] = platform
        rps = _get_rows_per_scan(sensor)
        if rps:
            i['rows_per_scan'] = rps
        i['reader'] = 'clavrx'

        return i


class CLAVRXHDF4FileHandler(HDF4FileHandler, _CLAVRxHelper):
    """A file handler for CLAVRx files."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init method."""
        super(CLAVRXHDF4FileHandler, self).__init__(filename,
                                                    filename_info,
                                                    filetype_info)

    @property
    def start_time(self):
        """Get the start time."""
        return self.filename_info['start_time']

    @property
    def end_time(self):
        """Get the end time."""
        return self.filename_info.get('end_time', self.start_time)

    def get_dataset(self, dataset_id, ds_info):
        """Get a dataset."""
        var_name = ds_info.get('file_key', dataset_id['name'])
        data = self[var_name]
        data = _CLAVRxHelper._get_data(data, dataset_id)
        data.attrs = _CLAVRxHelper.get_metadata(self.sensor, self.platform,
                                                data.attrs, ds_info)
        return data

    def get_nadir_resolution(self, sensor):
        """Get nadir resolution."""
        for k, v in NADIR_RESOLUTION.items():
            if sensor.startswith(k):
                return v
        res = self.filename_info.get('resolution')
        if res.endswith('m'):
            return int(res[:-1])
        elif res is not None:
            return int(res)

    def available_datasets(self, configured_datasets=None):
        """Automatically determine datasets provided by this file."""
        self.sensor = _get_sensor(self.file_content.get('/attr/sensor'))
        self.platform = _get_platform(self.file_content.get('/attr/platform'))

        nadir_resolution = self.get_nadir_resolution(self.sensor)
        coordinates = ('longitude', 'latitude')
        handled_variables = set()

        # update previously configured datasets
        for is_avail, ds_info in (configured_datasets or []):
            this_res = ds_info.get('resolution')
            this_coords = ds_info.get('coordinates')
            # some other file handler knows how to load this
            if is_avail is not None:
                yield is_avail, ds_info

            var_name = ds_info.get('file_key', ds_info['name'])
            matches = self.file_type_matches(ds_info['file_type'])
            # we can confidently say that we can provide this dataset and can
            # provide more info
            if matches and var_name in self and this_res != nadir_resolution:
                handled_variables.add(var_name)
                new_info = ds_info.copy()  # don't mess up the above yielded
                new_info['resolution'] = nadir_resolution
                if self._is_polar() and this_coords is None:
                    new_info['coordinates'] = coordinates
                yield True, new_info
            elif is_avail is None:
                # if we didn't know how to handle this dataset and no one else did
                # then we should keep it going down the chain
                yield is_avail, ds_info

        # add new datasets
        for var_name, val in self.file_content.items():
            if isinstance(val, SDS):
                ds_info = {
                    'file_type': self.filetype_info['file_type'],
                    'resolution': nadir_resolution,
                    'name': var_name,
                }
                if self._is_polar():
                    ds_info['coordinates'] = ['longitude', 'latitude']
                yield True, ds_info

    def get_shape(self, dataset_id, ds_info):
        """Get the shape."""
        var_name = ds_info.get('file_key', dataset_id['name'])
        return self[var_name + '/shape']

    def _is_polar(self):
        l1b_att, inst_att = (str(self.file_content.get('/attr/L1B', None)),
                             str(self.file_content.get('/attr/sensor', None)))

        return (inst_att != 'AHI' and 'GOES' not in inst_att) or (l1b_att is None)

    def get_area_def(self, key):
        """Get the area definition of the data at hand."""
        if self._is_polar():  # then it doesn't have a fixed grid
            return super(CLAVRXHDF4FileHandler, self).get_area_def(key)

        l1b_att = str(self.file_content.get('/attr/L1B', None))
        area_def = _CLAVRxHelper._read_axi_fixed_grid(self.filename, l1b_att)
        return area_def


class CLAVRXNetCDFFileHandler(_CLAVRxHelper, BaseFileHandler):
    """File Handler for CLAVRX netcdf files."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init method."""
        super(CLAVRXNetCDFFileHandler, self).__init__(filename,
                                                      filename_info,
                                                      filetype_info,
                                                      )

        self.nc = xr.open_dataset(filename,
                                  decode_cf=True,
                                  mask_and_scale=False,
                                  decode_coords=True,
                                  chunks=CHUNK_SIZE)
        # y,x is used in satpy, bands rather than channel using in xrimage
        self.nc = self.nc.rename_dims({'scan_lines_along_track_direction': "y",
                                       'pixel_elements_along_scan_direction': "x"})

        self.platform = _get_platform(
            self.filename_info.get('platform_shortname', None))
        self.sensor = self.nc.attrs.get('sensor', None)
        # coordinates need scaling and valid_range (mask_and_scale won't work on valid_range)
        self.nc.coords["latitude"] = _CLAVRxHelper._get_data(self.nc.coords["latitude"],
                                                             {"name": "latitude"})
        self.nc.coords["longitude"] = _CLAVRxHelper._get_data(self.nc.coords["longitude"],
                                                              {"name": "longitude"})

    def _get_ds_info_for_data_arr(self, var_name):
        ds_info = {
            'file_type': self.filetype_info['file_type'],
            'name': var_name,
        }
        return ds_info

    def _is_2d_yx_data_array(self, data_arr):
        has_y_dim = data_arr.dims[0] == "y"
        has_x_dim = data_arr.dims[1] == "x"
        return has_y_dim and has_x_dim

    def _available_new_datasets(self, handled_vars):
        """Metadata for available variables other than BT."""
        possible_vars = list(self.nc.items()) + list(self.nc.coords.items())
        for var_name, data_arr in possible_vars:
            if var_name in handled_vars:
                continue
            if data_arr.ndim != 2:
                # we don't currently handle non-2D variables
                continue
            if not self._is_2d_yx_data_array(data_arr):
                # we need 'traditional' y/x dimensions currently
                continue

            ds_info = self._get_ds_info_for_data_arr(var_name)
            yield True, ds_info

    def available_datasets(self, configured_datasets=None):
        """Dynamically discover what variables can be loaded from this file.

        See :meth:`satpy.readers.file_handlers.BaseHandler.available_datasets`
        for more information.

        """
        handled_vars = set()
        for is_avail, ds_info in (configured_datasets or []):
            if is_avail is not None:
                # some other file handler said it has this dataset
                # we don't know any more information than the previous
                # file handler so let's yield early
                yield is_avail, ds_info
                continue
            if self.file_type_matches(ds_info['file_type']):
                handled_vars.add(ds_info['name'])
            yield self.file_type_matches(ds_info['file_type']), ds_info
        yield from self._available_new_datasets(handled_vars)

    def _is_polar(self):
        l1b_att, inst_att = (str(self.nc.attrs.get('L1B', None)),
                             str(self.nc.attrs.get('sensor', None)))

        return (inst_att != 'AHI' and 'GOES' not in inst_att) or (l1b_att is None)

    def get_area_def(self, key):
        """Get the area definition of the data at hand."""
        if self._is_polar():  # then it doesn't have a fixed grid
            return super(CLAVRXNetCDFFileHandler, self).get_area_def(key)

        l1b_att = str(self.nc.attrs.get('L1B', None))
        return _CLAVRxHelper._read_axi_fixed_grid(self.filename, l1b_att)

    def get_dataset(self, dataset_id, ds_info):
        """Get a dataset."""
        var_name = ds_info.get('name', dataset_id['name'])
        data = self[var_name]
        data = _CLAVRxHelper._get_data(data, dataset_id)
        data.attrs = _CLAVRxHelper.get_metadata(self.sensor, self.platform,
                                                data.attrs, ds_info)
        return data

    def __getitem__(self, item):
        """Wrap around `self.nc[item]`."""
        data = self.nc[item]
        return data
