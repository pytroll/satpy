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
"""Interface to CLAVR-X HDF4 products.
"""
import os
import logging
import numpy as np
import netCDF4
from glob import glob
from satpy.readers.hdf4_utils import HDF4FileHandler, SDS
from pyresample import geometry

LOG = logging.getLogger(__name__)


CF_UNITS = {
    'none': '1',
}


class CLAVRXFileHandler(HDF4FileHandler):
    sensors = {
        'MODIS': 'modis',
        'VIIRS': 'viirs',
        'AVHRR': 'avhrr',
        'AHI': 'ahi',
        # 'ABI': 'abi',
    }
    platforms = {
        'SNPP': 'npp',
        'HIM8': 'himawari8',
        'HIM9': 'himawari9',
        'H08': 'himawari8',
        'H09': 'himawari9',
        # 'G16': 'GOES-16',
        # 'G17': 'GOES-17'
    }
    rows_per_scan = {
        'viirs': 16,
        'modis': 10,
    }
    nadir_resolution = {
        'viirs': 742,
        'modis': 1000,
        'avhrr': 1050,
        'ahi': 2000,
        # 'abi': 2004,
    }

    def get_sensor(self, sensor):
        for k, v in self.sensors.items():
            if k in sensor:
                return v
        raise ValueError("Unknown sensor '{}'".format(sensor))

    def get_platform(self, platform):
        for k, v in self.platforms.items():
            if k in platform:
                return v
        return platform

    def get_rows_per_scan(self, sensor):
        for k, v in self.rows_per_scan.items():
            if sensor.startswith(k):
                return v

    def get_nadir_resolution(self, sensor):
        for k, v in self.nadir_resolution.items():
            if sensor.startswith(k):
                return v
        res = self.filename_info.get('resolution')
        if res.endswith('m'):
            return int(res[:-1])
        elif res is not None:
            return int(res)

    @property
    def start_time(self):
        return self.filename_info['start_time']

    @property
    def end_time(self):
        return self.filename_info.get('end_time', self.start_time)

    def available_datasets(self, configured_datasets=None):
        """Automatically determine datasets provided by this file"""
        sensor = self.get_sensor(self['/attr/sensor'])
        nadir_resolution = self.get_nadir_resolution(sensor)
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
        var_name = ds_info.get('file_key', dataset_id.name)
        return self[var_name + '/shape']

    def get_metadata(self, data_arr, ds_info):
        i = {}
        i.update(data_arr.attrs)
        i.update(ds_info)

        flag_meanings = i.get('flag_meanings')
        if not i.get('SCALED', 1) and not flag_meanings:
            i['flag_meanings'] = '<flag_meanings_unknown>'
            i.setdefault('flag_values', [None])

        u = i.get('units')
        if u in CF_UNITS:
            # CF compliance
            i['units'] = CF_UNITS[u]

        i['sensor'] = sensor = self.get_sensor(self['/attr/sensor'])
        platform = self.get_platform(self['/attr/platform'])
        i['platform'] = i['platform_name'] = platform
        i['resolution'] = i.get('resolution') or self.get_nadir_resolution(i['sensor'])
        rps = self.get_rows_per_scan(sensor)
        if rps:
            i['rows_per_scan'] = rps
        i['reader'] = 'clavrx'

        return i

    def get_dataset(self, dataset_id, ds_info):
        var_name = ds_info.get('file_key', dataset_id.name)
        data = self[var_name]
        if dataset_id.resolution:
            data.attrs['resolution'] = dataset_id.resolution
        data.attrs = self.get_metadata(data, ds_info)
        fill = data.attrs.pop('_FillValue', None)
        factor = data.attrs.pop('scale_factor', None)
        offset = data.attrs.pop('add_offset', None)
        valid_range = data.attrs.pop('valid_range', None)

        if factor is not None and offset is not None:
            def scale_inplace(data):
                data *= factor
                data += offset
                return data
        else:
            def scale_inplace(data):
                return data

        data = data.where(data != fill)
        scale_inplace(data)
        if valid_range is not None:
            valid_min, valid_max = scale_inplace(valid_range[0]), scale_inplace(valid_range[1])
            data = data.where((data >= valid_min) & (data <= valid_max))
            data.attrs['valid_min'], data.attrs['valid_max'] = valid_min, valid_max

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
    def _read_pug_fixed_grid(projection, distance_multiplier=1.0):
        """Read from recent PUG format, where axes are in meters
        """
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

    def _find_input_nc(self, l1b_base):
        dirname = os.path.split(self.filename)[0]
        glob_pat = os.path.join(dirname, l1b_base + '*R20*.nc')
        LOG.debug("searching for {0}".format(glob_pat))
        l1b_filenames = list(glob(glob_pat))
        if not l1b_filenames:
            raise IOError("Could not find navigation donor for {0}"
                          " in same directory as CLAVR-x data".format(l1b_base))
        LOG.debug('Candidate nav donors: {0}'.format(repr(l1b_filenames)))
        return l1b_filenames[0]

    def _read_axi_fixed_grid(self, l1b_attr):
        """CLAVR-x does not transcribe fixed grid parameters to its output
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
        l1b_path = self._find_input_nc(l1b_attr)
        LOG.info("Since CLAVR-x does not include fixed-grid parameters,"
                 " using input file {0} as donor".format(l1b_path))
        l1b = netCDF4.Dataset(l1b_path)
        proj = None
        proj_var = l1b.variables.get("Projection", None)
        if proj_var is not None:
            # hsd2nc input typically used by CLAVR-x uses old-form km for axes/height
            LOG.debug("found hsd2nc-style draft PUG fixed grid specification")
            proj = self._read_pug_fixed_grid(proj_var, 1000.0)
        if proj is None:  # most likely to come into play for ABI cases
            proj_var = l1b.variables.get("goes_imager_projection", None)
            if proj_var is not None:
                LOG.debug("found cmip-style final PUG fixed grid specification")
                proj = self._read_pug_fixed_grid(proj_var)
        if not proj:
            raise ValueError("Unable to recover projection information"
                             " for {0}".format(self.filename))

        h = float(proj['h'])
        x, y = l1b['x'], l1b['y']
        area_extent, ncols, nlines = self._area_extent(x, y, h)

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

    def _is_polar(self):
        l1b_att, inst_att = (str(self.file_content.get('/attr/L1B', None)),
                             str(self.file_content.get('/attr/sensor', None)))

        return (inst_att != 'AHI') or (l1b_att is None)

    def get_area_def(self, key):
        """Get the area definition of the data at hand."""
        if self._is_polar():  # then it doesn't have a fixed grid
            return super(CLAVRXFileHandler, self).get_area_def(key)

        l1b_att = str(self.file_content.get('/attr/L1B', None))
        return self._read_axi_fixed_grid(l1b_att)
