#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017.
#
# Author(s):
#
#   David Hoese <david.hoese@ssec.wisc.edu>
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
"""Interface to GEOCAT HDF4 or NetCDF4 products.

Note: GEOCAT files do not currently have projection information or precise
pixel resolution information. Additionally the longitude and latitude arrays
are stored as 16-bit integers which causes loss of precision. For this reason
the lon/lats can't be used as a reliable coordinate system to calculate the
projection X/Y coordinates.

Until GEOCAT adds projection information and X/Y coordinate arrays, this
reader will estimate the geostationary area the best it can. It currently
takes a single lon/lat point as reference and uses hardcoded resolution
and projection information to calculate the area extents.

"""
import logging
import numpy as np
from pyproj import Proj
from pyresample import geometry
from pyresample.utils import proj4_str_to_dict

from satpy.dataset import DatasetID
from satpy.readers.yaml_reader import FileYAMLReader
from satpy.readers.netcdf_utils import NetCDF4FileHandler, netCDF4

LOG = logging.getLogger(__name__)


CF_UNITS = {
    'none': '1',
}

# GEOCAT currently doesn't include projection information in it's files
GEO_PROJS = {
    'GOES-16': '+proj=geos +lon_0={lon_0:0.02f} +h=35786023.0 +a=6378137.0 +b=6356752.31414 +sweep=x +units=m +no_defs',
    'HIMAWARI-8': '+proj=geos +over +lon_0=140.7 +h=35785863 +a=6378137 +b=6356752.299581327 +units=m +no_defs',
}


class GEOCATFileHandler(NetCDF4FileHandler):
    sensors = {
        'goes': 'goes_imager',
        'himawari8': 'ahi',
        'goes16': 'abi',  # untested
        'goesr': 'abi',  # untested
    }
    platforms = {
    }
    resolutions = {
        'abi': {
            1: 1002.0086577437705,
            2: 2004.0173154875411,
        },
        'ahi': {
            1: 999.9999820317674,  # assumption
            2: 1999.999964063535,
        }
    }

    def get_sensor(self, sensor):
        last_resort = None
        for k, v in self.sensors.items():
            if k == sensor:
                return v
            elif k in sensor:
                last_resort = v
        if last_resort:
            return last_resort
        raise ValueError("Unknown sensor '{}'".format(sensor))

    def get_platform(self, platform):
        for k, v in self.platforms.items():
            if k in platform:
                return v
        return platform

    def _get_proj(self, platform, ref_lon):
        if platform == 'GOES-16' and -76. < ref_lon < -74.:
            # geocat file holds the *actual* subsatellite point, not the
            # projection (-75.2 actual versus -75 projection)
            ref_lon = -75.
        return GEO_PROJS[platform].format(lon_0=ref_lon)

    @property
    def start_time(self):
        return self.filename_info['start_time']

    @property
    def end_time(self):
        return self.filename_info.get('end_time', self.start_time)

    @property
    def is_geo(self):
        platform = self.get_platform(self['/attr/Platform_Name'])
        return platform in GEO_PROJS

    def available_dataset_ids(self):
        """Automatically determine datasets provided by this file"""
        elem_res = self['/attr/Element_Resolution']
        sensor = self.get_sensor(self['/attr/Sensor_Name'])
        res = self.resolutions.get(sensor, {}).get(int(elem_res), elem_res * 1000.)
        coordinates = ['pixel_longitude', 'pixel_latitude']
        for var_name, val in self.file_content.items():
            if isinstance(val, netCDF4.Variable):
                ds_info = {
                    'file_type': self.filetype_info['file_type'],
                    'resolution': res,
                }
                if not self.is_geo:
                    ds_info['coordinates'] = coordinates
                yield DatasetID(name=var_name, resolution=res), ds_info

    def get_shape(self, dataset_id, ds_info):
        var_name = ds_info.get('file_key', dataset_id.name)
        return self[var_name + '/shape']

    def _first_good_nav(self, lon_arr, lat_arr):
        if hasattr(lon_arr, 'mask'):
            good_indexes = np.nonzero(~lon_arr.mask)
        else:
            # no masked values found in auto maskandscale
            good_indexes = ([0], [0])
        # nonzero returns (<ndarray of row indexes>, <ndarray of col indexes>)
        return tuple(x[0] for x in good_indexes)

    def _get_extents(self, proj, res, lon_arr, lat_arr):
        p = Proj(proj)
        res = float(res)
        first_good = self._first_good_nav(lon_arr, lat_arr)
        one_x, one_y = p(lon_arr[first_good], lat_arr[first_good])
        left_x = one_x - res * first_good[1]
        right_x = left_x + res * lon_arr.shape[1]
        top_y = one_y + res * first_good[0]
        bot_y = top_y - res * lon_arr.shape[0]
        half_x = res / 2.
        half_y = res / 2.
        return (left_x - half_x,
                bot_y - half_y,
                right_x + half_x,
                top_y + half_y)

    def _load_nav(self, name):
        nav = self[name]
        factor = self[name + '/attr/scale_factor']
        offset = self[name + '/attr/add_offset']
        fill = self[name + '/attr/_FillValue']
        nav = nav[:]
        mask = nav == fill
        nav = np.ma.masked_array(nav * factor + offset, mask=mask)
        return nav[:]

    def get_area_def(self, dsid):
        if not self.is_geo:
            raise NotImplementedError("Don't know how to get the Area Definition for this file")

        platform = self.get_platform(self['/attr/Platform_Name'])
        res = dsid.resolution
        proj = self._get_proj(platform, float(self['/attr/Subsatellite_Longitude']))
        area_name = '{} {} Area at {}m'.format(
            platform,
            self.metadata.get('sector_id', ''),
            int(res))
        lon = self._load_nav('pixel_longitude')
        lat = self._load_nav('pixel_latitude')
        extents = self._get_extents(proj, res, lon, lat)
        area_def = geometry.AreaDefinition(
            area_name,
            area_name,
            area_name,
            proj_dict=proj4_str_to_dict(proj),
            x_size=lon.shape[1],
            y_size=lon.shape[0],
            area_extent=extents,
        )
        return area_def

    def get_metadata(self, dataset_id, ds_info):
        var_name = ds_info.get('file_key', dataset_id.name)
        i = {}
        i.update(ds_info)
        for a in ['standard_name', 'units', 'long_name', 'flag_meanings', 'flag_values', 'flag_masks']:
            attr_path = var_name + '/attr/' + a
            if attr_path in self:
                i[a] = self[attr_path]

        u = i.get('units')
        if u in CF_UNITS:
            # CF compliance
            i['units'] = CF_UNITS[u]

        i['sensor'] = self.get_sensor(self['/attr/Sensor_Name'])
        i['platform'] = self.get_platform(self['/attr/Platform_Name'])
        i['resolution'] = dataset_id.resolution
        if var_name == 'pixel_longitude':
            i['standard_name'] = 'longitude'
        elif var_name == 'pixel_latitude':
            i['standard_name'] = 'latitude'

        return i

    def get_dataset(self, dataset_id, ds_info, out=None,
                    xslice=slice(None), yslice=slice(None)):
        var_name = ds_info.get('file_key', dataset_id.name)
        # FUTURE: Metadata retrieval may be separate
        i = self.get_metadata(dataset_id, ds_info)
        data = self[var_name][yslice, xslice]
        fill = self[var_name + '/attr/_FillValue']
        factor = self.get(var_name + '/attr/scale_factor')
        offset = self.get(var_name + '/attr/add_offset')
        valid_range = self.get(var_name + '/attr/valid_range')

        mask = data == fill
        if valid_range is not None:
            mask |= (data < valid_range[0]) | (data > valid_range[1])
        data = data.astype(out.data.dtype)
        if factor is not None and offset is not None:
            data *= factor
            data += offset

        out.data[:] = data
        out.mask[:] |= mask
        out.info.update(i)
        return out


class GEOCATYAMLReader(FileYAMLReader):
    def create_filehandlers(self, filenames):
        super(GEOCATYAMLReader, self).create_filehandlers(filenames)
        self.load_ds_ids_from_files()

    def load_ds_ids_from_files(self):
        for file_handlers in self.file_handlers.values():
            fh = file_handlers[0]
            for ds_id, ds_info in fh.available_dataset_ids():
                # don't overwrite an existing dataset
                # especially from the yaml config
                self.ids.setdefault(ds_id, ds_info)

