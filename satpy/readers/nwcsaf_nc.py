#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017, 2018, 2019 Pytroll

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   Adam.Dybbroe <adam.dybbroe@smhi.se>

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

"""Nowcasting SAF common PPS&MSG NetCDF/CF format reader
"""

import logging
from datetime import datetime
import os

import numpy as np
import xarray as xr
import dask.array as da

from pyresample.utils import get_area_def
from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.utils import unzip_file
from satpy import CHUNK_SIZE

logger = logging.getLogger(__name__)

SENSOR = {'NOAA-19': 'avhrr/3',
          'NOAA-18': 'avhrr/3',
          'NOAA-15': 'avhrr/3',
          'Metop-A': 'avhrr/3',
          'Metop-B': 'avhrr/3',
          'Metop-C': 'avhrr/3',
          'EOS-Aqua': 'modis',
          'EOS-Terra': 'modis',
          'Suomi-NPP': 'viirs',
          'NOAA-20': 'viirs',
          'JPSS-1': 'viirs', }

PLATFORM_NAMES = {'MSG1': 'Meteosat-8',
                  'MSG2': 'Meteosat-9',
                  'MSG3': 'Meteosat-10',
                  'MSG4': 'Meteosat-11', }


class NcNWCSAF(BaseFileHandler):

    """NWCSAF PPS&MSG NetCDF reader."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init method."""
        super(NcNWCSAF, self).__init__(filename, filename_info,
                                       filetype_info)

        self._unzipped = unzip_file(self.filename)
        if self._unzipped:
            self.filename = self._unzipped

        self.cache = {}
        self.nc = xr.open_dataset(self.filename,
                                  decode_cf=True,
                                  mask_and_scale=False,
                                  chunks=CHUNK_SIZE)

        self.nc = self.nc.rename({'nx': 'x', 'ny': 'y'})
        self.sw_version = self.nc.attrs['source']
        self.pps = False

        try:
            # MSG:
            sat_id = self.nc.attrs['satellite_identifier']
            try:
                self.platform_name = PLATFORM_NAMES[sat_id]
            except KeyError:
                self.platform_name = PLATFORM_NAMES[sat_id.astype(str)]
        except KeyError:
            # PPS:
            self.platform_name = self.nc.attrs['platform']
            self.pps = True

        self.sensor = SENSOR.get(self.platform_name, 'seviri')

    def remove_timedim(self, var):
        """Remove time dimension from dataset"""
        if self.pps and var.dims[0] == 'time':
            data = var[0, :, :]
            data.attrs = var.attrs
            var = data

        return var

    def get_dataset(self, dsid, info):
        """Load a dataset."""

        dsid_name = dsid.name
        if dsid_name in self.cache:
            logger.debug('Get the data set from cache: %s.', dsid_name)
            return self.cache[dsid_name]
        if dsid_name in ['lon', 'lat'] and dsid_name not in self.nc:
            dsid_name = dsid_name + '_reduced'

        logger.debug('Reading %s.', dsid_name)
        variable = self.nc[dsid_name]
        variable = self.remove_timedim(variable)
        variable = self.scale_dataset(dsid, variable, info)

        if dsid_name.endswith('_reduced'):
            # Get full resolution lon,lat from the reduced (tie points) grid
            self.upsample_geolocation(dsid, info)

            return self.cache[dsid.name]

        return variable

    def scale_dataset(self, dsid, variable, info):
        """Scale the data set, applying the attributes from the netCDF file"""

        variable = remove_empties(variable)
        scale = variable.attrs.get('scale_factor', np.array(1))
        offset = variable.attrs.get('add_offset', np.array(0))
        if np.issubdtype((scale + offset).dtype, np.floating) or np.issubdtype(variable.dtype, np.floating):
            if '_FillValue' in variable.attrs:
                variable = variable.where(
                    variable != variable.attrs['_FillValue'])
                variable.attrs['_FillValue'] = np.nan
            if 'valid_range' in variable.attrs:
                variable = variable.where(
                    variable <= variable.attrs['valid_range'][1])
                variable = variable.where(
                    variable >= variable.attrs['valid_range'][0])
            if 'valid_max' in variable.attrs:
                variable = variable.where(
                    variable <= variable.attrs['valid_max'])
            if 'valid_min' in variable.attrs:
                variable = variable.where(
                    variable >= variable.attrs['valid_min'])
        attrs = variable.attrs
        variable = variable * scale + offset
        variable.attrs = attrs

        variable.attrs.update({'platform_name': self.platform_name,
                               'sensor': self.sensor})

        variable.attrs.setdefault('units', '1')

        ancillary_names = variable.attrs.get('ancillary_variables', '')
        try:
            variable.attrs['ancillary_variables'] = ancillary_names.split()
        except AttributeError:
            pass

        if 'palette_meanings' in variable.attrs:
            variable.attrs['palette_meanings'] = [int(val)
                                                  for val in variable.attrs['palette_meanings'].split()]
            if variable.attrs['palette_meanings'][0] == 1:
                variable.attrs['palette_meanings'] = [0] + variable.attrs['palette_meanings']
                variable = xr.DataArray(da.vstack((np.array(variable.attrs['fill_value_color']), variable.data)),
                                        coords=variable.coords, dims=variable.dims, attrs=variable.attrs)

            val, idx = np.unique(variable.attrs['palette_meanings'], return_index=True)
            variable.attrs['palette_meanings'] = val
            variable = variable[idx]

        if 'standard_name' in info:
            variable.attrs.setdefault('standard_name', info['standard_name'])
        if self.sw_version == 'NWC/PPS version v2014' and dsid.name == 'ctth_alti':
            # pps 2014 valid range and palette don't match
            variable.attrs['valid_range'] = (0., 9000.)
        if self.sw_version == 'NWC/PPS version v2014' and dsid.name == 'ctth_alti_pal':
            # pps 2014 palette has the nodata color (black) first
            variable = variable[1:, :]

        return variable

    def upsample_geolocation(self, dsid, info):
        """Upsample the geolocation (lon,lat) from the tiepoint grid"""
        from geotiepoints import SatelliteInterpolator
        # Read the fields needed:
        col_indices = self.nc['nx_reduced'].values
        row_indices = self.nc['ny_reduced'].values
        lat_reduced = self.scale_dataset(dsid, self.nc['lat_reduced'], info)
        lon_reduced = self.scale_dataset(dsid, self.nc['lon_reduced'], info)

        shape = (self.nc['y'].shape[0], self.nc['x'].shape[0])
        cols_full = np.arange(shape[1])
        rows_full = np.arange(shape[0])

        satint = SatelliteInterpolator((lon_reduced.values, lat_reduced.values),
                                       (row_indices,
                                        col_indices),
                                       (rows_full, cols_full))

        lons, lats = satint.interpolate()
        self.cache['lon'] = xr.DataArray(lons, attrs=lon_reduced.attrs, dims=['y', 'x'])
        self.cache['lat'] = xr.DataArray(lats, attrs=lat_reduced.attrs, dims=['y', 'x'])

        return

    def get_area_def(self, dsid):
        """Get the area definition of the datasets in the file.

        Only applicable for MSG products!
        """
        if self.pps:
            # PPS:
            raise NotImplementedError

        if dsid.name.endswith('_pal'):
            raise NotImplementedError

        proj_str, area_extent = self._get_projection()

        nlines, ncols = self.nc[dsid.name].shape

        area = get_area_def('some_area_name',
                            "On-the-fly area",
                            'geosmsg',
                            proj_str,
                            ncols,
                            nlines,
                            area_extent)

        return area

    def __del__(self):
        if self._unzipped:
            try:
                os.remove(self._unzipped)
            except (IOError, OSError):
                pass

    @property
    def start_time(self):
        """Return the start time of the object."""
        try:
            # MSG:
            try:
                return datetime.strptime(self.nc.attrs['time_coverage_start'],
                                         '%Y-%m-%dT%H:%M:%SZ')
            except TypeError:
                return datetime.strptime(self.nc.attrs['time_coverage_start'].astype(str),
                                         '%Y-%m-%dT%H:%M:%SZ')
        except ValueError:
            # PPS:
            return datetime.strptime(self.nc.attrs['time_coverage_start'],
                                     '%Y%m%dT%H%M%S%fZ')

    @property
    def end_time(self):
        """Return the end time of the object."""
        try:
            # MSG:
            try:
                return datetime.strptime(self.nc.attrs['time_coverage_end'],
                                         '%Y-%m-%dT%H:%M:%SZ')
            except TypeError:
                return datetime.strptime(self.nc.attrs['time_coverage_end'].astype(str),
                                         '%Y-%m-%dT%H:%M:%SZ')
        except ValueError:
            # PPS:
            return datetime.strptime(self.nc.attrs['time_coverage_end'],
                                     '%Y%m%dT%H%M%S%fZ')

    def _get_projection(self):
        """Get projection from the NetCDF4 attributes"""
        try:
            proj_str = self.nc.attrs['gdal_projection']
        except TypeError:
            proj_str = self.nc.attrs['gdal_projection'].decode()

        # Check the a/b/h units
        radius_a = proj_str.split('+a=')[-1].split()[0]
        if float(radius_a) > 10e3:
            units = 'm'
            scale = 1.0
        else:
            units = 'km'
            scale = 1e3

        if 'units' not in proj_str:
            proj_str = proj_str + ' +units=' + units

        area_extent = (float(self.nc.attrs['gdal_xgeo_up_left']) / scale,
                       float(self.nc.attrs['gdal_ygeo_low_right']) / scale,
                       float(self.nc.attrs['gdal_xgeo_low_right']) / scale,
                       float(self.nc.attrs['gdal_ygeo_up_left']) / scale)

        return proj_str, area_extent


def remove_empties(variable):
    """Remove empty objects from the *variable*'s attrs."""
    import h5py
    for key, val in variable.attrs.items():
        if isinstance(val, h5py._hl.base.Empty):
            variable.attrs.pop(key)

    return variable
