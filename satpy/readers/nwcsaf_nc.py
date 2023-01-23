#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2023 Satpy developers
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
"""Nowcasting SAF common PPS&MSG NetCDF/CF format reader.

References:
   - The NWCSAF GEO 2018 products documentation: http://www.nwcsaf.org/web/guest/archive

"""

import functools
import logging
import os
from contextlib import suppress
from datetime import datetime

import dask.array as da
import numpy as np
import xarray as xr
from pyproj import CRS
from pyresample.geometry import AreaDefinition

from satpy import CHUNK_SIZE
from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.utils import unzip_file

logger = logging.getLogger(__name__)

SENSOR = {'NOAA-19': 'avhrr-3',
          'NOAA-18': 'avhrr-3',
          'NOAA-15': 'avhrr-3',
          'Metop-A': 'avhrr-3',
          'Metop-B': 'avhrr-3',
          'Metop-C': 'avhrr-3',
          'EOS-Aqua': 'modis',
          'EOS-Terra': 'modis',
          'Suomi-NPP': 'viirs',
          'NOAA-20': 'viirs',
          'JPSS-1': 'viirs',
          'GOES-16': 'abi',
          'GOES-17': 'abi',
          'Himawari-8': 'ahi',
          'Himawari-9': 'ahi',
          }


PLATFORM_NAMES = {'MSG1': 'Meteosat-8',
                  'MSG2': 'Meteosat-9',
                  'MSG3': 'Meteosat-10',
                  'MSG4': 'Meteosat-11',
                  'GOES16': 'GOES-16',
                  'GOES17': 'GOES-17',
                  }


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
        self.platform_name = None
        self.sensor = None
        self.file_key_prefix = filetype_info.get("file_key_prefix", "")

        try:
            # NWCSAF/Geo:
            try:
                kwrgs = {'sat_id': self.nc.attrs['satellite_identifier']}
            except KeyError:
                kwrgs = {'sat_id': self.nc.attrs['satellite_identifier'].astype(str)}
        except KeyError:
            # NWCSAF/PPS:
            kwrgs = {'platform_name': self.nc.attrs['platform']}

        self.set_platform_and_sensor(**kwrgs)

        self.upsample_geolocation = functools.lru_cache(maxsize=1)(
            self._upsample_geolocation_uncached
        )

    def set_platform_and_sensor(self, **kwargs):
        """Set some metadata: platform_name, sensors, and pps (identifying PPS or Geo)."""
        try:
            # NWCSAF/Geo
            self.platform_name = PLATFORM_NAMES.get(kwargs['sat_id'], kwargs['sat_id'])
        except KeyError:
            # NWCSAF/PPS
            self.platform_name = kwargs['platform_name']
            self.pps = True

        self.sensor = set([SENSOR.get(self.platform_name, 'seviri')])

    def remove_timedim(self, var):
        """Remove time dimension from dataset."""
        if self.pps and var.dims[0] == 'time':
            data = var[0, :, :]
            data.attrs = var.attrs
            var = data
        return var

    def drop_xycoords(self, variable):
        """Drop x, y coords when y is scan line number."""
        try:
            if variable.coords['y'].attrs['long_name'] == "scan line number":
                return variable.drop_vars(['y', 'x'])
        except KeyError:
            pass
        return variable

    def get_dataset(self, dsid, info):
        """Load a dataset."""
        dsid_name = dsid['name']
        if dsid_name in self.cache:
            logger.debug('Get the data set from cache: %s.', dsid_name)
            return self.cache[dsid_name]
        if dsid_name in ['lon', 'lat'] and dsid_name not in self.nc:
            # Get full resolution lon,lat from the reduced (tie points) grid
            lon, lat = self.upsample_geolocation()
            if dsid_name == "lon":
                return lon
            else:
                return lat

        logger.debug('Reading %s.', dsid_name)
        file_key = self._get_filekeys(dsid_name, info)
        variable = self.nc[file_key]
        variable = self.remove_timedim(variable)
        variable = self.scale_dataset(variable, info)
        variable = self.drop_xycoords(variable)

        self.get_orbital_parameters(variable)
        variable.attrs["start_time"] = self.start_time
        variable.attrs["end_time"] = self.end_time

        return variable

    def get_orbital_parameters(self, variable):
        """Get the orbital parameters from the file if possible (geo)."""
        with suppress(KeyError):
            gdal_params = dict(elt.strip("+").split("=") for elt in self.nc.attrs["gdal_projection"].split())
            variable.attrs["orbital_parameters"] = dict(
                satellite_nominal_altitude=float(gdal_params["h"]),
                satellite_nominal_longitude=float(self.nc.attrs["sub-satellite_longitude"]),
                satellite_nominal_latitude=0)

    def _get_varname_in_file(self, info, info_type="file_key"):
        if isinstance(info[info_type], list):
            for key in info[info_type]:
                file_key = self.file_key_prefix + key
                if file_key in self.nc:
                    return file_key
        return self.file_key_prefix + info[info_type]

    def _get_filekeys(self, dsid_name, info):
        try:
            file_key = self._get_varname_in_file(info, info_type="file_key")
        except KeyError:
            file_key = dsid_name
        return file_key

    def scale_dataset(self, variable, info):
        """Scale the data set, applying the attributes from the netCDF file.

        The scale and offset attributes will then be removed from the resulting variable.
        """
        variable = remove_empties(variable)

        scale = variable.attrs.get('scale_factor', np.array(1))
        offset = variable.attrs.get('add_offset', np.array(0))
        if np.issubdtype((scale + offset).dtype, np.floating) or np.issubdtype(variable.dtype, np.floating):
            variable = self._mask_variable(variable)
        attrs = variable.attrs.copy()
        variable = variable * scale + offset
        variable.attrs = attrs
        if 'valid_range' in variable.attrs:
            variable.attrs['valid_range'] = variable.attrs['valid_range'] * scale + offset

        variable.attrs.pop('add_offset', None)
        variable.attrs.pop('scale_factor', None)

        variable.attrs.update({'platform_name': self.platform_name,
                               'sensor': self.sensor})

        if not variable.attrs.get('standard_name', '').endswith('status_flag'):
            # TODO: do we really need to add units to everything ?
            variable.attrs.setdefault('units', '1')

        ancillary_names = variable.attrs.get('ancillary_variables', '')
        try:
            variable.attrs['ancillary_variables'] = ancillary_names.split()
        except AttributeError:
            pass

        if 'palette_meanings' in variable.attrs:
            variable = self._prepare_variable_for_palette(variable, info)

        if 'standard_name' in info:
            variable.attrs.setdefault('standard_name', info['standard_name'])
        variable = self._adjust_variable_for_legacy_software(variable)

        return variable

    @staticmethod
    def _mask_variable(variable):
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
        return variable

    def _prepare_variable_for_palette(self, variable, info):
        try:
            so_dataset = self.nc[self._get_varname_in_file(info, info_type='scale_offset_dataset')]
        except KeyError:
            scale = 1
            offset = 0
        else:
            scale = so_dataset.attrs['scale_factor']
            offset = so_dataset.attrs['add_offset']
        variable.attrs['palette_meanings'] = [int(val)
                                              for val in variable.attrs['palette_meanings'].split()]
        if variable.attrs['palette_meanings'][0] == 1:
            variable.attrs['palette_meanings'] = [0] + variable.attrs['palette_meanings']
            variable = xr.DataArray(da.vstack((np.array(variable.attrs['fill_value_color']), variable.data)),
                                    coords=variable.coords, dims=variable.dims, attrs=variable.attrs)
        val, idx = np.unique(variable.attrs['palette_meanings'], return_index=True)
        variable.attrs['palette_meanings'] = val * scale + offset
        variable = variable[idx]
        return variable

    def _adjust_variable_for_legacy_software(self, variable):
        if self.sw_version == 'NWC/PPS version v2014' and variable.attrs.get('standard_name') == 'cloud_top_altitude':
            # pps 2014 valid range and palette don't match
            variable.attrs['valid_range'] = (0., 9000.)
        if (self.sw_version == 'NWC/PPS version v2014' and
                variable.attrs.get('long_name') == 'RGB Palette for ctth_alti'):
            # pps 2014 palette has the nodata color (black) first
            variable = variable[1:, :]
        return variable

    def _upsample_geolocation_uncached(self):
        """Upsample the geolocation (lon,lat) from the tiepoint grid."""
        from geotiepoints import SatelliteInterpolator

        # Read the fields needed:
        col_indices = self.nc['nx_reduced'].values
        row_indices = self.nc['ny_reduced'].values
        lat_reduced = self.scale_dataset(self.nc['lat_reduced'], {})
        lon_reduced = self.scale_dataset(self.nc['lon_reduced'], {})

        shape = (self.nc['y'].shape[0], self.nc['x'].shape[0])
        cols_full = np.arange(shape[1])
        rows_full = np.arange(shape[0])

        satint = SatelliteInterpolator((lon_reduced.values, lat_reduced.values),
                                       (row_indices,
                                        col_indices),
                                       (rows_full, cols_full))

        lons, lats = satint.interpolate()
        lon = xr.DataArray(lons, attrs=lon_reduced.attrs, dims=['y', 'x'])
        lat = xr.DataArray(lats, attrs=lat_reduced.attrs, dims=['y', 'x'])
        lat = self.drop_xycoords(lat)
        lon = self.drop_xycoords(lon)
        return lon, lat

    def get_area_def(self, dsid):
        """Get the area definition of the datasets in the file.

        Only applicable for MSG products!
        """
        if self.pps:
            # PPS:
            raise NotImplementedError

        if dsid['name'].endswith('_pal'):
            raise NotImplementedError

        crs, area_extent = self._get_projection()
        crs, area_extent = self._ensure_crs_extents_in_meters(crs, area_extent)
        nlines, ncols = self.nc[dsid['name']].shape
        area = AreaDefinition('some_area_name',
                              "On-the-fly area",
                              'geosmsg',
                              crs,
                              ncols,
                              nlines,
                              area_extent)

        return area

    @staticmethod
    def _ensure_crs_extents_in_meters(crs, area_extent):
        """Fix units in Earth shape, satellite altitude and 'units' attribute."""
        if 'kilo' in crs.axis_info[0].unit_name:
            proj_dict = crs.to_dict()
            proj_dict["units"] = "m"
            if "a" in proj_dict:
                proj_dict["a"] *= 1000.
            if "b" in proj_dict:
                proj_dict["b"] *= 1000.
            if "R" in proj_dict:
                proj_dict["R"] *= 1000.
            proj_dict["h"] *= 1000.
            area_extent = tuple([val * 1000. for val in area_extent])
            crs = CRS.from_dict(proj_dict)
        return crs, area_extent

    def __del__(self):
        """Delete the instance."""
        if self._unzipped:
            try:
                os.remove(self._unzipped)
            except OSError:
                pass

    @property
    def start_time(self):
        """Return the start time of the object."""
        return read_nwcsaf_time(self.nc.attrs['time_coverage_start'])

    @property
    def end_time(self):
        """Return the end time of the object."""
        return read_nwcsaf_time(self.nc.attrs['time_coverage_end'])

    @property
    def sensor_names(self):
        """List of sensors represented in this file."""
        return self.sensor

    def _get_projection(self):
        """Get projection from the NetCDF4 attributes."""
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

        crs = CRS.from_string(proj_str)
        return crs, area_extent


def remove_empties(variable):
    """Remove empty objects from the *variable*'s attrs."""
    import h5py
    for key, val in variable.attrs.items():
        if isinstance(val, h5py._hl.base.Empty):
            variable.attrs.pop(key)

    return variable


def read_nwcsaf_time(time_value):
    """Read the time, nwcsaf-style."""
    try:
        # MSG:
        try:
            return datetime.strptime(time_value, '%Y-%m-%dT%H:%M:%SZ')
        except TypeError:  # Remove this in summer 2024 (this is not needed since h5netcdf 0.14)
            return datetime.strptime(time_value.astype(str), '%Y-%m-%dT%H:%M:%SZ')
    except ValueError:
        # PPS:
        return datetime.strptime(time_value, '%Y%m%dT%H%M%S%fZ')
