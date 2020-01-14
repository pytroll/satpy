#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 Satpy developers
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
"""Advanced Meteorological Imager reader for the Level 1b NetCDF4 format."""

import logging
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import dask.array as da
import pyproj

from satpy.readers._geos_area import get_area_definition, get_area_extent
from pyspectral.blackbody import blackbody_wn_rad2temp as rad2temp
from satpy.readers.file_handlers import BaseFileHandler
from satpy import CHUNK_SIZE

logger = logging.getLogger(__name__)

PLATFORM_NAMES = {
    'GK-2A': 'GEO-KOMPSAT-2A',
    'GK-2B': 'GEO-KOMPSAT-2B',
}


class AMIL1bNetCDF(BaseFileHandler):
    """Base reader for AMI L1B NetCDF4 files."""

    def __init__(self, filename, filename_info, filetype_info,
                 calib_mode='PYSPECTRAL', allow_conditional_pixels=False):
        """Open the NetCDF file with xarray and prepare the Dataset for reading."""
        super(AMIL1bNetCDF, self).__init__(filename, filename_info, filetype_info)
        self.nc = xr.open_dataset(self.filename,
                                  decode_cf=True,
                                  mask_and_scale=False,
                                  chunks={'dim_image_x': CHUNK_SIZE, 'dim_image_y': CHUNK_SIZE})
        self.nc = self.nc.rename({'dim_image_x': 'x', 'dim_image_y': 'y'})

        platform_shortname = self.nc.attrs['satellite_name']
        self.platform_name = PLATFORM_NAMES.get(platform_shortname)
        self.sensor = 'ami'
        self.allow_conditional_pixels = allow_conditional_pixels
        calib_mode_choices = ('FILE', 'PYSPECTRAL')
        if calib_mode.upper() not in calib_mode_choices:
            raise ValueError('Invalid calibration mode: {}. Choose one of {}'.format(
                calib_mode, calib_mode_choices))
        self.calib_mode = calib_mode.upper()

    @property
    def start_time(self):
        """Get observation start time."""
        base = datetime(2000, 1, 1, 12, 0, 0)
        return base + timedelta(seconds=self.nc.attrs['observation_start_time'])

    @property
    def end_time(self):
        """Get observation end time."""
        base = datetime(2000, 1, 1, 12, 0, 0)
        return base + timedelta(seconds=self.nc.attrs['observation_end_time'])

    def get_area_def(self, dsid):
        """Get area definition for this file."""
        pdict = {}
        pdict['a'] = self.nc.attrs['earth_equatorial_radius']
        pdict['b'] = self.nc.attrs['earth_polar_radius']
        pdict['h'] = self.nc.attrs['nominal_satellite_height'] - pdict['a']
        pdict['ssp_lon'] = self.nc.attrs['sub_longitude'] * 180 / np.pi  # it's in radians?
        pdict['ncols'] = self.nc.attrs['number_of_columns']
        pdict['nlines'] = self.nc.attrs['number_of_lines']
        obs_mode = self.nc.attrs['observation_mode']
        resolution = self.nc.attrs['channel_spatial_resolution']

        # Example offset: 11000.5
        # the 'get_area_extent' will handle this half pixel for us
        pdict['cfac'] = self.nc.attrs['cfac']
        pdict['coff'] = self.nc.attrs['coff']
        pdict['lfac'] = -self.nc.attrs['lfac']
        pdict['loff'] = self.nc.attrs['loff']
        pdict['scandir'] = 'N2S'
        pdict['a_name'] = 'ami_geos_{}'.format(obs_mode.lower())
        pdict['a_desc'] = 'AMI {} Area at {} resolution'.format(obs_mode, resolution)
        pdict['p_id'] = 'ami_fixed_grid'

        area_extent = get_area_extent(pdict)
        fg_area_def = get_area_definition(pdict, area_extent)
        return fg_area_def

    def get_orbital_parameters(self):
        """Collect orbital parameters for this file."""
        a = float(self.nc.attrs['earth_equatorial_radius'])
        b = float(self.nc.attrs['earth_polar_radius'])
        # nominal_satellite_height seems to be from the center of the earth
        h = float(self.nc.attrs['nominal_satellite_height']) - a
        lon_0 = self.nc.attrs['sub_longitude'] * 180 / np.pi  # it's in radians?
        sc_position = self.nc['sc_position'].attrs['sc_position_center_pixel']

        # convert ECEF coordinates to lon, lat, alt
        ecef = pyproj.Proj(proj='geocent', a=a, b=b)
        lla = pyproj.Proj(proj='latlong', a=a, b=b)
        sc_position = pyproj.transform(
            ecef, lla, sc_position[0], sc_position[1], sc_position[2])

        orbital_parameters = {
            'projection_longitude': float(lon_0),
            'projection_latitude': 0.0,
            'projection_altitude': h,
            'satellite_actual_longitude': sc_position[0],
            'satellite_actual_latitude': sc_position[1],
            'satellite_actual_altitude': sc_position[2],  # meters
        }
        return orbital_parameters

    def get_dataset(self, dataset_id, ds_info):
        """Load a dataset as a xarray DataArray."""
        file_key = ds_info.get('file_key', dataset_id.name)
        data = self.nc[file_key]
        # hold on to attributes for later
        attrs = data.attrs
        # highest 2 bits are data quality flags
        # 00=no error
        # 01=available under conditions
        # 10=outside the viewing area
        # 11=Error exists
        if self.allow_conditional_pixels:
            qf = data & 0b1000000000000000
        else:
            qf = data & 0b1100000000000000

        # mask DQF bits
        bits = attrs['number_of_valid_bits_per_pixel']
        data &= 2**bits - 1
        # only take "no error" pixels as valid
        data = data.where(qf == 0)

        # Calibration values from file, fall back to built-in if unavailable
        gain = self.nc.attrs['DN_to_Radiance_Gain']
        offset = self.nc.attrs['DN_to_Radiance_Offset']

        if dataset_id.calibration in ('radiance', 'reflectance', 'brightness_temperature'):
            data = gain * data + offset
        if dataset_id.calibration == 'reflectance':
            # depends on the radiance calibration above
            rad_to_alb = self.nc.attrs['Radiance_to_Albedo_c']
            if ds_info.get('units') == '%':
                rad_to_alb *= 100
            data = data * rad_to_alb
        elif dataset_id.calibration == 'brightness_temperature':
            data = self._calibrate_ir(dataset_id, data)
        elif dataset_id.calibration not in ('counts', 'radiance'):
            raise ValueError("Unknown calibration: '{}'".format(dataset_id.calibration))

        for attr_name in ('standard_name', 'units'):
            attrs[attr_name] = ds_info[attr_name]
        attrs.update(dataset_id.to_dict())
        attrs['orbital_parameters'] = self.get_orbital_parameters()
        attrs['platform_name'] = self.platform_name
        attrs['sensor'] = self.sensor
        data.attrs = attrs
        return data

    def _calibrate_ir(self, dataset_id, data):
        """Calibrate radiance data to BTs using either pyspectral or in-file coefficients."""
        if self.calib_mode == 'PYSPECTRAL':
            # depends on the radiance calibration above
            # Convert um to m^-1 (SI units for pyspectral)
            wn = 1 / (dataset_id.wavelength[1] / 1e6)
            # Convert cm^-1 (wavenumbers) and (mW/m^2)/(str/cm^-1) (radiance data)
            # to SI units m^-1, mW*m^-3*str^-1.
            bt_data = rad2temp(wn, data.data * 1e-5)
            if isinstance(bt_data, np.ndarray):
                # old versions of pyspectral produce numpy arrays
                data.data = da.from_array(bt_data, chunks=data.data.chunks)
            else:
                # new versions of pyspectral can do dask arrays
                data.data = bt_data
        else:
            # IR coefficients from the file
            # Channel specific
            c0 = self.nc.attrs['Teff_to_Tbb_c0']
            c1 = self.nc.attrs['Teff_to_Tbb_c1']
            c2 = self.nc.attrs['Teff_to_Tbb_c2']

            # These should be fixed, but load anyway
            cval = self.nc.attrs['light_speed']
            kval = self.nc.attrs['Boltzmann_constant_k']
            hval = self.nc.attrs['Plank_constant_h']

            # Compute wavenumber as cm-1
            wn = (10000 / dataset_id.wavelength[1]) * 100

            # Convert radiance to effective brightness temperature
            e1 = (2 * hval * cval * cval) * np.power(wn, 3)
            e2 = (data.data * 1e-5)
            t_eff = ((hval * cval / kval) * wn) / np.log((e1 / e2) + 1)

            # Now convert to actual brightness temperature
            bt_data = c0 + c1 * t_eff + c2 * t_eff * t_eff
            data.data = bt_data
        return data
