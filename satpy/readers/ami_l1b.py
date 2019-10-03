#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Satpy developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Advanced Meteorological Imager reader for the Level 1b NetCDF4 format."""

import logging
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import dask.array as da
import pyproj

from pyresample import geometry
from pyspectral.blackbody import blackbody_wn_rad2temp as rad2temp
from satpy.readers.file_handlers import BaseFileHandler
from satpy import CHUNK_SIZE

logger = logging.getLogger(__name__)

PLATFORM_NAMES = {
    'GK-2A': 'GEO-KOMPSAT-2A',
    'GK-2B': 'GEO-KOMPSAT-2B',
}

# Copied from 20190415_GK-2A_AMI_Conversion_Table_v3.0.xlsx
# Sheet: coeff.& equation_WN
# Visible channels
# channel_name -> (DN2Rad_Gain, DN2Rad_Offset, Rad. to Albedo)
# IR channels
# channel_name -> (DN2Rad_Gain, DN2Rad_Offset, c0, c1, c2)
CALIBRATION_COEFFS = {
    "VI004": (0.363545805215835, -7.27090454101562, 0.001558245),
    "VI005": (0.343625485897064, -6.87249755859375, 0.0016595767),
    "VI006": (0.154856294393539, -6.19424438476562, 0.001924484),
    "VI008": (0.0457241721451282, -3.65792846679687, 0.0032723873),
    "NR013": (0.0346878096461296, -1.38751220703125, 0.0087081313),
    "NR016": (0.0498007982969284, -0.996017456054687, 0.0129512876),
    "SW038": (-0.00108296517282724, 17.699987411499, -0.447843939824124, 1.00065568090389, -0.0000000633824089912448),
    "WV063": (-0.0108914673328399, 44.1777038574218, -1.76279494011147, 1.00414910562278, -0.000000983310914319385),
    "WV069": (-0.00818779878318309, 66.7480773925781, -0.334311414359106, 1.00097359874468, -0.000000494603070252304),
    "WV073": (-0.0096982717514038, 79.0608520507812, -0.0613124859696595, 1.00019008722941, -0.000000105863656750499),
    "IR087": (-0.0144806550815701, 118.050903320312, -0.141418528203155, 1.00052232906885, -0.00000036287276076109),
    "IR096": (-0.0178435463458299, 145.464874267578, -0.114017728158198, 1.00047380585402, -0.000000374931509928403),
    "IR105": (-0.0198196955025196, 161.580139160156, -0.142866448475177, 1.00064069572049, -0.000000550443294960498),
    "IR112": (-0.0216744858771562, 176.713439941406, -0.249111718496148, 1.00121166873756, -0.00000113167964011665),
    "IR123": (-0.023379972204566, 190.649627685546, -0.458113885722738, 1.00245520975535, -0.00000253064314720476),
    "IR133": (-0.0243037566542625, 198.224365234375, -0.0938521568527657, 1.00053982112966, -0.000000594913715312849),
}


class AMIL1bNetCDF(BaseFileHandler):
    """Base reader for AMI L1B NetCDF4 files."""

    def __init__(self, filename, filename_info, filetype_info, allow_conditional_pixels=False):
        """Open the NetCDF file with xarray and prepare the Dataset for reading."""
        super(AMIL1bNetCDF, self).__init__(filename, filename_info, filetype_info)
        self.nc = xr.open_dataset(self.filename,
                                  decode_cf=True,
                                  mask_and_scale=False,
                                  chunks={'dim_image_x': CHUNK_SIZE, 'dim_image_y': CHUNK_SIZE})
        self.nc = self.nc.rename_dims({'dim_image_x': 'x', 'dim_image_y': 'y'})

        platform_shortname = self.nc.attrs['satellite_name']
        self.platform_name = PLATFORM_NAMES.get(platform_shortname)
        self.sensor = 'ami'
        self.allow_conditional_pixels = allow_conditional_pixels

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
        a = self.nc.attrs['earth_equatorial_radius']
        b = self.nc.attrs['earth_polar_radius']
        h = self.nc.attrs['nominal_satellite_height'] - a
        lon_0 = self.nc.attrs['sub_longitude'] * 180 / np.pi  # it's in radians?
        cols = self.nc.attrs['number_of_columns']
        rows = self.nc.attrs['number_of_lines']
        obs_mode = self.nc.attrs['observation_mode']
        resolution = self.nc.attrs['channel_spatial_resolution']

        cfac = self.nc.attrs['cfac']
        coff = self.nc.attrs['coff']
        lfac = self.nc.attrs['lfac']
        loff = self.nc.attrs['loff']
        bit_shift = 2**16
        area_extent = (
            h * np.deg2rad((0 - coff - 0.5) * bit_shift / cfac),
            h * np.deg2rad((0 - loff - 0.5) * bit_shift / lfac),
            h * np.deg2rad((cols - coff + 0.5) * bit_shift / cfac),
            h * np.deg2rad((rows - loff + 0.5) * bit_shift / lfac),
        )

        proj_dict = {
            'proj': 'geos',
            'lon_0': float(lon_0),
            'a': float(a),
            'b': float(b),
            'h': h,
            'units': 'm'
        }

        fg_area_def = geometry.AreaDefinition(
            'ami_geos_{}'.format(obs_mode.lower()),
            'AMI {} Area at {} resolution'.format(obs_mode, resolution),
            'ami_fixed_grid',
            proj_dict,
            cols,
            rows,
            np.asarray(area_extent))

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
            'satellite_nominal_longitude': sc_position[0],
            'satellite_nominal_latitude': sc_position[1],
            'satellite_nominal_altitude': sc_position[2] / 1000.0,  # km
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
        # noticing better results for some bands when using:
        # data &= 2**14 - 1
        # only take "no error" pixels as valid
        data = data.where(qf == 0)

        channel_name = attrs.get('channel_name', dataset_id.name)
        coeffs = CALIBRATION_COEFFS.get(channel_name)
        if coeffs is None and dataset_id.calibration is not None:
            raise ValueError("No coefficients configured for {}".format(dataset_id))
        if dataset_id.calibration in ('radiance', 'reflectance', 'brightness_temperature'):
            gain = coeffs[0]
            offset = coeffs[1]
            data = gain * data + offset
        if dataset_id.calibration == 'reflectance':
            # depends on the radiance calibration above
            rad_to_alb = coeffs[2]
            if ds_info.get('units') == '%':
                rad_to_alb *= 100
            data = data * rad_to_alb
        elif dataset_id.calibration == 'brightness_temperature':
            # depends on the radiance calibration above
            # Convert um to m^-1 (SI units for pyspectral)
            wn = 1 / dataset_id.wavelength[1] / 1e6
            # Convert cm^-1 (wavenumbers) and (mW/m^2)/(str/cm^-1) (radiance data)
            # to SI units m^-1, mW*m^-3*str^-1.
            bt_data = rad2temp(wn, data.data * 1e-5)
            if isinstance(bt_data, np.ndarray):
                # old versions of pyspectral produce numpy arrays
                data.data = da.from_array(bt_data, chunks=data.data.chunks)
            else:
                # new versions of pyspectral can do dask arrays
                data.data = bt_data

        for attr_name in ('standard_name', 'units'):
            attrs[attr_name] = ds_info[attr_name]
        attrs.update(dataset_id.to_dict())
        attrs['orbital_parameters'] = self.get_orbital_parameters()
        attrs['platform_name'] = self.platform_name
        attrs['sensor'] = self.sensor
        data.attrs = attrs
        return data
