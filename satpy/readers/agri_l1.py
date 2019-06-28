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
"""Advanced Geostationary Radiation Imager reader for the Level_1 HDF format

The files read by this reader are described in the official Real Time Data Service:

    http://fy4.nsmc.org.cn/data/en/data/realtime.html

"""

import logging
import numpy as np
import xarray as xr
import dask.array as da
from datetime import datetime
from pyresample import geometry
from satpy.readers.hdf5_utils import HDF5FileHandler

logger = logging.getLogger(__name__)

# info of 500 m, 1 km, 2 km and 4 km data
_resolution_list = [500, 1000, 2000, 4000]
_COFF_list = [10991.5, 5495.5, 2747.5, 1373.5]
_CFAC_list = [81865099.0, 40932549.0, 20466274.0, 10233137.0]
_LOFF_list = [10991.5, 5495.5, 2747.5, 1373.5]
_LFAC_list = [81865099.0, 40932549.0, 20466274.0, 10233137.0]


class HDF_AGRI_L1(HDF5FileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(HDF_AGRI_L1, self).__init__(filename, filename_info, filetype_info)

    def get_dataset(self, dataset_id, ds_info):
        """Load a dataset."""
        logger.debug('Reading in get_dataset %s.', dataset_id.name)
        file_key = ds_info.get('file_key', dataset_id.name)
        lut_key = ds_info.get('lut_key', dataset_id.name)
        data = self.get(file_key)
        lut = self.get(lut_key)
        if data.ndim >= 2:
            data = data.rename({data.dims[-2]: 'y', data.dims[-1]: 'x'})

        # convert bytes to string
        data.attrs['long_name'] = data.attrs['long_name'].decode('gbk')
        data.attrs['band_names'] = data.attrs['band_names'].decode('gbk')
        if ds_info['file_type'] != 'agri_l1_4000m_geo':
            data.attrs['center_wavelength'] = data.attrs['center_wavelength'].decode('gbk')

        # calibration
        calibration = ds_info['calibration']

        if calibration == 'counts':
            data.attrs['units'] = ds_info['units']
            ds_info['valid_range'] = data.attrs['valid_range']
            return data

        elif calibration in ['reflectance', 'radiance']:
            logger.debug("Calibrating to reflectances")
            # using the corresponding SCALE and OFFSET
            cal_coef = 'CALIBRATION_COEF(SCALE+OFFSET)'
            num_channel = self.get(cal_coef).shape[0]

            if num_channel == 1:
                # only channel_2, resolution = 500 m
                slope = self.get(cal_coef)[0, 0]
                offset = self.get(cal_coef)[0, 1]
            else:
                slope = self.get(cal_coef)[int(file_key[-2:])-1, 0]
                offset = self.get(cal_coef)[int(file_key[-2:])-1, 1]

            data = self.dn2(data, calibration, slope, offset)

            if calibration == 'reflectance':
                ds_info['valid_range'] = (xr.DataArray(data.attrs['valid_range']) * slope + offset) * 100
            else:
                ds_info['valid_range'] = (xr.DataArray(data.attrs['valid_range']) * slope + offset)

        elif calibration == 'brightness_temperature':
            logger.debug("Calibrating to brightness_temperature")
            # the value of dn is the index of brightness_temperature
            data = self.calibrate(data, lut)
            ds_info['valid_range'] = lut.attrs['valid_range']

        data.attrs.update({'platform_name': self['/attr/Satellite Name'],
                           'sensor': self['/attr/Sensor Identification Code'],
                           'orbital_parameters': {
                           'satellite_nominal_latitude': self['/attr/NOMCenterLat'],
                           'satellite_nominal_longitude': self['/attr/NOMCenterLon'],
                           'satellite_nominal_altitude': self['/attr/NOMSatHeight']}})
        data.attrs.update(ds_info)

        # remove attributes that could be confusing later
        data.attrs.pop('FillValue', None)
        data.attrs.pop('Intercept', None)
        data.attrs.pop('Slope', None)

        data = data.where((data >= min(data.attrs['valid_range'])) &
                          (data <= max(data.attrs['valid_range'])))

        return data

    def get_area_def(self, key):
        # Coordination Group for Meteorological Satellites LRIT/HRIT Global Specification
        # https://www.cgms-info.org/documents/cgms-lrit-hrit-global-specification-(v2-8-of-30-oct-2013).pdf
        res = key.resolution
        coff = _COFF_list[_resolution_list.index(res)]
        loff = _LOFF_list[_resolution_list.index(res)]
        cfac = _CFAC_list[_resolution_list.index(res)]
        lfac = _LFAC_list[_resolution_list.index(res)]
        a = self.file_content['/attr/dEA'] * 1E3  # equator radius (m)
        b = a * (1 - 1 / self.file_content['/attr/dObRecFlat'])  # polar radius (m)
        h = self.file_content['/attr/NOMSatHeight']  # the altitude of satellite (m)

        lon_0 = self.file_content['/attr/NOMCenterLon']
        nlines = self.file_content['/attr/RegLength']
        ncols = self.file_content['/attr/RegWidth']

        cols = 0
        left_x = (cols - coff) * (2.**16 / cfac)
        cols += ncols - 1
        right_x = (cols - coff) * (2.**16 / cfac)

        lines = self.file_content['/attr/Begin Line Number']
        upper_y = -(lines - loff) * (2.**16 / lfac)
        lines = self.file_content['/attr/End Line Number']
        lower_y = -(lines - loff) * (2.**16 / lfac)
        area_extent = (np.deg2rad(left_x) * h, np.deg2rad(lower_y) * h,
                       np.deg2rad(right_x) * h, np.deg2rad(upper_y) * h)

        proj_dict = {'a': float(a),
                     'b': float(b),
                     'lon_0': float(lon_0),
                     'h': float(h),
                     'proj': 'geos',
                     'units': 'm',
                     'sweep': 'y'}

        area = geometry.AreaDefinition(
            self.filename_info['observation_type'],
            "AGRI {} area".format(self.filename_info['observation_type']),
            'FY-4A',
            proj_dict,
            ncols,
            nlines,
            area_extent)

        return area

    def dn2(self, dn, calibration, slope, offset):
        """Convert digital number (DN) to reflectance or radiance

        Args:
            dn: Raw detector digital number
            slope: Slope
            offset: Offset

        Returns:
            Reflectance [%]
            or Radiance [mW/ (m2 cm-1 sr)]
        """
        ref = dn * slope + offset
        if calibration == 'reflectance':
            ref *= 100  # set unit to %
        ref = ref.clip(min=0)
        ref.attrs = dn.attrs

        return ref

    @staticmethod
    def _getitem(block, lut):
        return lut[block]

    def calibrate(self, data, lut):
        """Calibrate digital number (DN) to brightness_temperature
        Args:
            dn: Raw detector digital number
            lut: the look up table
        Returns:
            brightness_temperature [K]
        """
        # append nan to the end of lut for fillvalue
        lut = np.append(lut, np.nan)
        data.data = da.where(data.data > lut.shape[0], lut.shape[0] - 1, data.data)
        res = data.data.map_blocks(self._getitem, lut, dtype=lut.dtype)
        res = xr.DataArray(res, dims=data.dims,
                           attrs=data.attrs, coords=data.coords)

        return res

    @property
    def start_time(self):
        start_time = self['/attr/Observing Beginning Date'] + 'T' + self['/attr/Observing Beginning Time'] + 'Z'
        return datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S.%fZ')

    @property
    def end_time(self):
        end_time = self['/attr/Observing Ending Date'] + 'T' + self['/attr/Observing Ending Time'] + 'Z'
        return datetime.strptime(end_time, '%Y-%m-%dT%H:%M:%S.%fZ')
