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
"""Base reader for the L1 HDF data from the AGRI and GHI instruments aboard the FengYun-4A/B satellites.

The files read by this reader are described in the official Real Time Data Service:

    http://fy4.nsmc.org.cn/data/en/data/realtime.html

"""

import logging
from datetime import datetime

import dask.array as da
import numpy as np
import xarray as xr

from satpy._compat import cached_property
from satpy.readers._geos_area import get_area_definition, get_area_extent
from satpy.readers.hdf5_utils import HDF5FileHandler

logger = logging.getLogger(__name__)

RESOLUTION_LIST = [250, 500, 1000, 2000, 4000]


class FY4Base(HDF5FileHandler):
    """The base class for the FengYun4 AGRI and GHI readers."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init filehandler."""
        super(FY4Base, self).__init__(filename, filename_info, filetype_info)

        self.sensor = filename_info['instrument']

        # info of 250m, 500m, 1km, 2km and 4km data
        self._COFF_list = [21983.5, 10991.5, 5495.5, 2747.5, 1373.5]
        self._LOFF_list = [21983.5, 10991.5, 5495.5, 2747.5, 1373.5]

        self._CFAC_list = [163730199.0, 81865099.0, 40932549.0, 20466274.0, 10233137.0]
        self._LFAC_list = [163730199.0, 81865099.0, 40932549.0, 20466274.0, 10233137.0]

        self.PLATFORM_NAMES = {'FY4A': 'FY-4A',
                               'FY4B': 'FY-4B',
                               'FY4C': 'FY-4C'}

        try:
            self.PLATFORM_ID = self.PLATFORM_NAMES[filename_info['platform_id']]
        except KeyError:
            raise KeyError(f"Unsupported platform ID: {filename_info['platform_id']}")
        self.CHANS_ID = 'NOMChannel'
        self.SAT_ID = 'NOMSatellite'
        self.SUN_ID = 'NOMSun'

    @staticmethod
    def scale(dn, slope, offset):
        """Convert digital number (DN) to calibrated quantity through scaling.

        Args:
            dn: Raw detector digital number
            slope: Slope
            offset: Offset

        Returns:
            Scaled data

        """
        ref = dn * slope + offset
        ref = ref.clip(min=0)
        ref.attrs = dn.attrs

        return ref

    def apply_lut(self, data, lut):
        """Calibrate digital number (DN) by applying a LUT.

        Args:
            data: Raw detector digital number
            lut: the look up table
        Returns:
            Calibrated quantity
        """
        # append nan to the end of lut for fillvalue
        lut = np.append(lut, np.nan)
        data.data = da.where(data.data > lut.shape[0], lut.shape[0] - 1, data.data)
        res = data.data.map_blocks(self._getitem, lut, dtype=lut.dtype)
        res = xr.DataArray(res, dims=data.dims,
                           attrs=data.attrs, coords=data.coords)

        return res

    @staticmethod
    def _getitem(block, lut):
        return lut[block]

    @cached_property
    def reflectance_coeffs(self):
        """Retrieve the reflectance calibration coefficients from the HDF file."""
        # using the corresponding SCALE and OFFSET
        if self.PLATFORM_ID == 'FY-4A':
            cal_coef = 'CALIBRATION_COEF(SCALE+OFFSET)'
        elif self.PLATFORM_ID == 'FY-4B':
            cal_coef = 'Calibration/CALIBRATION_COEF(SCALE+OFFSET)'
        else:
            raise KeyError(f"Unsupported platform ID for calibration: {self.PLATFORM_ID}")
        return self.get(cal_coef).values

    def calibrate(self, data, ds_info, ds_name, file_key):
        """Calibrate the data."""
        # Check if calibration is present, if not assume dataset is an angle
        calibration = ds_info.get('calibration')
        # Return raw data in case of counts or no calibration
        if calibration in ('counts', None):
            data.attrs['units'] = ds_info['units']
            ds_info['valid_range'] = data.attrs['valid_range']
            ds_info['fill_value'] = data.attrs['FillValue'].item()
        elif calibration == 'reflectance':
            channel_index = int(file_key[-2:]) - 1
            data = self.calibrate_to_reflectance(data, channel_index, ds_info)
        elif calibration == 'brightness_temperature':
            data = self.calibrate_to_bt(data, ds_info, ds_name)
        elif calibration == 'radiance':
            raise NotImplementedError("Calibration to radiance is not supported.")
        # Apply range limits, but not for counts or we convert to float!
        if calibration != 'counts':
            data = data.where((data >= min(data.attrs['valid_range'])) &
                              (data <= max(data.attrs['valid_range'])))
        else:
            data.attrs['_FillValue'] = data.attrs['FillValue'].item()
        return data

    def calibrate_to_reflectance(self, data, channel_index, ds_info):
        """Calibrate to reflectance [%]."""
        logger.debug("Calibrating to reflectances")
        # using the corresponding SCALE and OFFSET
        if self.sensor != 'AGRI' and self.sensor != 'GHI':
            raise ValueError(f'Unsupported sensor type: {self.sensor}')

        coeffs = self.reflectance_coeffs
        num_channel = coeffs.shape[0]

        if self.sensor == 'AGRI' and num_channel == 1:
            # only channel_2, resolution = 500 m
            channel_index = 0
        data.data = da.where(data.data == data.attrs['FillValue'].item(), np.nan, data.data)
        data.attrs['scale_factor'] = coeffs[channel_index, 0].item()
        data.attrs['add_offset'] = coeffs[channel_index, 1].item()
        data = self.scale(data, data.attrs['scale_factor'], data.attrs['add_offset'])
        data *= 100
        ds_info['valid_range'] = (data.attrs['valid_range'] * data.attrs['scale_factor'] + data.attrs['add_offset'])
        ds_info['valid_range'] = ds_info['valid_range'] * 100
        return data

    def calibrate_to_bt(self, data, ds_info, ds_name):
        """Calibrate to Brightness Temperatures [K]."""
        logger.debug("Calibrating to brightness_temperature")

        if self.sensor not in ['GHI', 'AGRI']:
            raise ValueError("Error, sensor must be GHI or AGRI.")

        # The key is sometimes prefixes with `Calibration/` so we try both options here
        lut_key = ds_info.get('lut_key', ds_name)
        try:
            lut = self[lut_key]
        except KeyError:
            lut_key = f'Calibration/{ds_info.get("lut_key", ds_name)}'
            lut = self[lut_key]

        # the value of dn is the index of brightness_temperature
        data = self.apply_lut(data, lut)
        ds_info['valid_range'] = lut.attrs['valid_range']
        return data

    @property
    def start_time(self):
        """Get the start time."""
        start_time = self['/attr/Observing Beginning Date'] + 'T' + self['/attr/Observing Beginning Time'] + 'Z'
        try:
            return datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S.%fZ')
        except ValueError:
            # For some data there is no sub-second component
            return datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%SZ')

    @property
    def end_time(self):
        """Get the end time."""
        end_time = self['/attr/Observing Ending Date'] + 'T' + self['/attr/Observing Ending Time'] + 'Z'
        try:
            return datetime.strptime(end_time, '%Y-%m-%dT%H:%M:%S.%fZ')
        except ValueError:
            # For some data there is no sub-second component
            return datetime.strptime(end_time, '%Y-%m-%dT%H:%M:%SZ')

    def get_area_def(self, key):
        """Get the area definition."""
        # Coordination Group for Meteorological Satellites LRIT/HRIT Global Specification
        # https://www.cgms-info.org/documents/cgms-lrit-hrit-global-specification-(v2-8-of-30-oct-2013).pdf
        res = key['resolution']
        pdict = {}
        pdict['coff'] = self._COFF_list[RESOLUTION_LIST.index(res)]
        pdict['loff'] = self._LOFF_list[RESOLUTION_LIST.index(res)]
        pdict['cfac'] = self._CFAC_list[RESOLUTION_LIST.index(res)]
        pdict['lfac'] = self._LFAC_list[RESOLUTION_LIST.index(res)]
        try:
            pdict['a'] = float(self.file_content['/attr/Semimajor axis of ellipsoid'])
        except KeyError:
            pdict['a'] = float(self.file_content['/attr/dEA'])
        if pdict['a'] < 10000:
            pdict['a'] = pdict['a'] * 1E3  # equator radius (m)
        try:
            pdict['b'] = float(self.file_content['/attr/Semiminor axis of ellipsoid'])
        except KeyError:
            pdict['b'] = pdict['a'] * (1 - 1 / self.file_content['/attr/dObRecFlat'])  # polar radius (m)

        pdict['h'] = self.file_content['/attr/NOMSatHeight']  # the altitude of satellite (m)
        if pdict['h'] > 42000000.0:
            pdict['h'] = pdict['h'] - pdict['a']

        pdict['ssp_lon'] = float(self.file_content['/attr/NOMCenterLon'])
        pdict['nlines'] = float(self.file_content['/attr/RegLength'])
        pdict['ncols'] = float(self.file_content['/attr/RegWidth'])

        pdict['scandir'] = 'N2S'
        pdict['a_desc'] = "FY-4 {} area".format(self.filename_info['observation_type'])
        pdict['a_name'] = f'{self.filename_info["observation_type"]}_{res}m'
        pdict['p_id'] = f'FY-4, {res}m'

        pdict['nlines'] = pdict['nlines'] - 1
        pdict['ncols'] = pdict['ncols'] - 1

        pdict['coff'] = pdict['coff'] - 0.5
        pdict['loff'] = pdict['loff'] + 1

        area_extent = get_area_extent(pdict)
        area_extent = (area_extent[0],
                       area_extent[1],
                       area_extent[2],
                       area_extent[3])

        pdict['nlines'] = pdict['nlines'] + 1
        pdict['ncols'] = pdict['ncols'] + 1

        area = get_area_definition(pdict, area_extent)

        return area
