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
"""Base reader for the Level_1 HDF format data from the AGRI and GHI instruments aboard the
FY-4A and FY-4B satellites.

The files read by this reader are described in the official Real Time Data Service:

    http://fy4.nsmc.org.cn/data/en/data/realtime.html

"""

import logging

from datetime import datetime
import dask.array as da
import numpy as np
import xarray as xr

from satpy.readers.hdf5_utils import HDF5FileHandler

logger = logging.getLogger(__name__)

RESOLUTION_LIST_AGRI = [500, 1000, 2000, 4000]
RESOLUTION_LIST_GHI = [250, 500, 1000, 2000]


class FY4Base(HDF5FileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        """Init filehandler."""
        super(FY4Base, self).__init__(filename, filename_info, filetype_info)

        self.sensor = filename_info['instrument']

        # info of 250m, 500m, 1km, 2km and 4km data
        self._COFF_list = [21982.5, 10991.5, 5495.5, 2747.5, 1373.5]
        self._CFAC_list = [163730198.0, 81865099.0, 40932549.0, 20466274.0, 10233137.0]
        self._LOFF_list = [21982.5, 10991.5, 5495.5, 2747.5, 1373.5]
        self._LFAC_list = [163730198.0, 81865099.0, 40932549.0, 20466274.0, 10233137.0]
        self._NLINE_list = [43960 / 2., 21980 / 2., 10990 / 2., 5495 / 2.]
        self._NCOLS_list = [43960 / 2., 21980 / 2., 10990 / 2., 5495 / 2.]

        self.PLATFORM_NAMES = {'FY4A': 'FY-4A',
                               'FY4B': 'FY-4B',
                               'FY4C': 'FY-4C'}

        self.CHANS_ID = 'NOMChannel'
        self.SAT_ID = 'NOMSatellite'
        self.SUN_ID = 'NOMSun'
        self.sensor = None

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

    def calibrate(self, data, ds_info, ds_name, file_key):
        """Calibrate the data."""
        # Check if calibration is present, if not assume dataset is an angle
        calibration = ds_info.get('calibration')
        # Return raw data in case of counts or no calibration
        if calibration in ('counts', None):
            data.attrs['units'] = ds_info['units']
            ds_info['valid_range'] = data.attrs['valid_range']
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
        if self.sensor == 'AGRI':
            cal_coef = 'CALIBRATION_COEF(SCALE+OFFSET)'
        elif self.sensor == 'GHI':
            cal_coef = 'Calibration/CALIBRATION_COEF(SCALE+OFFSET)'
        else:
            raise ValueError(f'Unsupported sensor type: {self.sensor}')

        num_channel = self.get(cal_coef).shape[0]
        if num_channel == 1:
            # only channel_2, resolution = 500 m
            channel_index = 0
        data.attrs['scale_factor'] = self.get(cal_coef)[channel_index, 0].values.item()
        data.attrs['add_offset'] = self.get(cal_coef)[channel_index, 1].values.item()
        data = self.scale(data, data.attrs['scale_factor'], data.attrs['add_offset'])
        data *= 100
        ds_info['valid_range'] = (data.attrs['valid_range'] * data.attrs['scale_factor'] + data.attrs['add_offset'])
        ds_info['valid_range'] = ds_info['valid_range'] * 100
        return data

    def calibrate_to_bt(self, data, ds_info, ds_name):
        """Calibrate to Brightness Temperatures [K]."""
        logger.debug("Calibrating to brightness_temperature")

        if self.sensor == 'AGRI':
            lut_key = ds_info.get('lut_key', ds_name)
        elif self.sensor == 'GHI':
            lut_key = f'Calibration/{ds_info.get("lut_key", ds_name)}'
        else:
            raise ValueError(f'Unsupported sensor type: {self.sensor}')
        lut = self.get(lut_key)
        # the value of dn is the index of brightness_temperature
        data = self.apply_lut(data, lut)
        ds_info['valid_range'] = lut.attrs['valid_range']
        return data

    @property
    def start_time(self):
        """Get the start time."""
        start_time = self['/attr/Observing Beginning Date'] + 'T' + self['/attr/Observing Beginning Time'] + 'Z'
        return datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S.%fZ')

    @property
    def end_time(self):
        """Get the end time."""
        end_time = self['/attr/Observing Ending Date'] + 'T' + self['/attr/Observing Ending Time'] + 'Z'
        return datetime.strptime(end_time, '%Y-%m-%dT%H:%M:%S.%fZ')
