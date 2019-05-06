#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016-2019 Satpy developers
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

"""Advanced Geostationary Radiation Imager for the Level_1 HDF format

The files read by this reader are described in the official Real Time Data Service:

    http://fy4.nsmc.org.cn/data/en/data/realtime.html

"""

import logging
import numpy as np
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


class HDF_AGRL_L1(HDF5FileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(HDF_AGRL_L1, self).__init__(filename, filename_info, filetype_info)

    def get_dataset(self, dataset_id, ds_info):
        """Load a dataset."""
        logger.debug('Reading in get_dataset %s.', dataset_id.name)
        file_key = ds_info.get('file_key', dataset_id.name)
        data = self.get(file_key)

        # convert bytes to string
        data.attrs['long_name'] = data.attrs['long_name'].decode("utf-8")
        data.attrs['band_names'] = data.attrs['band_names'].decode("utf-8")

        # fill_value = data.attrs['FillValue']
        data = data.where((data >= data.attrs['valid_range'][0]) &
                          (data <= data.attrs['valid_range'][1]))

        # calibration
        calibration = ds_info['calibration']

        if calibration == 'counts':
            return data
        elif calibration == 'reflectance':
            logger.debug("Calibrating to reflectances")
            # using the corresponding SCALE and OFFSET
            cal_coef = 'CALIBRATION_COEF(SCALE+OFFSET)'
            slope = self.get(cal_coef)[:, 0][int(file_key[-2:])-1].values
            offset = self.get(cal_coef)[:, 1][int(file_key[-2:])-1].values
            data = self.dn2reflectance(data, slope, offset)
        elif calibration == 'brightness_temperature':
            logger.debug("Calibrating to brightness_temperature")
            # the value of dn is the index of brightness_temperature
            data = self.calibrate(data, 'CAL'+file_key[3:])

        data.attrs.update({'platform_name': self['/attr/Satellite Name'],
                           'sensor': self['/attr/Sensor Identification Code']})
        data.attrs.update(ds_info)

        if data.attrs.get('calibration') == 'brightness_temperature':
            data.attrs.update({'units': 'K'})
        elif data.attrs.get('calibration') == 'reflectance':
            data.attrs.update({'units': '%'})
        else:
            data.attrs.update({'units': '1'})

        return data

    def get_area_def(self, key):
        # Coordination Group for Meteorological Satellites LRIT/HRIT Global Specification
        # https://www.cgms-info.org/documents/cgms-lrit-hrit-global-specification-(v2-8-of-30-oct-2013).pdf
        res = key.resolution
        coff = _COFF_list[_resolution_list.index(res)]
        loff = _LOFF_list[_resolution_list.index(res)]
        cfac = _CFAC_list[_resolution_list.index(res)]
        lfac = _LFAC_list[_resolution_list.index(res)]
        a = 6378137.0   # equator radius (m)
        b = 6356752.3   # polar radius(m)
        H = 42164000.0  # the distance between spacecraft and centre of earth (m)
        h = H - a       # the altitude of satellite

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
                     'units': 'm'}

        area = geometry.AreaDefinition(
            self.filename_info['observation_type'],
            "AGRL {} area".format(self.filename_info['observation_type']),
            'FY-4A',
            proj_dict,
            ncols,
            nlines,
            area_extent)

        self.area = area

        return area

    def dn2reflectance(self, dn, slope, offset):
        """Convert digital number (DN) to reflectance
        Args:
            dn: Raw detector digital number
            slope: Slope
            offset: Offset
        Returns:
            Reflectance [%]
        """
        ref = dn * slope + offset
        ref *= 100  # set unit to %

        return ref.clip(min=0)

    def calibrate(self, data, cal_key):
        """Calibrate digital number (DN) to brightness_temperature
        Args:
            dn: Raw detector digital number
            cal_key: the name of CAL variable
        Returns:
            brightness_temperature [K]
        """
        # get indexes of nan values
        nan_index = np.isnan(data).values
        # assign 0 to nan values temporarily
        data.values = data.fillna(0).values.astype(int)
        # dn is the index of CAL table
        data.values = np.take(self.get(cal_key).values, data.values)
        # restore none value
        data.values[nan_index] = np.nan

        return data

    @property
    def start_time(self):
        start_time = self['/attr/Observing Beginning Date'] + 'T' + self['/attr/Observing Beginning Time'] + 'Z'
        return datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S.%fZ')

    @property
    def end_time(self):
        end_time = self['/attr/Observing Ending Date'] + 'T' + self['/attr/Observing Ending Time'] + 'Z'
        return datetime.strptime(end_time, '%Y-%m-%dT%H:%M:%S.%fZ')
