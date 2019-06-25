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
"""Interface to VIRR (Visible and Infra-Red Radiometer) level 1b format.

The file format is HDF5. Important attributes:

    - Latitude
    - Longitude
    - SolarZenith
    - EV_Emissive
    - EV_RefSB
    - Emissive_Radiance_Offsets
    - Emissive_Radiance_Scales
    - RefSB_Cal_Coefficients
    - RefSB_Effective_Wavelength
    - Emmisive_Centroid_Wave_Number

Supported satellites:

    - FY-3B and FY-3C.

For more information:

    - https://www.wmo-sat.info/oscar/instruments/view/607.

"""

from datetime import datetime
from satpy.readers.hdf5_utils import HDF5FileHandler
from pyspectral.blackbody import blackbody_wn_rad2temp as rad2temp
import numpy as np
import dask.array as da
import logging

LOG = logging.getLogger(__name__)


class VIRR_L1B(HDF5FileHandler):
    """VIRR_L1B reader."""

    def __init__(self, filename, filename_info, filetype_info):
        super(VIRR_L1B, self).__init__(filename, filename_info, filetype_info)
        LOG.debug('day/night flag for {0}: {1}'.format(filename, self['/attr/Day Or Night Flag']))
        self.geolocation_prefix = filetype_info['geolocation_prefix']
        self.platform_id = filename_info['platform_id']
        self.l1b_prefix = 'Data/'
        self.wave_number = 'Emissive_Centroid_Wave_Number'
        # Else filename_info['platform_id'] == FY3C.
        if filename_info['platform_id'] == 'FY3B':
            self.l1b_prefix = ''
            self.wave_number = 'Emmisive_Centroid_Wave_Number'

    def get_dataset(self, dataset_id, ds_info):
        file_key = self.geolocation_prefix + ds_info.get('file_key', dataset_id.name)
        if self.platform_id == 'FY3B':
            file_key = file_key.replace('Data/', '')
        data = self[file_key]
        band_index = ds_info.get('band_index')
        if band_index is not None:
            data = data[band_index]
            data = data.where((data >= self[file_key + '/attr/valid_range'][0]) &
                              (data <= self[file_key + '/attr/valid_range'][1]))
            if 'E' in dataset_id.name:
                slope = self._correct_slope(self[self.l1b_prefix + 'Emissive_Radiance_Scales'].
                                            data[:, band_index][:, np.newaxis])
                intercept = self[self.l1b_prefix + 'Emissive_Radiance_Offsets'].data[:, band_index][:, np.newaxis]
                # Converts cm^-1 (wavenumbers) and (mW/m^2)/(str/cm^-1) (radiance data)
                # to SI units m^-1, mW*m^-3*str^-1.
                wave_number = self['/attr/' + self.wave_number][band_index] * 100
                bt_data = rad2temp(wave_number, (data.data * slope + intercept) * 1e-5)
                if isinstance(bt_data, np.ndarray):
                    # old versions of pyspectral produce numpy arrays
                    data.data = da.from_array(bt_data, chunks=data.data.chunks)
                else:
                    # new versions of pyspectral can do dask arrays
                    data.data = bt_data
            elif 'R' in dataset_id.name:
                slope = self._correct_slope(self['/attr/RefSB_Cal_Coefficients'][0::2])
                intercept = self['/attr/RefSB_Cal_Coefficients'][1::2]
                data = data * slope[band_index] + intercept[band_index]
        else:
            slope = self._correct_slope(self[file_key + '/attr/Slope'])
            intercept = self[file_key + '/attr/Intercept']
            data = data.where((data >= self[file_key + '/attr/valid_range'][0]) &
                              (data <= self[file_key + '/attr/valid_range'][1]))
            data = data * slope + intercept
        new_dims = {old: new for old, new in zip(data.dims, ('y', 'x'))}
        data = data.rename(new_dims)
        data.attrs.update({'platform_name': self['/attr/Satellite Name'],
                           'sensor': self['/attr/Sensor Identification Code']})
        data.attrs.update(ds_info)
        units = self.get(file_key + '/attr/units')
        if units is not None and str(units).lower() != 'none':
            data.attrs.update({'units': self.get(file_key + '/attr/units')})
        elif data.attrs.get('calibration') == 'reflectance':
            data.attrs.update({'units': '%'})
        else:
            data.attrs.update({'units': '1'})
        return data

    def _correct_slope(self, slope):
        # 0 slope is invalid. Note: slope can be a scalar or array.
        return da.where(slope == 0, 1, slope)

    @property
    def start_time(self):
        start_time = self['/attr/Observing Beginning Date'] + 'T' + self['/attr/Observing Beginning Time'] + 'Z'
        return datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S.%fZ')

    @property
    def end_time(self):
        end_time = self['/attr/Observing Ending Date'] + 'T' + self['/attr/Observing Ending Time'] + 'Z'
        return datetime.strptime(end_time, '%Y-%m-%dT%H:%M:%S.%fZ')
