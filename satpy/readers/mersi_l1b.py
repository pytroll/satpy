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
"""Reader for the FY-3D MERSI-2 L1B file format.

The files for this reader are HDF5 and come in four varieties; band data
and geolocation data, both at 250m and 1000m resolution.

This reader was tested on FY-3D MERSI-2 data, but should work on future
platforms as well assuming no file format changes.

"""
from datetime import datetime

import dask.array as da
import numpy as np
from pyspectral.blackbody import blackbody_wn_rad2temp as rad2temp

from satpy.readers.hdf5_utils import HDF5FileHandler

N_TOT_IR_CHANS_LL = 6


class MERSIL1B(HDF5FileHandler):
    """MERSI-2/MERSI-LL L1B file reader."""

    def _strptime(self, date_attr, time_attr):
        """Parse date/time strings."""
        date = self[date_attr]
        time = self[time_attr]  # "18:27:39.720"
        # cuts off microseconds because of unknown meaning
        # is .720 == 720 microseconds or 720000 microseconds
        return datetime.strptime(date + " " + time.split('.')[0], "%Y-%m-%d %H:%M:%S")

    @property
    def start_time(self):
        """Time for first observation."""
        return self._strptime('/attr/Observing Beginning Date', '/attr/Observing Beginning Time')

    @property
    def end_time(self):
        """Time for final observation."""
        return self._strptime('/attr/Observing Ending Date', '/attr/Observing Ending Time')

    @property
    def sensor_name(self):
        """Map sensor name to Satpy 'standard' sensor names."""
        file_sensor = self['/attr/Sensor Identification Code']
        sensor = {
            'MERSI': 'mersi-2',
            'MERSI LL': 'mersi-ll',
        }.get(file_sensor, file_sensor)
        return sensor

    def _get_single_slope_intercept(self, slope, intercept, cal_index):
        try:
            # convert scalar arrays to scalar
            return slope.item(), intercept.item()
        except ValueError:
            # numpy array but has more than one element
            return slope[cal_index], intercept[cal_index]

    def _get_coefficients(self, cal_key, cal_index):
        coeffs = self[cal_key][cal_index]
        slope = coeffs.attrs.pop('Slope', None)
        intercept = coeffs.attrs.pop('Intercept', None)
        if slope is not None:
            slope, intercept = self._get_single_slope_intercept(
                slope, intercept, cal_index)
            coeffs = coeffs * slope + intercept
        return coeffs

    def get_dataset(self, dataset_id, ds_info):
        """Load data variable and metadata and calibrate if needed."""
        file_key = ds_info.get('file_key', dataset_id['name'])
        band_index = ds_info.get('band_index')
        data = self[file_key]
        if band_index is not None:
            data = data[band_index]
        if data.ndim >= 2:
            data = data.rename({data.dims[-2]: 'y', data.dims[-1]: 'x'})
        attrs = data.attrs.copy()  # avoid contaminating other band loading
        attrs.update(ds_info)
        if 'rows_per_scan' in self.filetype_info:
            attrs.setdefault('rows_per_scan', self.filetype_info['rows_per_scan'])

        data = self._mask_data(data, dataset_id, attrs)

        slope = attrs.pop('Slope', None)
        intercept = attrs.pop('Intercept', None)
        if slope is not None and dataset_id.get('calibration') != 'counts':
            if band_index is not None:
                slope = slope[band_index]
                intercept = intercept[band_index]
            data = data * slope + intercept

        if dataset_id.get('calibration') == "reflectance":
            coeffs = self._get_coefficients(ds_info['calibration_key'],
                                            ds_info['calibration_index'])
            data = coeffs[0] + coeffs[1] * data + coeffs[2] * data ** 2
        elif dataset_id.get('calibration') == "brightness_temperature":
            calibration_index = ds_info['calibration_index']
            # Converts um^-1 (wavenumbers) and (mW/m^2)/(str/cm^-1) (radiance data)
            # to SI units m^-1, mW*m^-3*str^-1.
            wave_number = 1. / (dataset_id['wavelength'][1] / 1e6)

            data = self._get_bt_dataset(data, calibration_index, wave_number)

        data.attrs = attrs
        # convert bytes to str
        for key, val in attrs.items():
            # python 3 only
            if bytes is not str and isinstance(val, bytes):
                data.attrs[key] = val.decode('utf8')

        data.attrs.update({
            'platform_name': self['/attr/Satellite Name'],
            'sensor': self.sensor_name,
        })

        return data

    def _mask_data(self, data, dataset_id, attrs):
        """Mask the data using fill_value and valid_range attributes."""
        fill_value = attrs.pop('FillValue', np.nan)  # covered by valid_range
        valid_range = attrs.pop('valid_range', None)
        if dataset_id.get('calibration') == 'counts':
            # preserve integer type of counts if possible
            attrs['_FillValue'] = fill_value
            new_fill = fill_value
        else:
            new_fill = np.nan
        if valid_range is not None:
            # Due to a bug in the valid_range upper limit in the 10.8(24) and 12.0(25)
            # in the HDF data, this is hardcoded here.
            if dataset_id['name'] in ['24', '25'] and valid_range[1] == 4095:
                valid_range[1] = 25000
            # typically bad_values == 65535, saturated == 65534
            # dead detector == 65533
            data = data.where((data >= valid_range[0]) &
                              (data <= valid_range[1]), new_fill)
        return data

    def _get_bt_dataset(self, data, calibration_index, wave_number):
        """Get the dataset as brightness temperature.

        Apparently we don't use these calibration factors for Rad -> BT::

            coeffs = self._get_coefficients(ds_info['calibration_key'], calibration_index)
            # coefficients are per-scan, we need to repeat the values for a
            # clean alignment
            coeffs = np.repeat(coeffs, data.shape[0] // coeffs.shape[1], axis=1)
            coeffs = coeffs.rename({
                coeffs.dims[0]: 'coefficients', coeffs.dims[1]: 'y'
            })  # match data dims
            data = coeffs[0] + coeffs[1] * data + coeffs[2] * data**2 + coeffs[3] * data**3

        """
        # pass the dask array
        bt_data = rad2temp(wave_number, data.data * 1e-5)  # brightness temperature
        if isinstance(bt_data, np.ndarray):
            # old versions of pyspectral produce numpy arrays
            data.data = da.from_array(bt_data, chunks=data.data.chunks)
        else:
            # new versions of pyspectral can do dask arrays
            data.data = bt_data

        # Some BT bands seem to have 0 in the first 10 columns
        # and it is an invalid measurement, so let's mask
        data = data.where(data != 0)

        # additional corrections from the file
        if self.sensor_name == 'mersi-2':
            corr_coeff_a = float(self['/attr/TBB_Trans_Coefficient_A'][calibration_index])
            corr_coeff_b = float(self['/attr/TBB_Trans_Coefficient_B'][calibration_index])
        elif self.sensor_name == 'mersi-ll':
            # MERSI-LL stores these coefficients differently
            try:
                coeffs = self['/attr/TBB_Trans_Coefficient']
                corr_coeff_a = coeffs[calibration_index]
                corr_coeff_b = coeffs[calibration_index + N_TOT_IR_CHANS_LL]
            except KeyError:
                return data

        if corr_coeff_a != 0:
            data = (data - corr_coeff_b) / corr_coeff_a
        # some bands have 0 counts for the first N columns and
        # seem to be invalid data points
        data = data.where(data != 0)
        return data
