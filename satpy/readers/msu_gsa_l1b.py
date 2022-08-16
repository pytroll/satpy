#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2022 Satpy developers
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
"""Reader for the Arctica-M1 MSU-GS/A data.

The files for this reader are HDF5 and contain channel data at 1km resolution
for the VIS channels and 4km resolution for the IR channels. Geolocation data
is available at both resolutions, as is sun and satellite geometry.

This reader was tested on sample data provided by EUMETSAT.

"""
from datetime import datetime

import numpy as np

from satpy.readers.hdf5_utils import HDF5FileHandler


class MSUGSAFileHandler(HDF5FileHandler):
    """MSU-GS/A L1B file reader."""

    @property
    def start_time(self):
        """Time for timeslot scan start."""
        dtstr = self['/attr/timestamp_without_timezone']
        return datetime.strptime(dtstr, "%Y-%m-%dT%H:%M:%S")

    @property
    def satellite_altitude(self):
        """Satellite altitude at time of scan.

        There is no documentation but this appears to be
        height above surface in meters.
        """
        return float(self['/attr/satellite_observation_point_height'])

    @property
    def satellite_latitude(self):
        """Satellite latitude at time of scan."""
        return float(self['/attr/satellite_observation_point_latitude'])

    @property
    def satellite_longitude(self):
        """Satellite longitude at time of scan."""
        return float(self['/attr/satellite_observation_point_longitude'])

    @property
    def sensor_name(self):
        """Sensor name is hardcoded."""
        sensor = 'msu_gsa'
        return sensor

    @property
    def platform_name(self):
        """Platform name is also hardcoded."""
        platform = 'Arctica-M-N1'
        return platform

    @staticmethod
    def _apply_scale_offset(in_data):
        """Apply the scale and offset to data."""
        scl = in_data.attrs['scale']
        off = in_data.attrs['offset']
        return in_data * scl + off

    def get_dataset(self, dataset_id, ds_info):
        """Load data variable and metadata and calibrate if needed."""
        file_key = ds_info.get('file_key', dataset_id['name'])
        data = self[file_key]
        attrs = data.attrs.copy()  # avoid contaminating other band loading
        attrs.update(ds_info)

        # The fill value also needs to be applied
        fill_val = attrs.pop('fill_value')
        data = data.where(data != fill_val, np.nan)

        # Data has a scale and offset that we must apply
        data = self._apply_scale_offset(data)

        # Data is given as radiance values, we must convert if we want reflectance
        if dataset_id.get('calibration') == "reflectance":
            solconst = float(attrs.pop('F_solar_constant'))
            data = np.pi * data / solconst
            # Satpy expects reflectance values in 0-100 range
            data = data * 100.

        data.attrs = attrs
        data.attrs.update({
            'platform_name': self.platform_name,
            'sensor': self.sensor_name,
            'sat_altitude': self.satellite_altitude,
            'sat_latitude': self.satellite_latitude,
            'sat_longitude': self.satellite_longitude,
        })

        return data
