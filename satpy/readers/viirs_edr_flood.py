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
"""Interface to VIIRS flood product."""

from satpy.readers.hdf4_utils import HDF4FileHandler
from pyresample import geometry
import numpy as np


class VIIRSEDRFlood(HDF4FileHandler):
    """VIIRS EDR Flood-product handler for HDF4 files."""

    @property
    def start_time(self):
        """Get start time."""
        return self.filename_info['start_time']

    @property
    def end_time(self):
        """Get end time."""
        return self.filename_info.get('end_time', self.start_time)

    @property
    def sensor_name(self):
        """Get sensor name."""
        sensor = self['/attr/SensorIdentifyCode']
        if isinstance(sensor, np.ndarray):
            return str(sensor.astype(str)).lower()
        return sensor.lower()

    @property
    def platform_name(self):
        """Get platform name."""
        platform_name = self['/attr/Satellitename']
        if isinstance(platform_name, np.ndarray):
            return str(platform_name.astype(str)).lower()
        return platform_name.lower()

    def get_metadata(self, data, ds_info):
        """Get metadata."""
        metadata = {}
        metadata.update(data.attrs)
        metadata.update(ds_info)
        metadata.update({
            'sensor': self.sensor_name,
            'platform_name': self.platform_name,
            'start_time': self.start_time,
            'end_time': self.end_time,
        })

        return metadata

    def get_dataset(self, ds_id, ds_info):
        """Get dataset."""
        data = self[ds_id.name]

        data.attrs = self.get_metadata(data, ds_info)

        fill = data.attrs.pop('_Fillvalue')
        offset = data.attrs.get('add_offset')
        scale_factor = data.attrs.get('scale_factor')

        data = data.where(data != fill)
        if scale_factor is not None and offset is not None:
            data *= scale_factor
            data += offset

        return data

    def get_area_def(self, ds_id):
        """Get area definition."""
        data = self[ds_id.name]

        proj_dict = {
            'proj': 'latlong',
            'datum': 'WGS84',
            'ellps': 'WGS84',
            'no_defs': True
        }

        area_extent = [data.attrs.get('ProjectionMinLongitude'), data.attrs.get('ProjectionMinLatitude'),
                       data.attrs.get('ProjectionMaxLongitude'), data.attrs.get('ProjectionMaxLatitude')]

        area = geometry.AreaDefinition(
            'viirs_flood_area',
            'name_of_proj',
            'id_of_proj',
            proj_dict,
            int(self.filename_info['dim0']),
            int(self.filename_info['dim1']),
            np.asarray(area_extent)
        )

        return area
