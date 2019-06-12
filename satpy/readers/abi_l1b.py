#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016-2018 Satpy developers
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
"""Advance Baseline Imager reader for the Level 1b format

The files read by this reader are described in the official PUG document:

    https://www.goes-r.gov/users/docs/PUG-L1b-vol3.pdf

"""

import logging

import numpy as np

from satpy.readers.abi_base import NC_ABI_BASE

logger = logging.getLogger(__name__)


class NC_ABI_L1B(NC_ABI_BASE):

    def __init__(self, filename, filename_info, filetype_info):
        super(NC_ABI_L1B, self).__init__(filename, filename_info, filetype_info)

    def get_dataset(self, key, info):
        """Load a dataset."""
        logger.debug('Reading in get_dataset %s.', key.name)
        radiances = self['Rad']

        if key.calibration == 'reflectance':
            logger.debug("Calibrating to reflectances")
            res = self._vis_calibrate(radiances)
        elif key.calibration == 'brightness_temperature':
            logger.debug("Calibrating to brightness temperatures")
            res = self._ir_calibrate(radiances)
        elif key.calibration != 'radiance':
            raise ValueError("Unknown calibration '{}'".format(key.calibration))
        else:
            res = radiances

        # convert to satpy standard units
        if res.attrs['units'] == '1':
            res *= 100
            res.attrs['units'] = '%'

        res.attrs.update({'platform_name': self.platform_name,
                          'sensor': self.sensor,
                          'satellite_latitude': float(self['nominal_satellite_subpoint_lat']),
                          'satellite_longitude': float(self['nominal_satellite_subpoint_lon']),
                          'satellite_altitude': float(self['nominal_satellite_height'])})
        res.attrs.update(key.to_dict())
        # remove attributes that could be confusing later
        res.attrs.pop('_FillValue', None)
        res.attrs.pop('scale_factor', None)
        res.attrs.pop('add_offset', None)
        res.attrs.pop('ancillary_variables', None)  # Can't currently load DQF
        # add in information from the filename that may be useful to the user
        for key in ('observation_type', 'scene_abbr', 'scan_mode', 'platform_shortname'):
            res.attrs[key] = self.filename_info[key]
        # copy global attributes to metadata
        for key in ('scene_id', 'orbital_slot', 'instrument_ID', 'production_site', 'timeline_ID'):
            res.attrs[key] = self.nc.attrs.get(key)
        # only include these if they are present
        for key in ('fusion_args',):
            if key in self.nc.attrs:
                res.attrs[key] = self.nc.attrs[key]

        return res

    def _vis_calibrate(self, data):
        """Calibrate visible channels to reflectance."""
        solar_irradiance = self['esun']
        esd = self["earth_sun_distance_anomaly_in_AU"].astype(float)

        factor = np.pi * esd * esd / solar_irradiance

        res = data * factor
        res.attrs = data.attrs
        res.attrs['units'] = '1'
        res.attrs['standard_name'] = 'toa_bidirectional_reflectance'
        return res

    def _ir_calibrate(self, data):
        """Calibrate IR channels to BT."""
        fk1 = float(self["planck_fk1"])
        fk2 = float(self["planck_fk2"])
        bc1 = float(self["planck_bc1"])
        bc2 = float(self["planck_bc2"])

        res = (fk2 / np.log(fk1 / data + 1) - bc1) / bc2
        res.attrs = data.attrs
        res.attrs['units'] = 'K'
        res.attrs['standard_name'] = 'toa_brightness_temperature'
        return res
