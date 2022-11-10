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
"""Geostationary High-speed Imager reader for the Level_1 HDF format.

This instrument is aboard the Fengyun-4B satellite. No document is available to describe this
format is available, but it's broadly similar to the co-flying AGRI instrument.

"""

import logging

from pyproj import Proj

from satpy.readers._geos_area import get_area_definition
from satpy.readers.fy4_base import FY4Base

logger = logging.getLogger(__name__)


class HDF_GHI_L1(FY4Base):
    """GHI l1 file handler."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init filehandler."""
        super(HDF_GHI_L1, self).__init__(filename, filename_info, filetype_info)
        self.sensor = 'GHI'

    def get_dataset(self, dataset_id, ds_info):
        """Load a dataset."""
        ds_name = dataset_id['name']
        logger.debug('Reading in get_dataset %s.', ds_name)
        file_key = ds_info.get('file_key', ds_name)
        if self.CHANS_ID in file_key:
            file_key = f'Data/{file_key}'
        elif self.SUN_ID in file_key or self.SAT_ID in file_key:
            file_key = f'Navigation/{file_key}'
        data = self.get(file_key)
        if data.ndim >= 2:
            data = data.rename({data.dims[-2]: 'y', data.dims[-1]: 'x'})

        data = self.calibrate(data, ds_info, ds_name, file_key)

        self.adjust_attrs(data, ds_info)

        return data

    def adjust_attrs(self, data, ds_info):
        """Adjust the attrs of the data."""
        satname = self.PLATFORM_NAMES.get(self['/attr/Satellite Name'], self['/attr/Satellite Name'])
        data.attrs.update({'platform_name': satname,
                           'sensor': self['/attr/Sensor Identification Code'].lower(),
                           'orbital_parameters': {
                               'satellite_nominal_latitude': self['/attr/NOMSubSatLat'].item(),
                               'satellite_nominal_longitude': self['/attr/NOMSubSatLon'].item(),
                               'satellite_nominal_altitude': self['/attr/NOMSatHeight'].item()}})
        data.attrs.update(ds_info)
        # remove attributes that could be confusing later
        data.attrs.pop('FillValue', None)
        data.attrs.pop('Intercept', None)
        data.attrs.pop('Slope', None)

    def get_area_def(self, key):
        """Get the area definition."""
        # Coordination Group for Meteorological Satellites LRIT/HRIT Global Specification
        # https://www.cgms-info.org/documents/cgms-lrit-hrit-global-specification-(v2-8-of-30-oct-2013).pdf
        res = key['resolution']

        pdict = {}

        c_lats = self.file_content['/attr/Corner-Point Latitudes']
        c_lons = self.file_content['/attr/Corner-Point Longitudes']

        p1 = (c_lons[0], c_lats[0])
        p2 = (c_lons[1], c_lats[1])
        p3 = (c_lons[2], c_lats[2])
        p4 = (c_lons[3], c_lats[3])

        pdict['a'] = self.file_content['/attr/Semi_major_axis'] * 1E3  # equator radius (m)
        pdict['b'] = self.file_content['/attr/Semi_minor_axis'] * 1E3  # equator radius (m)
        pdict['h'] = self.file_content['/attr/NOMSatHeight'] * 1E3  # the altitude of satellite (m)

        pdict['h'] = pdict['h'] - pdict['a']

        pdict['ssp_lon'] = float(self.file_content['/attr/NOMSubSatLon'])
        pdict['nlines'] = float(self.file_content['/attr/RegLength'])
        pdict['ncols'] = float(self.file_content['/attr/RegWidth'])

        pdict['scandir'] = 'S2N'

        pdict['a_desc'] = "FY-4 {} area".format(self.filename_info['observation_type'])
        pdict['a_name'] = f'{self.filename_info["observation_type"]}_{res}m'
        pdict['p_id'] = f'FY-4, {res}m'

        proj_dict = {'a': pdict['a'],
                     'b': pdict['b'],
                     'lon_0': pdict['ssp_lon'],
                     'h': pdict['h'],
                     'proj': 'geos',
                     'units': 'm',
                     'sweep': 'y'}

        p = Proj(proj_dict)
        o1 = (p(p1[0], p1[1]))  # Upper left
        o2 = (p(p2[0], p2[1]))  # Upper right
        o3 = (p(p3[0], p3[1]))  # Lower left
        o4 = (p(p4[0], p4[1]))  # Lower right

        deller = res / 2.

        area = get_area_definition(pdict, (o3[0] - deller, o4[1] - deller, o2[0], o1[1]))

        return area
