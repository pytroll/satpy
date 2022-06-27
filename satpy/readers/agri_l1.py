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
"""Advanced Geostationary Radiation Imager reader for the Level_1 HDF format.

The files read by this reader are described in the official Real Time Data Service:

    http://fy4.nsmc.org.cn/data/en/data/realtime.html

"""

import logging

from satpy.readers._geos_area import get_area_definition, get_area_extent
from satpy.readers.fy4_base import RESOLUTION_LIST, FY4Base

logger = logging.getLogger(__name__)


class HDF_AGRI_L1(FY4Base):
    """AGRI l1 file handler."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init filehandler."""
        super(HDF_AGRI_L1, self).__init__(filename, filename_info, filetype_info)
        self.sensor = 'AGRI'

    def get_dataset(self, dataset_id, ds_info):
        """Load a dataset."""
        ds_name = dataset_id['name']
        logger.debug('Reading in get_dataset %s.', ds_name)
        file_key = ds_info.get('file_key', ds_name)
        if self.PLATFORM_ID == 'FY-4B':
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
                               'satellite_nominal_latitude': self['/attr/NOMCenterLat'].item(),
                               'satellite_nominal_longitude': self['/attr/NOMCenterLon'].item(),
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
        pdict['coff'] = self._COFF_list[RESOLUTION_LIST.index(res)]
        pdict['loff'] = self._LOFF_list[RESOLUTION_LIST.index(res)]
        pdict['cfac'] = self._CFAC_list[RESOLUTION_LIST.index(res)]
        pdict['lfac'] = self._LFAC_list[RESOLUTION_LIST.index(res)]
        if self.PLATFORM_ID == 'FY-4A':
            pdict['a'] = self.file_content['/attr/dEA'] * 1e3  # equator radius (m)
        else:
            pdict['a'] = self.file_content['/attr/dEA']  # equator radius (m)
        pdict['b'] = pdict['a'] * (1 - 1 / self.file_content['/attr/dObRecFlat'])  # polar radius (m)
        pdict['h'] = self.file_content['/attr/NOMSatHeight']  # the altitude of satellite (m)
        if self.PLATFORM_ID == 'FY-4B':
            pdict['h'] = pdict['h'] - pdict['a']

        pdict['ssp_lon'] = self.file_content['/attr/NOMCenterLon']
        pdict['nlines'] = self.file_content['/attr/RegLength']
        pdict['ncols'] = self.file_content['/attr/RegWidth']

        pdict['scandir'] = 'S2N'

        b500 = ['C02']
        b1000 = ['C01', 'C03']
        b2000 = ['C04', 'C05', 'C06', 'C07']

        pdict['a_desc'] = "AGRI {} area".format(self.filename_info['observation_type'])

        pdict['a_name'] = f'{self.filename_info["observation_type"]}_{res}'
        pdict['p_id'] = f'{self.PLATFORM_ID}, {res}m'

        pdict['coff'] = pdict['coff'] + 0.5
        pdict['nlines'] = pdict['nlines'] - 1
        pdict['ncols'] = pdict['ncols'] - 1
        pdict['loff'] = (pdict['loff'] - self.file_content['/attr/End Line Number'] + 0.5)
        area_extent = get_area_extent(pdict)
        area_extent = (area_extent[0], area_extent[1], area_extent[2], area_extent[3])

        pdict['nlines'] = pdict['nlines'] + 1
        pdict['ncols'] = pdict['ncols'] + 1
        area = get_area_definition(pdict, area_extent)

        return area
