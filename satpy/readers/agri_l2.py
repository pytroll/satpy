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

"""Advanced Geostationary Radiation Imager reader for the Level_2 NC format

The files read by this reader are described in the official Real Time Data Service:

    http://fy4.nsmc.org.cn/data/en/data/realtime.html
"""

from satpy.readers.netcdf_utils import NetCDF4FileHandler, netCDF4
from satpy.readers._geos_area import get_area_extent, get_area_definition
import logging
import numpy as np
import xarray as xr
from satpy import CHUNK_SIZE

logger = logging.getLogger(__name__)

# info of 1 km and 4 km data
_resolution_list = [1000, 4000]
_COFF_list = [5495.5, 1373.5]
_CFAC_list = [40932549.0, 10233137.0]
_LOFF_list = [5495.5, 1373.5]
_LFAC_list = [40932549.0, 10233137.0]
_OBI_type = {0: 'Full_disk_observation',
             1: 'Southern_hemisphere_observation',
             2: 'Northern _hemisphere_observation',
             3: 'Regional observation'}


class AGRIL2FileHandler(NetCDF4FileHandler):
    """File handler for AGRI L2 netCDF files."""
    def __init__(self, filename, filename_info, filetype_info):
        """Open the NetCDF file with xarray and prepare the Dataset for reading."""
        super(AGRIL2FileHandler, self).__init__(filename, filename_info, filetype_info)
        self.nc = xr.open_dataset(self.filename,
                                  decode_cf=True,
                                  mask_and_scale=False,
                                  chunks={'x': CHUNK_SIZE, 'y': CHUNK_SIZE}, )

    def available_datasets(self, configured_datasets=None):
        """Automatically determine datasets provided by this file."""
        logger.debug('Available_datasets begin...')
        handled_variables = set()

        # Iterate over dataset contents
        for var_name, val in self.file_content.items():
            # Only evaluate variables
            if isinstance(val, netCDF4.Variable):
                logger.debug('Found valid additional dataset: %s', var_name)

                # Skip anything we have already configured
                if (var_name in handled_variables):
                    logger.debug('Already handled, skipping: %s', var_name)
                    continue
                handled_variables.add(var_name)

                # Create new ds_info object
                new_info = {
                    'name': var_name,
                    'file_key': var_name,
                    'file_type': self.filetype_info['file_type'],
                    'resolution': None,
                }
                yield True, new_info

    def get_metadata(self, data, ds_info):
        """Get metadata."""
        metadata = {}
        metadata.update(data.attrs)
        metadata.update(ds_info)
        metadata.update({
            'platform_shortname': self.platform_shortname,
            'sensor': self.sensor,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'orbital_parameters': {'satellite_nominal_latitude': self['nominal_satellite_subpoint_lat'].values.item(),
                                   'satellite_nominal_longitude': self['nominal_satellite_subpoint_lon'].values.item(),
                                   'satellite_nominal_altitude': self['nominal_satellite_height'].values.item()
                                   }
        })

        return metadata

    def get_dataset(self, ds_id, ds_info):
        """Get dataset."""
        logger.debug('Getting data for: %s', ds_id.name)
        file_key = ds_info.get('file_key', ds_id.name)

        # read data and attributes
        data = self[file_key]
        data.attrs = self.get_metadata(data, ds_info)

        scale_factor = data.attrs.get('scale_factor')
        add_offset = data.attrs.get('add_offset')
        valid_range = data.attrs.get('valid_range')
        fill_value = data.attrs.get('FillValue', np.float32(np.nan))
        data = data.squeeze()

        # preserve integer data types if possible
        if np.issubdtype(data.dtype, np.integer):
            new_fill = fill_value
        else:
            new_fill = np.float32(np.nan)
            data.attrs.pop('FillValue', None)
        good_mask = data != fill_value

        # scale data
        if scale_factor is not None:
            data.data = data.data * scale_factor + add_offset

        # mask data
        data = data.where(good_mask, new_fill)
        data = data.where((data >= valid_range[0]) &
                          (data <= valid_range[1]))

        # drop default coordinates
        data = data.drop_vars(['x', 'y'])

        return data

    def get_area_def(self, key):
        """Get the area definition of the data at hand."""
        res = self.filename_info['resolution']
        if 'km' in res.lower():
            res = int(res[:-2]) * 1e3  # convert to m
        else:
            res = int(res[:-1])

        if res in [1000, 4000]:
            return self._get_areadef_fixedgrid(key, res)
        # elif res == 64000:
            # return self._get_areadef_latlon(key)
        else:
            raise ValueError('Unsupported projection found in the dataset')

    def _get_areadef_fixedgrid(self, key, res):
        pdict = {}
        pdict['coff'] = _COFF_list[_resolution_list.index(res)]
        pdict['loff'] = _LOFF_list[_resolution_list.index(res)]
        pdict['cfac'] = _CFAC_list[_resolution_list.index(res)]
        pdict['lfac'] = _LFAC_list[_resolution_list.index(res)]

        # hard code
        pdict['a'] = 6378.14 * 1E3  # equator radius (m)
        pdict['b'] = pdict['a'] * (1 - 1 / 298.257223563)  # polar radius (m)
        pdict['h'] = self['nominal_satellite_height'].values.item()  # the altitude of satellite (m)

        pdict['ssp_lon'] = self['nominal_satellite_subpoint_lon'].values.item()
        pdict['nlines'] = self['geospatial_lat_lon_extent'].attrs['RegLength']
        pdict['ncols'] = self['geospatial_lat_lon_extent'].attrs['RegWidth']

        pdict['scandir'] = 'S2N'

        obi_type = self['OBIType'].values.item()
        pdict['a_desc'] = 'AGRI {} area'.format(obi_type)

        if res == 1000:
            pdict['a_name'] = _OBI_type[obi_type]+'_1000m'
            pdict['p_id'] = 'FY-4A, 1000m'
        elif res == 12000:
            pdict['a_name'] = _OBI_type[obi_type]+'_12000m'
            pdict['p_id'] = 'FY-4A, 12000m'
        elif res == 64000:
            pdict['a_name'] = _OBI_type[obi_type]+'_64000m'
            pdict['p_id'] = 'FY-4A, 64000m'
        else:
            pdict['a_name'] = _OBI_type[obi_type]+'_4000m'
            pdict['p_id'] = 'FY-4A, 4000m'

        pdict['coff'] = pdict['coff'] + 0.5
        pdict['nlines'] = pdict['nlines'] - 1
        pdict['ncols'] = pdict['ncols'] - 1
        pdict['loff'] = (pdict['loff'] - self['geospatial_lat_lon_extent'].attrs['end_line_number'] + 0.5)
        area_extent = get_area_extent(pdict)
        area_extent = (area_extent[0] + 2000, area_extent[1], area_extent[2] + 2000, area_extent[3])

        pdict['nlines'] = pdict['nlines'] + 1
        pdict['ncols'] = pdict['ncols'] + 1
        area = get_area_definition(pdict, area_extent)

        return area

    @property
    def start_time(self):
        """Get start time."""
        return self.filename_info['start_time']

    @property
    def end_time(self):
        """Get end time."""
        return self.filename_info.get('end_time', self.start_time)

    @property
    def platform_shortname(self):
        """Get platform shortname."""
        return self.filename_info['platform_shortname']

    @property
    def sensor(self):
        """Get sensor."""
        res = self.filename_info['instrument']
        if isinstance(res, np.ndarray):
            return str(res.astype(str)).lower()
        return res.lower()

    @property
    def sensor_names(self):
        """Get sensor set."""
        return {self.sensor}
