#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 Satpy developers
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
"""Reader for Himawari L2 cloud products from NOAA's big data programme."""

import logging
from datetime import datetime

import xarray as xr

from satpy._compat import cached_property
from satpy.readers._geos_area import get_area_definition, get_area_extent
from satpy.readers.file_handlers import BaseFileHandler
from satpy.utils import get_legacy_chunk_size

logger = logging.getLogger(__name__)

CHUNK_SIZE = get_legacy_chunk_size()

EXPECTED_DATA_AREA = 'Full Disk'


class HIML2NCFileHandler(BaseFileHandler):
    """File handler for Himawari L2 NOAA enterprise data in netCDF format."""

    def __init__(self, filename, filename_info, filetype_info, geo_data=None):
        """Initialize the reader."""
        super(HIML2NCFileHandler, self).__init__(filename, filename_info,
                                                 filetype_info)
        self.nc = xr.open_dataset(self.filename,
                                  decode_cf=True,
                                  mask_and_scale=False,
                                  chunks={'xc': CHUNK_SIZE, 'yc': CHUNK_SIZE})

        # Check that file is a full disk scene, we don't know the area for anything else
        if self.nc.attrs['cdm_data_type'] != EXPECTED_DATA_AREA:
            raise ValueError('File is not a full disk scene')

        self.sensor = self.nc.attrs['instrument_name'].lower()
        self.nlines = self.nc.dims['Columns']
        self.ncols = self.nc.dims['Rows']
        self.platform_name = self.nc.attrs['satellite_name']
        self.platform_shortname = filename_info['platform']
        self._meta = None

    @property
    def start_time(self):
        """Start timestamp of the dataset."""
        dt = self.nc.attrs['time_coverage_start']
        return datetime.strptime(dt, '%Y-%m-%dT%H:%M:%SZ')

    @property
    def end_time(self):
        """End timestamp of the dataset."""
        dt = self.nc.attrs['time_coverage_end']
        return datetime.strptime(dt, '%Y-%m-%dT%H:%M:%SZ')

    def get_dataset(self, key, info):
        """Load a dataset."""
        var = info['file_key']
        logger.debug('Reading in get_dataset %s.', var)
        variable = self.nc[var]
        variable.attrs.update(key.to_dict())
        return variable

    @cached_property
    def area(self):
        """Get AreaDefinition representing this file's data."""
        return self._get_area_def()

    def get_area_def(self, dsid):
        """Get the area definition."""
        del dsid
        return self.area

    def _get_area_def(self):
        logger.warning('This product misses metadata required to produce an appropriate area definition.'
                       'Assuming standard Himawari-8/9 full disk projection.')
        pdict = {}
        pdict['cfac'] = 20466275
        pdict['lfac'] = 20466275
        pdict['coff'] = 2750.5
        pdict['loff'] = 2750.5
        pdict['a'] = 6378137.0
        pdict['h'] = 35785863.0
        pdict['b'] = 6356752.3
        pdict['ssp_lon'] = 140.7
        pdict['nlines'] = self.nlines
        pdict['ncols'] = self.ncols
        pdict['scandir'] = 'N2S'

        aex = get_area_extent(pdict)

        pdict['a_name'] = 'Himawari_Area'
        pdict['a_desc'] = "AHI Full Disk area"
        pdict['p_id'] = f'geos{self.platform_shortname}'

        return get_area_definition(pdict, aex)