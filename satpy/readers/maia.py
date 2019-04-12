#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017.
#
# Author(s):
#
#   Pascale Roquet <pascale.roquet@meteo.fr>
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
"""Reader for NWPSAF AAPP MAIA Cloud product.

https://nwpsaf.eu/site/software/aapp/

Documentation reference:

    [NWPSAF-MF-UD-003] DATA Formats
    [NWPSAF-MF-UD-009] MAIA version 4 Scientific User Manual

"""
import logging

import h5py
import numpy as np
from xarray import DataArray
import dask.array as da

from satpy.readers.file_handlers import BaseFileHandler
from satpy import CHUNK_SIZE

logger = logging.getLogger(__name__)


class MAIAFileHandler(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(MAIAFileHandler, self).__init__(
            filename, filename_info, filetype_info)
        self.finfo = filename_info
        # set the day date part for end_time from the file name
        self.finfo['end_time'] = self.finfo['end_time'].replace(
            year=self.finfo['start_time'].year,
            month=self.finfo['start_time'].month,
            day=self.finfo['start_time'].day)
        if self.finfo['end_time'] < self.finfo['start_time']:
            myday = self.finfo['end_time'].day
            self.finfo['end_time'] = self.finfo['end_time'].replace(
                day=myday + 1)
        self.selected = None
        self.read(self.filename)

    def read(self, filename):
        self.h5 = h5py.File(filename, 'r')
        missing = -9999.
        self.Lat = da.from_array(self.h5[u'DATA/Latitude'], chunks=CHUNK_SIZE) / 10000.
        self.Lon = da.from_array(self.h5[u'DATA/Longitude'], chunks=CHUNK_SIZE) / 10000.
        self.selected = (self.Lon > missing)
        self.file_content = {}
        for key in self.h5['DATA'].keys():
            self.file_content[key] = da.from_array(self.h5[u'DATA/' + key], chunks=CHUNK_SIZE)
        for key in self.h5[u'HEADER'].keys():
            self.file_content[key] = self.h5[u'HEADER/' + key][:]

        # Cloud Mask on pixel
        mask = 2**0 + 2**1 + 2**2
        lst = self.file_content[u'CloudMask'] & mask
        lst = lst / 2**0
        self.file_content[u"cma"] = lst

        # Cloud Mask confidence
        mask = 2**5 + 2**6
        lst = self.file_content[u'CloudMask'] & mask
        lst = lst / 2**5
        self.file_content[u"cma_conf"] = lst

        # Cloud Mask Quality
        mask = 2**3 + 2**4
        lst = self.file_content[u'CloudMask'] & mask
        lst = lst / 2**3
        self.file_content[u'cma_qual'] = lst

        # Opaque Cloud
        mask = 2**21
        lst = self.file_content[u'CloudMask'] & mask
        lst = lst / 2**21
        self.file_content[u'opaq_cloud'] = lst

        # land /water Background
        mask = 2**15 + 2**16 + 2**17
        lst = self.file_content[u'CloudMask'] & mask
        lst = lst / 2**15
        self.file_content[u'land_water_background'] = lst

        # CT (Actual CloudType)
        mask = 2**4 + 2**5 + 2**6 + 2**7 + 2**8
        classif = self.file_content[u'CloudType'] & mask
        classif = classif / 2**4
        self.file_content['ct'] = classif.astype(np.uint8)

    def get_platform(self, platform):
        if self.file_content['sat_id'] in (14,):
            return "viirs"
        else:
            return "avhrr"

    @property
    def start_time(self):
        return self.finfo['start_time']

    @property
    def end_time(self):
        return self.finfo['end_time']

    def get_dataset(self, key, info, out=None):
        """Get a dataset from the file."""

        logger.debug("Reading %s.", key.name)
        values = self.file_content[key.name]
        selected = np.array(self.selected)
        if key.name in ("Latitude", "Longitude"):
            values = values / 10000.
        if key.name in ('Tsurf', 'CloudTopPres', 'CloudTopTemp'):
            goods = values > -9998.
            selected = np.array(selected & goods)
            if key.name in ('Tsurf', "Alt_surface", "CloudTopTemp"):
                values = values / 100.
            if key.name in ("CloudTopPres"):
                values = values / 10.
        else:
            selected = self.selected
        info.update(self.finfo)

        fill_value = np.nan

        if key.name == 'ct':
            fill_value = 0
            info['_FillValue'] = 0
        ds = DataArray(values, dims=['y', 'x'], attrs=info).where(selected, fill_value)

        # update dataset info with file_info
        return ds
