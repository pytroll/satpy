#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Martin Raspaud

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Compact viirs format.
"""

import logging
from datetime import datetime

import numpy as np

import h5netcdf
from satpy.projectable import Projectable
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)


class NCOLCI1B(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(NCOLCI1B, self).__init__(filename, filename_info,
                                       filetype_info)
        self.nc = h5netcdf.File(filename, 'r')
        self.channel = filename_info['dataset_name']

    def get_dataset(self, key, info):
        """Load a dataset
        """
        if self.channel != key.name:
            return
        logger.debug('Reading %s.', key.name)
        variable = self.nc[self.channel + '_radiance']
        proj = Projectable(np.ma.masked_equal(variable[:],
                                              variable.attrs['_FillValue']) *
                           variable.attrs['scale_factor'] +
                           variable.attrs['add_offset'],
                           units=variable.attrs['units'])
        return proj

    @property
    def start_time(self):
        return datetime.strptime(self.nc.attrs['start_time'], '%Y-%m-%dT%H:%M:%S.%fZ')

    @property
    def end_time(self):
        return datetime.strptime(self.nc.attrs['stop_time'], '%Y-%m-%dT%H:%M:%S.%fZ')
