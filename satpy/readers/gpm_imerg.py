#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 Satpy developers
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
"""Reader for GPM imerg data on half-hourly timesteps.

References:
   - The NASA IMERG ATBD:
     https://pmm.nasa.gov/sites/default/files/document_files/IMERG_ATBD_V06.pdf

"""

import logging
from datetime import datetime
import numpy as np
from satpy.readers.hdf5_utils import HDF5FileHandler
from pyresample.geometry import AreaDefinition
import h5py

logger = logging.getLogger(__name__)


class Hdf5IMERG(HDF5FileHandler):
    """IMERG hdf5 reader."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init method."""
        super(Hdf5IMERG, self).__init__(filename, filename_info,
                                        filetype_info)
        self.finfo = filename_info
        self.cache = {}

    @property
    def start_time(self):
        """Find the start time from filename info."""
        return datetime(self.finfo['date'].year,
                        self.finfo['date'].month,
                        self.finfo['date'].day,
                        self.finfo['start_time'].hour,
                        self.finfo['start_time'].minute,
                        self.finfo['start_time'].second)

    @property
    def end_time(self):
        """Find the end time from filename info."""
        return datetime(self.finfo['date'].year,
                        self.finfo['date'].month,
                        self.finfo['date'].day,
                        self.finfo['end_time'].hour,
                        self.finfo['end_time'].minute,
                        self.finfo['end_time'].second)

    def get_dataset(self, dataset_id, ds_info):
        """Load a dataset."""
        file_key = ds_info.get('file_key', dataset_id.name)
        dsname = 'Grid/' + file_key
        data = self[dsname].squeeze().transpose()
        data.values = np.flipud(data.values)

        fill = data.attrs['_FillValue']
        pts = (data.values == fill).nonzero()
        data.values[pts] = np.nan

        for key in list(data.attrs.keys()):
            val = data.attrs[key]
            if isinstance(val, h5py.h5r.Reference):
                del data.attrs[key]

        return data

    def get_area_def(self, dsid):
        """Create area definition from the gridded lat/lon values."""
        lats = self.__getitem__('Grid/lat').values
        lons = self.__getitem__('Grid/lon').values

        width = lons.shape[0]
        height = lats.shape[0]

        lower_left_x = lons[0]
        lower_left_y = lats[0]

        upper_right_x = lons[-1]
        upper_right_y = lats[-1]

        area_extent = (lower_left_x, lower_left_y, upper_right_x, upper_right_y)
        description = "IMERG GPM Equirectangular Projection"
        area_id = 'imerg'
        proj_id = 'equirectangular'
        proj_dict = {'proj': 'longlat', 'datum': 'WGS84', 'ellps': 'WGS84', }
        area_def = AreaDefinition(area_id, description, proj_id, proj_dict, width, height, area_extent, )
        return area_def
