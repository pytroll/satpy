#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016.
#
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

"""Reader for Vaisala Global Lightning Dataset 360 products
"""

import logging
import pandas as pd
import dask.array as da
import xarray as xr

from satpy import CHUNK_SIZE
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)


class VaisalaGLD360FileHandler(BaseFileHandler):
    """Vaisala Global Lightning Dataset Reader."""

    def __init__(self, filename, filename_info, filetype_info):
        super(VaisalaGLD360FileHandler, self).__init__(filename, filename_info, filetype_info)

        names = ['date', 'time', 'latitude', 'longitude', 'power', 'unit']
        types = ['str', 'str', 'float', 'float', 'float', 'str']
        dtypes = dict(zip(names, types))
        parse_dates = {'datetime': ['date', 'time']}

        self.data = pd.read_csv(filename, delim_whitespace=True, header=None,
                                names=names, dtype=dtypes, parse_dates=parse_dates)

    @property
    def start_time(self):
        return self.data['datetime'].iloc[0]

    @property
    def end_time(self):
        # import ipdb; ipdb.set_trace()
        return self.data['datetime'].iloc[-1]

    def get_dataset(self, dataset_id, dataset_info):
        """Load a dataset
        """
        xarr = xr.DataArray(da.from_array(self.data[dataset_id.name],
                                          chunks=CHUNK_SIZE), dims=["y"])
        xarr.attrs.update(dataset_info)

        return xarr
