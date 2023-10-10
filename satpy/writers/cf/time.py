#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2023 Satpy developers
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
"""CF processing of time dimension and coordinates."""
import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


EPOCH = u"seconds since 1970-01-01 00:00:00"


def add_time_bounds_dimension(ds, time="time"):
    """Add time bound dimension to xr.Dataset."""
    start_times = []
    end_times = []
    for _var_name, data_array in ds.items():
        start_times.append(data_array.attrs.get("start_time", None))
        end_times.append(data_array.attrs.get("end_time", None))

    start_time = min(start_time for start_time in start_times
                     if start_time is not None)
    end_time = min(end_time for end_time in end_times
                   if end_time is not None)
    ds['time_bnds'] = xr.DataArray([[np.datetime64(start_time),
                                     np.datetime64(end_time)]],
                                   dims=['time', 'bnds_1d'])
    ds[time].attrs['bounds'] = "time_bnds"
    ds[time].attrs['standard_name'] = "time"
    return ds


def process_time_coord(dataarray, epoch):
    """Process the 'time' coordinate, if existing.

    It expand the DataArray with a time dimension if does not yet exists.

    The function assumes

        - that x and y dimensions have at least shape > 1
        - the time coordinate has size 1

    """
    if 'time' in dataarray.coords:
        dataarray['time'].encoding['units'] = epoch
        dataarray['time'].attrs['standard_name'] = 'time'
        dataarray['time'].attrs.pop('bounds', None)

        if 'time' not in dataarray.dims and dataarray["time"].size not in dataarray.shape:
            dataarray = dataarray.expand_dims('time')

    return dataarray
