#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2019 Satpy developers
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

"""SEVIRI Bufr  format reader."""


import logging
from datetime import timedelta
import numpy as np
import xarray as xr
import dask.array as da

from satpy.resample import get_area_def

import eccodes as ec

from satpy.readers.file_handlers import BaseFileHandler
from satpy import CHUNK_SIZE

logger = logging.getLogger('BufrProductClasses')


sub_sat_dict = {"E0000": 0.0, "E0415": 41.5, "E0095": 9.5}
seg_area_dict = {"E0000": 'seviri_0deg', "E0415": 'seviri_iodc', "E0095": 'seviri_rss'}
seg_size_dict = {'seviri_l2_bufr_asr': 16, 'seviri_l2_bufr_cla': 16,
                 'seviri_l2_bufr_csr': 16, 'seviri_l2_bufr_gii': 3,
                 'seviri_l2_bufr_thu': 16, 'seviri_l2_bufr_toz': 3}


class MSGBUFRFileHandler(BaseFileHandler):
    """File handler for MSG BUFR data."""

    def __init__(self, filename, filename_info, filetype_info, **kwargs):
        """Initialise the File handler for MSG BUFR data."""
        super(MSGBUFRFileHandler, self).__init__(filename,
                                                 filename_info,
                                                 filetype_info)
        self.platform_name = filename_info['satellite']
        self.subsat = filename_info['subsat']
        self.rc_start = filename_info['start_time']
        self.filename = filename
        self.ssp_lon = sub_sat_dict[filename_info['subsat']]

        seg_size = seg_size_dict[filetype_info['file_type']]
        # here we get the latitude and longitude arrays used for the
        # segment geolocation
        lats = self.get_array('latitude')
        lons = self.get_array('longitude')

        # Use the subsat point to determine the area definition to use for
        # the geo location
        # reset the wight and height based on the segment size
        adef = get_area_def(seg_area_dict[self.subsat])
        adef.height = int(3712/seg_size)
        adef.width = int(3712/seg_size)

        # convert the lons/lats to rows and columns
        # compute is required as the lonlat2colrow function is not dask ready
        self.rows, self.cols = adef.lonlat2colrow(lons.compute(), lats.compute())
        self.nrows = int(np.ceil(3712.0/seg_size))
        self.ncols = int(np.ceil(3712.0/seg_size))

    @property
    def start_time(self):
        """Return the repeat cycle start time."""
        return self.rc_start

    @property
    def end_time(self):
        """Return the repeat cycle end time."""
        return self.rc_start+timedelta(minutes=15)

    def get_array(self, parameter):
        """Get data from BUFR file."""
        with open(self.filename, "rb") as fh:
            msgCount = 0
            while True:
                bufr = ec.codes_bufr_new_from_file(fh)
                if bufr is None:
                    break

                ec.codes_set(bufr, 'unpack', 1)
                # if is the first message initialise our final array
                if (msgCount == 0):
                    arr = da.from_array(ec.codes_get_array(bufr, parameter, float))
                else:
                    tmpArr = da.from_array(ec.codes_get_array(bufr, parameter, float))
                    arr = np.concatenate((arr, tmpArr))

                msgCount = msgCount+1
                ec.codes_release(bufr)

        if arr.size == 1:
            arr = arr[0]

        return arr

    def get_dataset(self, dsid, info):
        """Loop through the BUFR file for the required key and read array."""
        arr2 = np.empty((self.ncols, self.nrows)).astype(np.float)
        arr2.fill(np.nan)

        parameter = info['key']
        arr = self.get_array(parameter)
        arr[arr == info['fill_value']] = np.nan

        try:
            arr2[self.cols, self.rows] = da.from_array(arr.compute(), chunks=(CHUNK_SIZE))
            xarr = xr.DataArray(arr2, dims=['y', 'x'])
        except ValueError:
            print('mismatch in the BUFR data, lat/lon array has more entries than data array')
            print('This can happen if some but not all parameters have valid data')
        except Exception:
            print('BUFR data is corrupt')

        if xarr is None:
            dataset = None
        else:
            dataset = xarr

            dataset.attrs['units'] = info['units']
            dataset.attrs['wavelength'] = info['wavelength']
            dataset.attrs['standard_name'] = info['standard_name']
            dataset.attrs['platform_name'] = self.platform_name
            dataset.attrs['sensor'] = 'seviri'

        return dataset
