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
"""SEVIRI L2 BUFR format reader."""


import logging
from datetime import timedelta, datetime
import numpy as np
import xarray as xr
import dask.array as da
from satpy.readers.seviri_base import mpef_product_header
from satpy.readers.eum_base import recarray2dict

try:
    import eccodes as ec
except ImportError:
    raise ImportError(
        "Missing eccodes-python and/or eccodes C-library installation. Use conda to install eccodes")

from satpy.readers.file_handlers import BaseFileHandler
from satpy import CHUNK_SIZE

logger = logging.getLogger('SeviriL2Bufr')

data_center_dict = {55: {'ssp': 'E0415', 'name': '08'}, 56:  {'ssp': 'E0000', 'name': '09'},
                    57: {'ssp': 'E0095', 'name': '10'}, 70: {'ssp': 'E0000', 'name': '11'}}

seg_size_dict = {'seviri_l2_bufr_asr': 16, 'seviri_l2_bufr_cla': 16,
                 'seviri_l2_bufr_csr': 16, 'seviri_l2_bufr_gii': 3,
                 'seviri_l2_bufr_thu': 16, 'seviri_l2_bufr_toz': 3}


class SeviriL2BufrFileHandler(BaseFileHandler):
    """File handler for SEVIRI L2 BUFR products."""

    def __init__(self, filename, filename_info, filetype_info, **kwargs):
        """Initialise the file handler for SEVIRI L2 BUFR data."""
        super(SeviriL2BufrFileHandler, self).__init__(filename,
                                                      filename_info,
                                                      filetype_info)

        if ('server' in filename_info):
            # EUMETSAT Offline Bufr product
            self.mpef_header = self._read_mpef_header()
        else:
            # Product was retrieved from the EUMETSAT Data Center
            timeStr = self.get_attribute('typicalDate')+self.get_attribute('typicalTime')
            buf_start_time = datetime.strptime(timeStr, "%Y%m%d%H%M%S")
            sc_id = self.get_attribute('satelliteIdentifier')
            self.mpef_header = {}
            self.mpef_header['NominalTime'] = buf_start_time
            self.mpef_header['SpacecraftName'] = data_center_dict[sc_id]['name']
            self.mpef_header['RectificationLongitude'] = data_center_dict[sc_id]['ssp']

        self.seg_size = seg_size_dict[filetype_info['file_type']]

    @property
    def start_time(self):
        """Return the repeat cycle start time."""
        return self.mpef_header['NominalTime']

    @property
    def end_time(self):
        """Return the repeat cycle end time."""
        return self.start_time + timedelta(minutes=15)

    @property
    def platform_name(self):
        """Return spacecraft name."""
        return 'MET{}'.format(self.mpef_header['SpacecraftName'])

    @property
    def ssp_lon(self):
        """Return subsatellite point longitude."""
        # e.g. E0415
        ssp_lon = self.mpef_header['RectificationLongitude']
        return float(ssp_lon[1:])/10.

    def _read_mpef_header(self):
        """Read MPEF header."""
        hdr = np.fromfile(self.filename, mpef_product_header, 1)
        return recarray2dict(hdr)

    def get_attribute(self, key):
        ''' Get BUFR attributes '''
        # This function is inefficient as it is looping through the entire
        # file to get 1 attribute. It causes a problem though if you break
        # from the file early - dont know why but investigating - fix later
        fh = open(self.filename, "rb")
        while True:
            # get handle for message
            bufr = ec.codes_bufr_new_from_file(fh)
            if bufr is None:
                break
            ec.codes_set(bufr, 'unpack', 1)
            attr = ec.codes_get(bufr, key)
            ec.codes_release(bufr)

        fh.close()
        return attr

    def get_array(self, key):
        """Get all data from file for the given BUFR key."""
        with open(self.filename, "rb") as fh:
            msgCount = 0
            while True:
                bufr = ec.codes_bufr_new_from_file(fh)
                if bufr is None:
                    break

                ec.codes_set(bufr, 'unpack', 1)

                # if is the first message initialise our final array
                if (msgCount == 0):
                    arr = da.from_array(ec.codes_get_array(
                        bufr, key, float), chunks=CHUNK_SIZE)
                else:
                    tmpArr = da.from_array(ec.codes_get_array(
                        bufr, key, float), chunks=CHUNK_SIZE)
                    arr = da.concatenate((arr, tmpArr))

                msgCount = msgCount+1
                ec.codes_release(bufr)

        if arr.size == 1:
            arr = arr[0]

        return arr

    def get_dataset(self, dataset_id, dataset_info):
        """Get dataset using the BUFR key in dataset_info."""
        arr = self.get_array(dataset_info['key'])
        arr[arr == dataset_info['fill_value']] = np.nan

        xarr = xr.DataArray(arr, dims=["y"])
        xarr.attrs['sensor'] = 'SEVIRI'
        xarr.attrs['platform_name'] = self.platform_name
        xarr.attrs['ssp_lon'] = self.ssp_lon
        xarr.attrs['seg_size'] = self.seg_size
        xarr.attrs.update(dataset_info)

        return xarr
