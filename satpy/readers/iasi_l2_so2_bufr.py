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
r"""IASI L2 SO2 BUFR format reader.

Introduction
------------
The ``iasi_l2_so2_bufr`` reader reads IASI level2 SO2 data in BUFR format. The algorithm is described in the
Theoretical Basis Document, linked below.

Each BUFR file consists of a number of messages, one for each scan, each of which contains SO2 column amounts
in Dobson units for retrievals performed with plume heights of 7, 10, 13, 16 and 25 km.

Reader Arguments
----------------
A list of retrieval files, fnames, can be opened as follows::

  Scene(reader="iasi_l2_so2_bufr", filenames=fnames)

Example
-------
Here is an example how to read the data in satpy:

.. code-block:: python

    from satpy import Scene
    import glob

    filenames = glob.glob(
        '/test_data/W_XX-EUMETSAT-Darmstadt,SOUNDING+SATELLITE,METOPA+IASI_C_EUMC_20200204091455_68984_eps_o_so2_l2.bin')
    scn = Scene(filenames=filenames, reader='iasi_l2_so2_bufr')
    scn.load(['so2_height_3', 'so2_height_4'])
    print(scn['so2_height_3'])


Output:

.. code-block:: none

    <xarray.DataArray 'so2_height_3' (y: 23, x: 120)>
    dask.array<where, shape=(23, 120), dtype=float64, chunksize=(1, 120), chunktype=numpy.ndarray>
    Coordinates:
        crs      object +proj=latlong +datum=WGS84 +ellps=WGS84 +type=crs
    Dimensions without coordinates: y, x
    Attributes:
        sensor:               IASI
        units:                dobson
        file_type:            iasi_l2_so2_bufr
        wavelength:           None
        modifiers:            ()
        platform_name:        METOP-2
        resolution:           12000
        fill_value:           -1e+100
        level:                None
        polarization:         None
        coordinates:          ('longitude', 'latitude')
        calibration:          None
        key:                  #3#sulphurDioxide
        name:                 so2_height_3
        start_time:           2020-02-04 09:14:55
        end_time:             2020-02-04 09:17:51
        area:                 Shape: (23, 120)\nLons: <xarray.DataArray 'longitud...
        ancillary_variables:  []

References:
Algorithm Theoretical Basis Document:
https://acsaf.org/docs/atbd/Algorithm_Theoretical_Basis_Document_IASI_SO2_Jul_2016.pdf
"""


# TDB: this reader is based on iasi_l2.py and seviri_l2_bufr.py

import logging
from datetime import datetime

import dask.array as da
import numpy as np
import xarray as xr

try:
    import eccodes as ec
except ImportError as e:
    raise ImportError(
        """Missing eccodes-python and/or eccodes C-library installation. Use conda to install eccodes.
           Error: """, e)

from satpy import CHUNK_SIZE
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger('IASIL2SO2BUFR')

data_center_dict = {3: 'METOP-1', 4: 'METOP-2', 5: 'METOP-3'}


class IASIL2SO2BUFR(BaseFileHandler):
    """File handler for the IASI L2 SO2 BUFR product."""

    def __init__(self, filename, filename_info, filetype_info, **kwargs):
        """Initialise the file handler for the IASI L2 SO2 BUFR data."""
        super(IASIL2SO2BUFR, self).__init__(filename, filename_info, filetype_info)

        start_time, end_time = self.get_start_end_date()

        sc_id = self.get_attribute('satelliteIdentifier')

        self.metadata = {}
        self.metadata['start_time'] = start_time
        self.metadata['end_time'] = end_time
        self.metadata['SpacecraftName'] = data_center_dict[sc_id]

    @property
    def start_time(self):
        """Return the start time of data acqusition."""
        return self.metadata['start_time']

    @property
    def end_time(self):
        """Return the end time of data acquisition."""
        return self.metadata['end_time']

    @property
    def platform_name(self):
        """Return spacecraft name."""
        return '{}'.format(self.metadata['SpacecraftName'])

    def get_start_end_date(self):
        """Get the first and last date from the bufr file."""
        fh = open(self.filename, "rb")
        i = 0
        while True:
            # get handle for message
            bufr = ec.codes_bufr_new_from_file(fh)
            if bufr is None:
                break
            ec.codes_set(bufr, 'unpack', 1)
            year = ec.codes_get(bufr, 'year')
            month = ec.codes_get(bufr, 'month')
            day = ec.codes_get(bufr, 'day')
            hour = ec.codes_get(bufr, 'hour')
            minute = ec.codes_get(bufr, 'minute')
            second = ec.codes_get(bufr, 'second')

            obs_time = datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)

            if i == 0:
                start_time = obs_time

            ec.codes_release(bufr)

            i += 1

        end_time = obs_time

        fh.close()

        return start_time, end_time

    def get_attribute(self, key):
        """Get BUFR attributes."""
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

                values = ec.codes_get_array(
                        bufr, key, float)

                if len(values) == 1:
                    values = np.repeat(values, 120)

                # if is the first message initialise our final array
                if (msgCount == 0):

                    arr = da.from_array([values], chunks=CHUNK_SIZE)
                else:
                    tmpArr = da.from_array([values], chunks=CHUNK_SIZE)

                    arr = da.concatenate((arr, tmpArr), axis=0)

                msgCount = msgCount+1
                ec.codes_release(bufr)

        if arr.size == 1:
            arr = arr[0]

        return arr

    def get_dataset(self, dataset_id, dataset_info):
        """Get dataset using the BUFR key in dataset_info."""
        arr = self.get_array(dataset_info['key'])
        arr[arr == dataset_info['fill_value']] = np.nan

        xarr = xr.DataArray(arr, dims=["y", "x"], name=dataset_info['name'])
        xarr.attrs['sensor'] = 'IASI'
        xarr.attrs['platform_name'] = self.platform_name
        xarr.attrs.update(dataset_info)

        return xarr
