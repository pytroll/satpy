#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023
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


"""A reader for the Top of Atmosphere outgoing fluxes from the Geostationary Earth Radiation
Budget instrument aboard the Meteosat Second Generation satellites."""


import logging
from datetime import timedelta

import dask.array as da
import h5py
import numpy as np
import xarray as xr

from satpy.readers.file_handlers import BaseFileHandler
from satpy.resample import get_area_def

LOG = logging.getLogger(__name__)

def gerb_get_dataset(hfile, name):
    """
    Load a GERB dataset in memory from a HDF5 file

    The routine takes into account the quantisation factor and fill values.
    """
    ds = hfile[name]
    if 'Quantisation Factor' in ds.attrs and 'Unit' in ds.attrs:
        ds_real = ds[...]*ds.attrs['Quantisation Factor']
    else:
        ds_real = ds[...]*1.
    ds_min = ds[...].min()
    if ds_min < 0:
        mask = ds == ds_min
        ds_real[mask] = np.nan
    return ds_real


class GERB_HR_FileHandler(BaseFileHandler):
    """File handler for GERB L2 High Resolution H5 files."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init the file handler."""
        super(GERB_HR_FileHandler, self).__init__(filename,
                                                  filename_info,
                                                  filetype_info)
        self._h5fh = h5py.File(self.filename, 'r')
        self.ssp_lon = self._h5fh["Geolocation"].attrs["Nominal Satellite Longitude (degrees)"][()]

    @property
    def end_time(self):
        """Get end time."""
        return self.start_time + timedelta(minutes=14, seconds=59)


    @property
    def start_time(self):
        """Get start time."""
        return self.filename_info['sensing_time']


    def _get_dataset(self, ds_name):
        """Access the GERB dataset from the HDF5 file."""
        if ds_name in ['Solar Flux', 'Thermal Flux', 'Solar Radiance', 'Thermal Radiance']:
            return gerb_get_dataset(self._h5fh, f'Radiometry/{ds_name}')
        else:
            raise ValueError


    def get_dataset(self, ds_id, ds_info):
        """Read a HDF5 file into an xarray DataArray."""
        ds = self._get_dataset(ds_id['name'])
        ds_info = {}

        ds_info['start_time'] = self.start_time
        ds_info['data_time'] = self.start_time
        ds_info['end_time'] = self.end_time

        data = da.from_array(ds)
        return xr.DataArray(data, attrs=ds_info, dims=('y', 'x'))


    def get_area_def(self, dsid):
        """Area definition for the GERB product"""

        if abs(self.ssp_lon) < 1e-6:
            return get_area_def("msg_seviri_fes_9km")
        elif abs(self.ssp_lon - 9.5) < 1e-6:
            return get_area_def("msg_seviri_fes_9km")
        elif abs(self.ssp_lon - 45.5) < 1e-6:
            return get_area_def("msg_seviri_iodc_9km")
        else:
            raise ValueError

