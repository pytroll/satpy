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


"""GERB L2 HR HDF5 reader.

A reader for the Top of Atmosphere outgoing fluxes from the Geostationary Earth Radiation
Budget instrument aboard the Meteosat Second Generation satellites.
"""


import logging
from datetime import timedelta

from satpy.readers.hdf5_utils import HDF5FileHandler
from satpy.resample import get_area_def

LOG = logging.getLogger(__name__)


def gerb_get_dataset(ds, ds_info):
    """
    Load a GERB dataset in memory from a HDF5 file or HDF5FileHandler.

    The routine takes into account the quantisation factor and fill values.
    """
    ds_attrs = ds.attrs
    ds_fill = ds_info['fill_value']
    fill_mask = ds != ds_fill
    if 'Quantisation Factor' in ds_attrs and 'Unit' in ds_attrs:
        ds = ds*ds_attrs['Quantisation Factor']
    else:
        ds = ds*1.
    ds = ds.where(fill_mask)
    return ds


class GERB_HR_FileHandler(HDF5FileHandler):
    """File handler for GERB L2 High Resolution H5 files."""

    @property
    def end_time(self):
        """Get end time."""
        return self.start_time + timedelta(minutes=15)

    @property
    def start_time(self):
        """Get start time."""
        return self.filename_info['sensing_time']

    def get_dataset(self, ds_id, ds_info):
        """Read a HDF5 file into an xarray DataArray."""
        ds_name = ds_id['name']
        if ds_name not in ['Solar Flux', 'Thermal Flux', 'Solar Radiance', 'Thermal Radiance']:
            raise KeyError(f"{ds_name} is an unknown dataset for this reader.")

        ds = gerb_get_dataset(self[f'Radiometry/{ds_name}'], ds_info)

        ds.attrs.update({'start_time': self.start_time, 'data_time': self.start_time, 'end_time': self.end_time})

        return ds

    def get_area_def(self, dsid):
        """Area definition for the GERB product."""
        ssp_lon = self.file_content["Geolocation/attr/Nominal Satellite Longitude (degrees)"]

        if abs(ssp_lon) < 1e-6:
            return get_area_def("msg_seviri_fes_9km")
        elif abs(ssp_lon - 9.5) < 1e-6:
            return get_area_def("msg_seviri_fes_9km")
        elif abs(ssp_lon - 45.5) < 1e-6:
            return get_area_def("msg_seviri_iodc_9km")
        else:
            raise ValueError(f"There is no matching grid for SSP longitude {self.ssp_lon}")
