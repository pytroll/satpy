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
"""HY-2B L2B Reader, distributed by Eumetsat in HDF5 format"""

import numpy as np
import xarray as xr
import dask.array as da
from satpy import CHUNK_SIZE
from datetime import datetime

from satpy.readers.hdf5_utils import HDF5FileHandler


class HY2SCATL2BH5FileHandler(HDF5FileHandler):

    @property
    def start_time(self):
        """Time for first observation."""
        return datetime.strptime(self['/attr/Range_Beginning_Time'],
                                 '%Y%m%dT%H:%M:%S')

    @property
    def end_time(self):
        """Time for final observation."""
        return datetime.strptime(self['/attr/Range_Ending_Time'],
                                 '%Y%m%dT%H:%M:%S')

    @property
    def platform_name(self):
        """Platform ShortName"""
        return self['/attr/Platform_ShortName']

    def get_variable_metadata(self):
        info = getattr(self, 'attrs', {})
        info.update({
            "Equator_Crossing_Longitude": self['/attr/Equator_Crossing_Longitude'],
            "Equator_Crossing_Time": self['/attr/Equator_Crossing_Time'],
            "Input_L2A_Filename": self['/attr/Input_L2A_Filename'],
            "L2B_Actual_WVC_Rows": self['/attr/L2B_Actual_WVC_Rows'],
            "Orbit_Inclination": self['/attr/Orbit_Inclination'],
            "Orbit_Number": self['/attr/Orbit_Number'],
            "Output_L2B_Filename": self['/attr/Output_L2B_Filename'],
            "Production_Date_Time": self['/attr/Production_Date_Time'],
            "L2B_Expected_WVC_Rows": self['/attr/L2B_Expected_WVC_Rows'],
            "L2B_Number_WVC_cells": self['/attr/L2B_Number_WVC_cells'],
        })
        return info

    def get_metadata(self):
        info = getattr(self, 'attrs', {})
        info.update({
            "WVC_Size": self['/attr/WVC_Size'],
            "HDF_Version_Id": self['/attr/HDF_Version_Id'],
            "Instrument_ShorName": self['/attr/Instrument_ShorName'],
            "L2A_Inputdata_Version": self['/attr/L2A_Inputdata_Version'],
            "L2B_Algorithm_Descriptor": self['/attr/L2B_Algorithm_Descriptor'],
            "L2B_Data_Version": self['/attr/L2B_Data_Version'],
            "L2B_Processing_Type": self['/attr/L2B_Processing_Type'],
            "L2B_Processor_Name": self['/attr/L2B_Processor_Name'],
            "L2B_Processor_Version": self['/attr/L2B_Processor_Version'],
            "Long_Name": self['/attr/Long_Name'],
            "Platform_LongName": self['/attr/Platform_LongName'],
            "Platform_ShortName": self['/attr/Platform_ShortName'],
            "Platform_Type": self['/attr/Platform_Type'],
            "Producer_Agency": self['/attr/Producer_Agency'],
            "Producer_Institution": self['/attr/Producer_Institution'],
            "Rev_Orbit_Perio": self['/attr/Rev_Orbit_Period'],
            "Short_Name": self['/attr/Short_Name'],
            "Sigma0_Granularity": self['/attr/Sigma0_Granularity'],
        })
        return info

    def get_dataset(self, key, info):
        dims = ['y', 'x']
        if self[key.name].ndim == 3:
            dims = ['y', 'x', 'selection']
        if key.name in 'wvc_row_time':
            data = xr.DataArray(da.from_array(self[key.name][:]),
                                attrs={'fill_value': self[key.name].attrs['fill_value']},
                                name=key.name,
                                dims=['y', ])
        else:
            data = xr.DataArray(da.from_array(self[key.name][:],
                                              chunks=CHUNK_SIZE),
                                name=key.name, dims=dims)

            data = self._mask_data(key.name, data)
            data = self._scale_data(key.name, data)

            if key.name in 'wvc_lon':
                data = xr.where(data > 180, data - 360., data)
        data.attrs.update(info)
        data.attrs.update(self.get_metadata())
        data.attrs.update(self.get_variable_metadata())
        if "Platform_ShortName" in data.attrs:
            data.attrs.update({'platform_name': data.attrs['Platform_ShortName']})

        return data

    def _scale_data(self, key_name, data):
        return data * self[key_name].attrs['scale_factor'] + self[key_name].attrs['add_offset']

    def _mask_data(self, key_name, data):
        data = xr.where(data == self[key_name].attrs['fill_value'], np.nan, data)

        valid_range = self[key_name].attrs['valid range']
        data = xr.where(data < valid_range[0], np.nan, data)
        data = xr.where(data > valid_range[1], np.nan, data)
        return data
