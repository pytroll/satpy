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
"""Reader for the old NWCSAF/Geo (v2013 and earlier) cloud product format.

References:
   - The NWCSAF GEO 2013 products documentation:
     http://www.nwcsaf.org/web/guest/archive - Search for Code "ICD/3"; Type
     "MSG" and the box to the right should say 'Status' (which means any
     status). Version 7.0 seems to be for v2013

     http://www.nwcsaf.org/aemetRest/downloadAttachment/2623

"""

import logging
from datetime import datetime
import numpy as np
from satpy.readers.hdf5_utils import HDF5FileHandler
from pyresample.geometry import AreaDefinition
import h5py

logger = logging.getLogger(__name__)

PLATFORM_NAMES = {'MSG1': 'Meteosat-8',
                  'MSG2': 'Meteosat-9',
                  'MSG3': 'Meteosat-10',
                  'MSG4': 'Meteosat-11', }


class Hdf5NWCSAF(HDF5FileHandler):
    """NWCSAF MSG hdf5 reader."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init method."""
        super(Hdf5NWCSAF, self).__init__(filename, filename_info,
                                         filetype_info)

        self.cache = {}

    def get_dataset(self, dataset_id, ds_info):
        """Load a dataset."""
        file_key = ds_info.get('file_key', dataset_id.name)
        data = self[file_key]

        nodata = None
        if 'SCALING_FACTOR' in data.attrs and 'OFFSET' in data.attrs:
            dtype = np.dtype(data.data)
            if dataset_id.name in ['ctth_alti']:
                data.attrs['valid_range'] = (0, 27000)
                data.attrs['_FillValue'] = np.nan

            if dataset_id.name in ['ctth_alti', 'ctth_pres', 'ctth_tempe', 'ctth_effective_cloudiness']:
                dtype = np.dtype('float32')
                nodata = 255

            if dataset_id.name in ['ct']:
                data.attrs['valid_range'] = (0, 20)
                data.attrs['_FillValue'] = 255
                # data.attrs['palette_meanings'] = list(range(21))

            attrs = data.attrs
            scaled_data = (data * data.attrs['SCALING_FACTOR'] + data.attrs['OFFSET']).astype(dtype)
            if nodata:
                scaled_data = scaled_data.where(data != nodata)
                scaled_data = scaled_data.where(scaled_data >= 0)
            data = scaled_data
            data.attrs = attrs

        for key in list(data.attrs.keys()):
            val = data.attrs[key]
            if isinstance(val, h5py.h5r.Reference):
                del data.attrs[key]

        return data

    def get_area_def(self, dsid):
        """Get the area definition of the datasets in the file."""
        if dsid.name.endswith('_pal'):
            raise NotImplementedError

        cfac = self.file_content['/attr/CFAC']
        lfac = self.file_content['/attr/LFAC']
        coff = self.file_content['/attr/COFF']
        loff = self.file_content['/attr/LOFF']
        numcols = int(self.file_content['/attr/NC'])
        numlines = int(self.file_content['/attr/NL'])

        aex = get_area_extent(cfac, lfac, coff, loff, numcols, numlines)
        pname = self.file_content['/attr/PROJECTION_NAME']
        proj = {}
        if pname.startswith("GEOS"):
            proj["proj"] = "geos"
            proj["a"] = "6378169.0"
            proj["b"] = "6356583.8"
            proj["h"] = "35785831.0"
            proj["lon_0"] = str(float(pname.split("<")[1][:-1]))
        else:
            raise NotImplementedError("Only geos projection supported yet.")

        area_def = AreaDefinition(self.file_content['/attr/REGION_NAME'],
                                  self.file_content['/attr/REGION_NAME'],
                                  pname,
                                  proj,
                                  numcols,
                                  numlines,
                                  aex)

        return area_def

    @property
    def start_time(self):
        """Return the start time of the object."""
        return datetime.strptime(self.file_content['/attr/IMAGE_ACQUISITION_TIME'], '%Y%m%d%H%M')


def get_area_extent(cfac, lfac, coff, loff, numcols, numlines):
    """Get the area extent from msg parameters."""
    xur = (numcols - coff) * 2 ** 16 / (cfac * 1.0)
    xur = np.deg2rad(xur) * 35785831.0
    xll = (-1 - coff) * 2 ** 16 / (cfac * 1.0)
    xll = np.deg2rad(xll) * 35785831.0
    xres = (xur - xll) / numcols
    xur, xll = xur - xres / 2, xll + xres / 2
    yll = (numlines - loff) * 2 ** 16 / (-lfac * 1.0)
    yll = np.deg2rad(yll) * 35785831.0
    yur = (-1 - loff) * 2 ** 16 / (-lfac * 1.0)
    yur = np.deg2rad(yur) * 35785831.0
    yres = (yur - yll) / numlines
    yll, yur = yll + yres / 2, yur - yres / 2

    return xll, yll, xur, yur
