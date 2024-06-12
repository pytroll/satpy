#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024.
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
"""A reader for NetCDF Hydrology SAF products. In this beta version, only H63 is supported."""
import logging
import os
from contextlib import suppress
from datetime import timedelta
from pyresample import geometry

import dask.array as da
import numpy as np
import xarray as xr
from satpy.readers._geos_area import get_area_definition, get_area_extent

from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.utils import unzip_file
from satpy.utils import get_chunk_size_limit

logger = logging.getLogger(__name__)

CHUNK_SIZE = get_chunk_size_limit()

PLATFORM_NAMES = {"MSG1": "Meteosat-8",
                  "MSG2": "Meteosat-9",
                  "MSG3": "Meteosat-10",
                  "MSG4": "Meteosat-11",
                  "GOES16": "GOES-16",
                  "GOES17": "GOES-17",
                  }

class HSAFNCFileHandler(BaseFileHandler):
    """NWCSAF PPS&MSG NetCDF reader."""
    
    def __init__(self, filename, filename_info, filetype_info):
        """Init method."""
        super(HSAFNCFileHandler, self).__init__(filename, filename_info,
                                                filetype_info)

        self._unzipped = unzip_file(self.filename)
        if self._unzipped:
            self.filename = self._unzipped

        self.cache = {}
        self.nc = xr.open_dataset(self.filename,
                                  decode_cf=True,
                                  mask_and_scale=False,
                                  chunks=CHUNK_SIZE)
        
        if 'xc' in self.nc.dims:
            self.nc = self.nc.rename({"xc": "x", "yc": "y"})
        elif 'nx' in self.nc.dims:
            self.nc = self.nc.rename({"nx": "x", "ny": "y"})

        try:
            kwrgs = {"sat_id": self.nc.attrs["satellite_identifier"]}
        except KeyError:
            kwrgs = {"sat_id": None}

        self.set_platform_and_sensor(**kwrgs)

    def __del__(self):
        """Delete the instance."""
        if self._unzipped:
            try:
                os.remove(self._unzipped)
            except OSError:
                pass

    def set_platform_and_sensor(self, **kwargs):
        """Set some metadata: platform_name, sensors, and pps (identifying PPS or Geo)."""
        self.platform_name = PLATFORM_NAMES.get(kwargs["sat_id"], 'N/A')
        
        self.sensor = "seviri"

    def get_dataset(self, dsid, info):
            """Load a dataset."""
            dsid_name = info["file_key"]
            if dsid_name in self.cache:
                logger.debug("Get the data set from cache: %s.", dsid_name)
                return self.cache[dsid_name]

            logger.debug("Reading %s.", dsid_name)
            variable = self.nc[dsid_name]

            # Data is transposed in file, fix it here
            variable.data = variable.data.T

            variable.attrs["start_time"] = self.start_time
            variable.attrs["end_time"] = self.end_time

            # Fill value is not defined as an attribute, manually specify
            variable.data = da.where(variable.data > 0, variable.data, np.nan)

            return variable
            
    def get_area_def(self, dsid):
        """Get the area definition of the band."""

        val_dict = {}
        for keyval in self.nc.attrs['cgms_projection'].split():
            try:
                key, val = keyval.split('=')
                val_dict[key] = val
            except:
                pass

        pdict = {}
        pdict['scandir'] = 'N2S'
        pdict["ssp_lon"] = np.float32(self.nc.attrs["sub-satellite_longitude"][:-1])
        pdict['a'] = float(val_dict['+r_eq'])*1000
        pdict['b'] = float(val_dict['+r_pol'])*1000
        pdict['h'] = float(val_dict['+h'])*1000 - pdict['a']
        pdict['loff'] = float(val_dict['+loff'])
        pdict['coff'] = float(val_dict['+coff'])
        pdict['lfac'] = -float(val_dict['+lfac'])
        pdict['cfac'] = -float(val_dict['+cfac'])
        pdict['ncols'] = self.nc.x.size
        pdict['nlines'] = self.nc.y.size
        pdict["a_name"] = "seviri_geos_fds"
        pdict["a_desc"] = "SEVIRI full disk area at native resolution"
        pdict["p_id"] = "seviri_fixed_grid"

        area_extent = get_area_extent(pdict)
        fg_area_def = get_area_definition(pdict, area_extent)
        return fg_area_def

    @property
    def start_time(self):
        """Return the start time of the object."""
        return self.filename_info["start_time"]

    @property
    def end_time(self):
        """Return the end time of the object.
        
        The product does not provide the end time, so the start time is used."""
        return self.filename_info["start_time"]