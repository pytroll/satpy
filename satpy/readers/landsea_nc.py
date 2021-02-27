#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2020 Satpy developers
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
"""Land-sea mask from MODIS format reader.

References:
   - The land sea mask from MODIS, downloaded from https://www.ghrsst.org/ghrsst-data-services/tools/
     lsmask_modis_byte.nc
     The NAVOCEANO SST processing software only generates SST retrievals for pixels
     determined to be over water. A global land/sea tag file is utilized within the
     processing software to make the determination before a SST retrieval is generated.
     Prior to April 2002, NAVOCEANO used a global 7km land/sea tag file to make the
     decision. Since April 2002, NAVOCEANO has utilized a 1km land/sea tag file. The
     1km file was derived from a 1km land sea mask from USGS and from the GTOPO30
     land mask from USGS. The land mask covers latitudes 80.3N to 80.3S and all
     longitudes and includes all coastal regions and lakes. Each 1km land mask
     cell also contains the distance that the cell is from land. This value is zero
     over land and up to a maximum of 50km for cells over water.

"""

import logging
import os
from datetime import datetime

import numpy as np
import xarray as xr

from pyresample import get_area_def
from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.utils import unzip_file

import h5py

logger = logging.getLogger(__name__)

SENSOR = {'Terra': 'modis', }

PLATFORM_NAMES = {'Terra': 'Terra', }


class NcLandSea(BaseFileHandler):
    """MODIS land sea mask NetCDF reader."""

    def __init__(self, filename, filename_info, filetype_info):

        """Init method."""
        super(NcLandSea, self).__init__(filename, filename_info,
                                        filetype_info)

        self._unzipped = unzip_file(self.filename)
        if self._unzipped:
            self.filename = self._unzipped

        self.cache = {}
        self.nc = xr.open_dataset(self.filename,
                                  decode_cf=True,
                                  mask_and_scale=False,
                                  chunks=3600)

        self.nc = self.nc.rename({'lon': 'x', 'lat': 'y'})

        self.platform_name = "Terra/Aqua"
        self.sensor = 'modis'

    def get_dataset(self, dsid, info):
        """Load a dataset."""
        dsid_name = dsid.name
        if dsid_name in self.cache:
            logger.debug('Get the data set from cache: %s.', dsid_name)
            return self.cache[dsid_name]
        if dsid_name in ['lon', 'lat'] and dsid_name not in self.nc:
            dsid_name = dsid_name + '_reduced'

        logger.debug('Reading %s.', dsid_name)
        variable = self.nc[dsid_name]
        variable = self.scale_dataset(dsid, variable, info)

        return variable

    def scale_dataset(self, dsid, variable, info):
        """Scale the data set, applying the attributes from the netCDF file.

        The scale and offset attributes will then be removed from the resulting variable.
        """

        for key, val in variable.attrs.items():
            if isinstance(val, h5py._hl.base.Empty):
                variable.attrs.pop(key)

        if '_fill_value' in variable.attrs:
            variable = variable.where(variable != variable.attrs['_fill_value'])

        if 'valid_range' in variable.attrs:
            variable = variable.where(
                variable <= variable.attrs['valid_range'][1])

        attrs = variable.attrs.copy()

        variable.attrs = attrs

        variable.attrs.update({'platform_name': self.platform_name,
                               'sensor': self.sensor})
        if 'units' in variable.attrs:
            variable.attrs.setdefault('units', str(variable.units))
        else:
            variable.attrs.setdefault('units', '1')

        return variable

    def upsample_geolocation(self, dsid, info):
        """Upsample the geolocation (lon,lat) from the tiepoint grid."""
        from geotiepoints import SatelliteInterpolator
        # Read the fields needed:
        col_indices = self.nc['nx_reduced'].values
        row_indices = self.nc['ny_reduced'].values
        lat_reduced = self.scale_dataset(dsid, self.nc['lat_reduced'], info)
        lon_reduced = self.scale_dataset(dsid, self.nc['lon_reduced'], info)

        shape = (self.nc['y'].shape[0], self.nc['x'].shape[0])
        cols_full = np.arange(shape[1])
        rows_full = np.arange(shape[0])

        satint = SatelliteInterpolator((lon_reduced.values, lat_reduced.values),
                                       (row_indices,
                                        col_indices),
                                       (rows_full, cols_full))

        lons, lats = satint.interpolate()
        self.cache['lon'] = xr.DataArray(lons, attrs=lon_reduced.attrs, dims=['y', 'x'])
        self.cache['lat'] = xr.DataArray(lats, attrs=lat_reduced.attrs, dims=['y', 'x'])

        return

    def get_area_def(self, dsid):
        """Get the area definition of the datasets.
           fixed area definition worldeqc4km21
        """

        nlines, ncols = self.nc[dsid.name].shape

        if 'grid_resolution' in self.nc.attrs.keys():
            import re
            dx = float(re.sub(r"\D", "", self.nc.attrs['grid_resolution']))
        else:
            dx = 0

        if 'southernmost_latitude' in self.nc.attrs.keys():
            lat_min = self.nc.attrs['southernmost_latitude'] - dx
        else:
            lat_min = -90.

        if 'northernmost_latitude' in self.nc.attrs.keys():
            lat_max = self.nc.attrs['northernmost_latitude'] + dx
        else:
            lat_max = 90.

        if 'westernmost_longitude' in self.nc.attrs.keys():
            lon_min = self.nc.attrs['westernmost_longitude'] - dx
        else:
            lon_min = -180.

        if 'easternmost_longitude' in self.nc.attrs.keys():
            lon_max = self.nc.attrs['easternmost_longitude'] + dx
        else:
            lon_max = 180.

        area = get_area_def("worldeqc",
                            "World, platecarree",
                            'eqc',
                            '+ellps=WGS84 +lat_0=0 +lat_ts=0 +lon_0=0 +no_defs' +
                            ' +proj=eqc +type=crs +units=m +x_0=0 +y_0=0',
                            ncols,
                            nlines,
                            (-20037508.3428*lon_min/(-180.), -10018754.1714*lat_min/(-90.),
                             20037508.3428*lon_max/(180.),    10018754.1714*lat_max/90.))

        return area

    def __del__(self):
        """Delete the instance."""
        if self._unzipped:
            try:
                os.remove(self._unzipped)
            except (IOError, OSError):
                pass

    @property
    def start_time(self):
        """Return the start time of the object."""
        if "creation_date" in self.nc.attrs.keys():
            start_time = datetime.strptime(self.nc.attrs["creation_date"], "%Y-%m-%d")
        else:
            start_time = datetime.strptime("2007-06-28 00:00:00", "%Y-%m-%d %H:%M:%S")
        return start_time

    @property
    def end_time(self):
        """Return the end time of the object."""
        if "creation_date" in self.nc.attrs.keys():
            end_time = datetime.strptime(self.nc.attrs["creation_date"], "%Y-%m-%d")
        else:
            end_time = datetime.strptime("2007-06-28 00:00:00", "%Y-%m-%d %H:%M:%S")
        return end_time
