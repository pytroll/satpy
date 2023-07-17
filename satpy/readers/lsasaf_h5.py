#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023 Satpy developers
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
"""Nowcasting SAF common PPS&MSG NetCDF/CF format reader.

References:
   - The LSASAF products documentation: https://landsaf.ipma.pt/en/data/products/

"""

import logging
from datetime import datetime

import numpy as np
import xarray as xr
from pyproj import CRS

from satpy.readers._geos_area import get_area_definition, get_area_extent
from satpy.readers.file_handlers import BaseFileHandler
from satpy.utils import get_legacy_chunk_size

logger = logging.getLogger(__name__)

CHUNK_SIZE = get_legacy_chunk_size()

PLATFORM_NAMES = {'MSG1': 'Meteosat-8',
                  'MSG2': 'Meteosat-9',
                  'MSG3': 'Meteosat-10',
                  'MSG4': 'Meteosat-11',
                  }

INST_NAMES = {'SEVI': 'SEVIRI'}


class H5LSASAF(BaseFileHandler):
    """LSA-SAF MSG HDF5 reader."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init method."""
        super(H5LSASAF, self).__init__(filename, filename_info,
                                       filetype_info)

        self.cache = {}
        self.nc = xr.open_dataset(self.filename,
                                  decode_cf=True,
                                  mask_and_scale=False,
                                  chunks=CHUNK_SIZE)

        self.nc = self.nc.rename({'phony_dim_0': 'x', 'phony_dim_1': 'y'})

        self.platform_name = self.nc.attrs['SATELLITE'][0].rstrip()
        self.sensor = self._get_inst_name()

    def _get_inst_name(self):
        """Get the instrument name."""
        inst_id = self.nc.attrs['INSTRUMENT_ID'][0].rstrip()
        try:
            return INST_NAMES[inst_id]
        except KeyError:
            raise KeyError(f'Unknown instrument ID: {inst_id}')

    def get_dataset(self, dsid, info):
        """Load a dataset."""
        dsid_name = dsid['name']
        if dsid_name in self.cache:
            logger.debug('Get the data set from cache: %s.', dsid_name)
            return self.cache[dsid_name]

        logger.debug('Reading %s.', dsid_name)
        file_key = info.get('file_key', dsid_name)
        variable = self.nc[file_key]
        variable.data = variable.data.transpose()
        variable = self.scale_dataset(variable)

        variable.attrs["start_time"] = self.start_time
        variable.attrs["end_time"] = self.end_time

        return variable

    def scale_dataset(self, variable):
        """Scale the data set, applying the attributes from the netCDF file.

        The scale and offset attributes will then be removed from the resulting variable.
        """
        variable = remove_empties(variable)

        scale = variable.attrs.get('SCALING_FACTOR', np.array(1))
        offset = variable.attrs.get('OFFSET', np.array(0))
        attrs = variable.attrs.copy()
        variable = variable / scale + offset
        variable.attrs = attrs
        if 'MISSING_VALUE' in variable.attrs:
            variable.attrs['MISSING_VALUE'] = variable.attrs['MISSING_VALUE'] / scale + offset

        variable = self._mask_variable(variable)
        variable.attrs.pop('add_offset', None)
        variable.attrs.pop('scale_factor', None)

        variable.attrs.update({'platform_name': self.platform_name,
                               'sensor': self.sensor})

        return variable

    @staticmethod
    def _mask_variable(variable):
        if 'MISSING_VALUE' in variable.attrs:
            variable = variable.where(
                variable != variable.attrs['MISSING_VALUE'])
            variable.attrs['MISSING_VALUE'] = np.nan
        return variable

    def _get_proj_lon(self):
        """Find projection longitude from the projection name info.

        LSA SAF data doesn't contain the projected subsatellite longitude, only the actual satellite longitude.
        So we have to find the projection longitude from the name of the projection given in the attributes.
        """
        proj_name = self.nc.attrs['PROJECTION_NAME']
        pos_s = proj_name.find('<')
        pos_e = proj_name.find('>')

        try:
            proj_lon = float(proj_name[pos_s + 1:pos_e])
        except ValueError:
            raise ValueError(f'Could not find projection longitude from the projection name: {proj_name}')

        return proj_lon

    def get_area_def(self, dsid):
        """Get the area definition of the datasets in the file.

        Only applicable for MSG products!
        """
        pdict = {}
        pdict['coff'] = self.nc.attrs['COFF']
        pdict['loff'] = -self.nc.attrs['LOFF'] + 1
        pdict['cfac'] = self.nc.attrs['CFAC']
        pdict['lfac'] = self.nc.attrs['LFAC']

        # Unfortunately this dataset does not store a, b or h.
        # We assume a, rf and h here based on EUMETSAT standard values.
        pdict['a'] = 6378169
        pdict['b'] = 6356583.8
        pdict['h'] = 35785831.0

        pdict['ssp_lon'] = self._get_proj_lon()
        pdict['nlines'] = float(self.nc.attrs['NL'])
        pdict['ncols'] = float(self.nc.attrs['NC'])

        pdict['scandir'] = 'S2N'
        pdict['a_desc'] = 'MSG/SEVIRI low resolution channel area'
        pdict['a_name'] = 'geosmsg'
        pdict['p_id'] = 'msg_lowres'

        area_extent = get_area_extent(pdict)
        area_extent = (area_extent[0],
                       area_extent[1],
                       area_extent[2],
                       area_extent[3])

        area = get_area_definition(pdict, area_extent)

        return area

    @staticmethod
    def _ensure_crs_extents_in_meters(crs, area_extent):
        """Fix units in Earth shape, satellite altitude and 'units' attribute."""
        if 'kilo' in crs.axis_info[0].unit_name:
            proj_dict = crs.to_dict()
            proj_dict["units"] = "m"
            if "a" in proj_dict:
                proj_dict["a"] *= 1000.
            if "b" in proj_dict:
                proj_dict["b"] *= 1000.
            if "R" in proj_dict:
                proj_dict["R"] *= 1000.
            proj_dict["h"] *= 1000.
            area_extent = tuple([val * 1000. for val in area_extent])
            crs = CRS.from_dict(proj_dict)
        return crs, area_extent

    @property
    def start_time(self):
        """Return the start time of the object."""
        return datetime.strptime(self.nc.attrs['SENSING_START_TIME'], '%Y%m%d%H%M%S')

    @property
    def end_time(self):
        """Return the end time of the object.

        LSA SAF data only includes the nominal start time, so we have to calculate the end time.
        """
        from datetime import timedelta
        tmptime = datetime.strptime(self.nc.attrs['IMAGE_ACQUISITION_TIME'], '%Y%m%d%H%M%S')
        delta = self.nc.attrs['TIME_RANGE']
        if delta == '15-min':
            tdelt = timedelta(minutes=15)
        elif delta == '5-min':
            tdelt = timedelta(minutes=5)
        else:
            raise ValueError('Unknown scanning time range: {}'.format(delta))
        return tmptime + tdelt

    @property
    def sensor_names(self):
        """List of sensors represented in this file."""
        return [self.sensor]


def remove_empties(variable):
    """Remove empty objects from the *variable*'s attrs."""
    import h5py
    for key, val in variable.attrs.items():
        if isinstance(val, h5py._hl.base.Empty):
            variable.attrs.pop(key)

    return variable
