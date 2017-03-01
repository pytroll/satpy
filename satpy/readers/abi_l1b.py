#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Martin Raspaud

# Author(s):

#   Guido della Bruna <Guido.DellaBruna@meteoswiss.ch>
#   Marco Sassi       <Marco.Sassi@meteoswiss.ch>
#   Martin Raspaud    <Martin.Raspaud@smhi.se>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Advance Baseline Imager reader
"""

import logging
import os
from datetime import datetime

import h5netcdf
import numpy as np

from pyresample import geometry
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)

PLATFORM_NAMES = {'ABI': 'GOES-R'}


class NC_ABI_L1B(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(NC_ABI_L1B, self).__init__(filename, filename_info,
                                         filetype_info)
        self.nc = h5netcdf.File(filename, 'r')

#        cal_file = os.path.join(os.path.dirname(
#            filename), 'instrument_data.nc')
#        self.cal = h5netcdf.File(cal_file, 'r')
        self.platform_name = PLATFORM_NAMES[filename_info['mission_id']]
        self.sensor = 'abi'
        self.nlines, self.ncols = self.nc["Rad"].shape

    def get_shape(self, key, info):
        """Get the shape of the data."""
        return self.nlines, self.ncols

    def get_dataset(self, key, info, out=None,
                    xslice=slice(None), yslice=slice(None)):
        """Load a dataset."""
        logger.debug('Reading in get_dataset %s.', key.name)

        variable = self.nc["Rad"]

        radiances = (np.ma.masked_equal(variable[yslice, xslice],
                                        variable.attrs['_FillValue'], copy=False) *
                     variable.attrs['scale_factor'] +
                     variable.attrs['add_offset'])
        # units = variable.attrs['units']
        units = self.calibrate(radiances, key)

        # convert to satpy standard units
        if units == '1':
            radiances[:] *= 100.
            units = '%'

        out.data[:] = radiances
        out.mask[:] = np.ma.getmask(radiances)
        out.info.update({'units': units,
                         'platform_name': self.platform_name,
                         'sensor': self.sensor})
        out.info.update(key.to_dict())

        return out

    def calc_area_extent(self, key):
        # Calculate the area extent of the swath based on start line and column
        # information, total number of segments and channel resolution
        xyres = {500: 22272, 1000: 11136, 2000: 5568}
        chkres = xyres[key.resolution]
        logger.debug(chkres)
        logger.debug("ROW/COLS: %d / %d" % (self.nlines, self.ncols))
        logger.debug("START/END ROW: %d / %d" % (self.startline, self.endline))
        logger.debug("START/END COL: %d / %d" % (self.startcol, self.endcol))
        total_segments = 70

        # Calculate full globe line extent
        max_y = 5432229.9317116784
        min_y = -5429229.5285458621
        full_y = max_y + abs(min_y)
        # Single swath line extent
        res_y = full_y / chkres  # Extent per pixel resolution
        startl = min_y + res_y * self.startline - 0.5 * (res_y)
        endl = min_y + res_y * self.endline + 0.5 * (res_y)
        logger.debug("START / END EXTENT: %d / %d" % (startl, endl))

        chk_extent = (-5432229.9317116784, endl,
                      5429229.5285458621, startl)
        return(chk_extent)

    def get_area_def(self, key):
        a = self.nc["goes_imager_projection"].attrs['semi_major_axis'][...]
        h = self.nc["goes_imager_projection"].attrs[
            'perspective_point_height'][...]
        b = self.nc["goes_imager_projection"].attrs['semi_minor_axis'][...]
        lon_0 = self.nc["goes_imager_projection"].attrs[
            'longitude_of_projection_origin'][...]
        sweep_axis = self.nc["goes_imager_projection"].attrs[
            'sweep_angle_axis'][0]

        scale_x = self.nc['x'].attrs["scale_factor"][0]
        scale_y = self.nc['y'].attrs["scale_factor"][0]
        offset_x = self.nc['x'].attrs["add_offset"][0]
        offset_y = self.nc['y'].attrs["add_offset"][0]

        # x and y extents in m
        h = float(h)
        x_l = h * (self.nc['x'][0] * scale_x + offset_x)
        x_r = h * (self.nc['x'][-1] * scale_x + offset_x)
        y_l = h * (self.nc['y'][-1] * scale_y + offset_y)
        y_u = h * (self.nc['y'][0] * scale_y + offset_y)
        x_half = (x_r - x_l) / (self.ncols - 1) / 2.
        y_half = (y_u - y_l) / (self.nlines - 1) / 2.
        area_extent = (x_l - x_half, y_l - y_half, x_r + x_half, y_u + y_half)

        proj_dict = {'a': float(a),
                     'b': float(b),
                     'lon_0': float(lon_0),
                     'h': h,
                     'proj': 'geos',
                     'units': 'm',
                     'sweep': sweep_axis}

        area = geometry.AreaDefinition(
            'some_area_name',
            "On-the-fly area",
            'geosabii',
            proj_dict,
            self.ncols,
            self.nlines,
            area_extent)

        return area

    def _vis_calibrate(self, data, key):

        esun = self.nc['esun'][...]
        d = self.nc["earth_sun_distance_anomaly_in_AU"]

        rf = data * np.pi * d * d / esun

        data.data[:] = rf
        return '1'

    def _ir_calibrate(self, data, key):

        fk1 = self.nc["planck_fk1"][...]
        fk2 = self.nc["planck_fk2"][...]
        bc1 = self.nc["planck_bc1"][...]
        bc2 = self.nc["planck_bc2"][...]

        bt = (fk2 / (np.log((fk1 / data) + 1)) - bc1) / bc2

        data.data[:] = bt
        return 'K'

    def calibrate(self, data, key):
        logger.debug("CALIBRATE")

        ch = self.nc["band_id"][...]
        if ch < 7:
            return self._vis_calibrate(data, key)
        else:
            return self._ir_calibrate(data, key)

    @property
    def start_time(self):
        return datetime.strptime(self.nc.attrs['time_coverage_start'], '%Y-%m-%dT%H:%M:%S.%fZ')

    @property
    def end_time(self):
        return datetime.strptime(self.nc.attrs['time_coverage_end'], '%Y-%m-%dT%H:%M:%S.%fZ')
