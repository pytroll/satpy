#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Martin Raspaud

# Author(s):

#   Guido della Bruna <Guido.DellaBruna@meteoswiss.ch>
#   Marco Sassi       <Marco.Sassi@meteoswiss.ch>

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
from satpy.projectable import Projectable
from satpy.readers import DatasetID
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)

PLATFORM_NAMES = {'ABI': 'GOES-R'}


class NC_ABI_L1B(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(NC_ABI_L1B, self).__init__(filename, filename_info,
                                         filetype_info)
        self.nc = h5netcdf.File(filename, 'r')
        self.channel = filename_info['dataset_name']

#        cal_file = os.path.join(os.path.dirname(
#            filename), 'instrument_data.nc')
#        self.cal = h5netcdf.File(cal_file, 'r')
        self.platform_name = PLATFORM_NAMES[filename_info['mission_id']]
        self.sensor = 'abi'

    def get_dataset(self, key, info):
        """Load a dataset
        """
        logger.debug('Reading in get_dataset%s.', key.name)

        variable = self.nc["Rad"]

        radiances = (np.ma.masked_equal(variable[:],
                                        variable.attrs['_FillValue'], copy=False) *
                     variable.attrs['scale_factor'] +
                     variable.attrs['add_offset'])
        units = variable.attrs['units']

        self.calibrate(radiances, key)

        proj = Projectable(radiances,
                           copy=False,
                           units=units,
                           platform_name=self.platform_name,
                           sensor=self.sensor)
        self.nlines, self.ncols = radiances.shape

        return proj

    # def get_lonlats(self, navid, nav_info, lon_out, lat_out):
    #     """Load an area.
    #     """
    #     lon_out[:] = self.get_dataset(DatasetID('longitude'))
    #     lat_out[:] = self.get_dataset(DatasetID('latitude'))

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

    def get_area_def(self, key, info):
        # TODO Projection information are hard coded for 0 degree geos projection
        # Test dataset doen't provide the values in the file container.
        # Only fill values are inserted
        #cfac = np.uint32(self.proj_info['CFAC'])
        #lfac = np.uint32(self.proj_info['LFAC'])
        #coff = np.float32(self.proj_info['COFF'])
        #loff = np.float32(self.proj_info['LOFF'])
        a = self.nc["goes_imager_projection"].attrs['semi_major_axis'][...]
        h = self.nc["goes_imager_projection"].attrs[
            'perspective_point_height'][...]
        b = self.nc["goes_imager_projection"].attrs['semi_minor_axis'][...]
        lon_0 = self.nc["goes_imager_projection"].attrs[
            'longitude_of_projection_origin'][...]
        #nlines = self.nc['/state/processor/reference_grid_number_of_columns']
        #ncols = self.nc['/state/processor/reference_grid_number_of_rows']
        #nlines = 5568
        #ncols = 5568
        # Channel dependent swath resoultion

        scale_x = self.nc['x'].attrs["scale_factor"]
        scale_y = self.nc['y'].attrs["scale_factor"]
        offset_x = self.nc['x'].attrs["add_offset"]
        offset_y = self.nc['x'].attrs["add_offset"]

        # x and y extents in m
        x_ext = abs(h * scale_x * (self.nc['x'][0] - self.nc['x'][-1]))[0]
        y_ext = abs(h * scale_y * (self.nc['y'][0] - self.nc['y'][-1]))[0]

        area_extent = (-x_ext / 2, -y_ext / 2, x_ext / 2, y_ext / 2)

        proj_dict = {'a': float(a),
                     'b': float(b),
                     'lon_0': float(lon_0),
                     'h': float(h),
                     'proj': 'geos',
                     'units': 'm'}

        area = geometry.AreaDefinition(
            'some_area_name',
            "On-the-fly area",
            'geosabii',
            proj_dict,
            self.ncols,
            self.nlines,
            area_extent)

        self.area = area

        return area

    def _vis_calibrate(self, data, key):

        esun = self.nc['esun'][...]
        rad = self.nc["Rad"]
        sf = self.nc["Rad"].attrs["scale_factor"]
        of = self.nc["Rad"].attrs["add_offset"]
        lv = self.nc["Rad"] * sf + of
        d = self.nc["earth_sun_distance_anomaly_in_AU"]

        rf = lv * np.pi * d * d / esun

        data.data[:] = rf

    def _ir_calibrate(self, data, key):

        fk1 = self.nc["planck_fk1"][...]
        fk2 = self.nc["planck_fk2"][...]
        sf = nc["Rad"].attrs["scale_factor"]
        of = nc["Rad"].attrs["add_offset"]
        lv = nc["Rad"] * sf + of
        bc1 = self.nc["planck_bc1"][...]
        bc2 = self.nc["planck_bc2"][...]

        bt = (fk2 / (log((fk1 / lv) + 1)) - bc1) / bc2

        data.data[:] = bt

    def calibrate(self, data, key):
        logger.debug("CALIBRATE")

        ch = self.nc["band_id"][...]
        if ch < 7:
            self._vis_calibrate(data, key)
        else:
            self._ir_calibrate(data, key)

    @property
    def start_time(self):
        return datetime.strptime(self.nc.attrs['time_coverage_start'], '%Y-%m-%dT%H:%M:%S.%fZ')

    @property
    def end_time(self):
        return datetime.strptime(self.nc.attrs['time_coverage_end'], '%Y-%m-%dT%H:%M:%S.%fZ')
