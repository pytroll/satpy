#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016.

# Author(s):

#
#   Thomas Leppelt <thomas.leppelt@gmail.com>
#   Sauli Joro <sauli.joro@icloud.com>

# This file is part of satpy.

# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.

"""Interface to MTG-FCI Retrieval NetCDF files

"""
import os.path
from datetime import datetime, timedelta
import numpy as np
from pyresample import geometry
import h5netcdf
import logging
from collections import defaultdict

from satpy.projectable import Projectable
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)


class FCIFDHSIFileHandler(BaseFileHandler):

    """MTG FCI FDHSI File Reader
    """

    def __init__(self, filename, filename_info, filetype_info):
        super(FCIFDHSIFileHandler, self).__init__(filename, filename_info,
                                                  filetype_info)
        logger.debug("READING: %s" % filename)
        logger.debug("START: %s" % self.start_time)
        logger.debug("START: %s" % self.end_time)

        self.nc = h5netcdf.File(filename, 'r')
        self.filename = filename
        self.cache = {}

    @property
    def start_time(self):
        return self.filename_info['start_time']

    @property
    def end_time(self):
        return self.filename_info['end_time']

    def get_dataset(self, key, info=None):
        """Load a dataset
        """
        if key in self.cache:
            return self.cache[key]

        logger.debug('Reading %s.', key.name)
        measured = self.nc['/data/%s/measured' % key.name]
        variable = self.nc['/data/%s/measured/effective_radiance' % key.name]

        # Get start/end line and column of loaded swath.

        self.startline = int(measured.variables['start_position_row'][...])
        self.endline = int(measured.variables['end_position_row'][...])
        self.startcol = int(measured.variables['start_position_column'][...])
        self.endcol = int(measured.variables['end_position_column'][...])

        ds = (np.ma.masked_equal(variable[:],
                                 variable.attrs['_FillValue']) *
              (variable.attrs['scale_factor'] * 1.0) +
              variable.attrs.get('add_offset', 0))

        self.calibrate(ds, key)

        out = Projectable(ds, dtype=np.float32)

        self.cache[key] = out
        self.nlines, self.ncols = ds.shape

        return out

    def calc_area_extent(self, key):
        """Calculate area extent for a dataset.
        """
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
        """Get area definition for a dataset.
        """
        # TODO Projection information are hard coded for 0 degree geos projection
        # Test dataset doen't provide the values in the file container.
        # Only fill values are inserted
        #cfac = np.uint32(self.proj_info['CFAC'])
        #lfac = np.uint32(self.proj_info['LFAC'])
        #coff = np.float32(self.proj_info['COFF'])
        #loff = np.float32(self.proj_info['LOFF'])
        #a = self.nc['/state/processor/earth_equatorial_radius']
        a = 6378169.
        #h = self.nc['/state/processor/reference_altitude'] * 1000 - a
        h = 35785831.
        #b = self.nc['/state/processor/earth_polar_radius']
        b = 6356583.8
        #lon_0 = self.nc['/state/processor/projection_origin_longitude']
        lon_0 = 0.
        #nlines = self.nc['/state/processor/reference_grid_number_of_columns']
        #ncols = self.nc['/state/processor/reference_grid_number_of_rows']
        #nlines = 5568
        #ncols = 5568
        # Channel dependent swath resoultion
        area_extent = self.calc_area_extent(key)
        logger.debug("Calculated area extent: %s" % ''.join(str(area_extent)))

        #c, l = 0, (1 + self.total_segments - self.segment_number) * nlines
        #ll_x, ll_y = (c - coff) / cfac * 2**16, (l - loff) / lfac * 2**16
        # c, l = ncols, (1 + self.total_segments -
        #                self.segment_number) * nlines - nlines
        #ur_x, ur_y = (c - coff) / cfac * 2**16, (l - loff) / lfac * 2**16

        # area_extent = (np.deg2rad(ll_x) * h, np.deg2rad(ur_y) * h,
        #               np.deg2rad(ur_x) * h, np.deg2rad(ll_y) * h)
        # area_extent = (-5432229.9317116784, -5429229.5285458621,
        #                5429229.5285458621, 5432229.9317116784)

        proj_dict = {'a': float(a),
                     'b': float(b),
                     'lon_0': float(lon_0),
                     'h': float(h),
                     'proj': 'geos',
                     'units': 'm'}

        area = geometry.AreaDefinition(
            'some_area_name',
            "On-the-fly area",
            'geosfci',
            proj_dict,
            self.ncols,
            self.nlines,
            area_extent)

        self.area = area

        return area

    def calibrate(self, data, key):
        """Calibratie data
        """

        logger.debug('Calibration: %s' % key.calibration)
        if key.calibration == 'brightness_temperature':
            logger.warning("IR channel calibration disabled!!")
#            self._ir_calibrate(data, key)
        elif key.calibration == 'reflectance':
            logger.warning("VIS channel calibration disabled!!")
#            self._vis_calibrate(data, key)
        else:
            pass

    def _ir_calibrate(self, data, key):
        """IR channel calibration.
        """
        # Not sure if Lv is correct, FCI User Guide is a bit unclear

        Lv = data.data * \
            self.nc[
                '/data/%s/measured/radiance_unit_conversion_coefficient' % key.name][...]

        vc = self.nc['/data/%s/central_wavelength_actual' % key.name][...]
        a, b, dummy = self.nc[
            '/data/%s/measured/radiance_to_bt_conversion_coefficients' % key.name][...]
        c1, c2 = self.nc[
            '/data/%s/measured/radiance_to_bt_conversion_constants' % key.name][...]

        nom = c2 * vc
        denom = a * np.log(1 + (c1 * vc**3) / Lv)

        data.data[:] = nom / denom - b / a

    def _vis_calibrate(self, data, key):
        """VIS channel calibration
        """
        # radiance to reflectance taken as in mipp/xrit/MSG.py
        # again FCI User Guide is not clear on how to do this

        sirr = self.nc[
            '/data/%s/measured/channel_effective_solar_irradiance' % key.name][...]

        # reflectance = radiance / sirr * 100
        data.data[:] /= sirr
        data.data[:] *= 100
