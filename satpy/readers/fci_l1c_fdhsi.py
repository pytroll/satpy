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
import numpy as np
from pyresample import geometry
import h5py
import xarray as xr
import logging

from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)


class FCIFDHSIFileHandler(BaseFileHandler):

    """MTG FCI FDHSI File Reader
    """

    def __init__(self, filename, filename_info, filetype_info):
        super(FCIFDHSIFileHandler, self).__init__(filename, filename_info,
                                                  filetype_info)
        logger.debug('Reading: {}'.format(self.filename))
        logger.debug('Start: {}'.format(self.start_time))
        logger.debug('End: {}'.format(self.end_time))

        self.nc = h5py.File(self.filename, 'r')
        self.cache = {}

    @property
    def start_time(self):
        return self.filename_info['start_time']

    @property
    def end_time(self):
        return self.filename_info['end_time']

    def get_dataset(self, key, info=None):
        """Load a dataset."""
        if key in self.cache:
            return self.cache[key]

        logger.debug('Reading {}'.format(key.name))
        # Get the dataset
        # Get metadata for given dataset
        variable = self.nc['/data/{}/measured/effective_radiance'
                           .format(key.name)]
        # Convert to xarray
        radiances = xr.DataArray(np.asarray(variable, np.float32), dims=['y', 'x'])
        radiances.attrs['scale_factor'] = variable.attrs['scale_factor']
        radiances.attrs['offset'] = variable.attrs.get('add_offset', 0)
        radiances.attrs['FillValue'] = variable.attrs['_FillValue']
        # Set invalid values to NaN
        radiances.values[radiances == radiances.attrs['FillValue']] = np.nan
        # Apply scale factor and offset
        radiances = radiances * (radiances.attrs['scale_factor'] * 1.0) + radiances.attrs['offset']

        # TODO: Calibration is disabled, waiting for calibration parameters from EUMETSAT
        res = self.calibrate(radiances, key)

        self.cache[key] = res
        self.nlines, self.ncols = res.shape

        return res

    def calc_area_extent(self, key):
        """Calculate area extent for a dataset."""
        # Calculate the area extent of the swath based on start line and column
        # information, total number of segments and channel resolution
        xyres = {500: 22272, 1000: 11136, 2000: 5568}
        chkres = xyres[key.resolution]

        # Get metadata for given dataset
        measured = self.nc['/data/{}/measured'.format(key.name)]
        variable = self.nc['/data/{}/measured/effective_radiance'
                           .format(key.name)]
        # Get start/end line and column of loaded swath.
        self.startline = int(measured['start_position_row'][...])
        self.endline = int(measured['end_position_row'][...])
        self.startcol = int(measured['start_position_column'][...])
        self.endcol = int(measured['end_position_column'][...])
        self.nlines, self.ncols = variable[:].shape

        logger.debug('Channel {} resolution: {}'.format(key.name, chkres))
        logger.debug('Row/Cols: {} / {}'.format(self.nlines, self.ncols))
        logger.debug('Start/End row: {} / {}'.format(self.startline, self.endline))
        logger.debug('Start/End col: {} / {}'.format(self.startcol, self.endcol))
        # total_segments = 70

        # Calculate full globe line extent
        max_y = 5432229.9317116784
        min_y = -5429229.5285458621
        full_y = max_y + abs(min_y)
        # Single swath line extent
        res_y = full_y / chkres  # Extent per pixel resolution
        startl = min_y + res_y * self.startline - 0.5 * (res_y)
        endl = min_y + res_y * self.endline + 0.5 * (res_y)
        logger.debug('Start / end extent: {} / {}'.format(startl, endl))

        chk_extent = (-5432229.9317116784, endl,
                      5429229.5285458621, startl)
        return(chk_extent)

    def get_area_def(self, key, info=None):
        """Calculate on-fly area definition for 0 degree geos-projection for a dataset."""
        # TODO Projection information are hard coded for 0 degree geos projection
        # Test dataset doen't provide the values in the file container.
        # Only fill values are inserted
        # cfac = np.uint32(self.proj_info['CFAC'])
        # lfac = np.uint32(self.proj_info['LFAC'])
        # coff = np.float32(self.proj_info['COFF'])
        # loff = np.float32(self.proj_info['LOFF'])
        # a = self.nc['/state/processor/earth_equatorial_radius']
        a = 6378169.
        # h = self.nc['/state/processor/reference_altitude'] * 1000 - a
        h = 35785831.
        # b = self.nc['/state/processor/earth_polar_radius']
        b = 6356583.8
        # lon_0 = self.nc['/state/processor/projection_origin_longitude']
        lon_0 = 0.
        # nlines = self.nc['/state/processor/reference_grid_number_of_columns']
        # ncols = self.nc['/state/processor/reference_grid_number_of_rows']
        # nlines = 5568
        # ncols = 5568
        # Channel dependent swath resoultion
        area_extent = self.calc_area_extent(key)
        logger.debug('Calculated area extent: {}'
                     .format(''.join(str(area_extent))))

        # c, l = 0, (1 + self.total_segments - self.segment_number) * nlines
        # ll_x, ll_y = (c - coff) / cfac * 2**16, (l - loff) / lfac * 2**16
        #  c, l = ncols, (1 + self.total_segments -
        #                 self.segment_number) * nlines - nlines
        # ur_x, ur_y = (c - coff) / cfac * 2**16, (l - loff) / lfac * 2**16

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
        """Data calibration."""

        # logger.debug('Calibration: %s' % key.calibration)
        logger.warning('Calibration disabled!')
        if key.calibration == 'brightness_temperature':
            # self._ir_calibrate(data, key)
            pass
        elif key.calibration == 'reflectance':
            # self._vis_calibrate(data, key)
            pass
        else:
            pass

        return data

    def _ir_calibrate(self, data, key):
        """IR channel calibration."""
        # Not sure if Lv is correct, FCI User Guide is a bit unclear

        Lv = data.data * \
            self.nc[
                '/data/{}/measured/radiance_unit_conversion_coefficient'
                .format(key.name)][...]

        vc = self.nc['/data/{}/central_wavelength_actual'
                     .format(key.name)][...]
        a, b, dummy = self.nc[
            '/data/{}/measured/radiance_to_bt_conversion_coefficients'
            .format(key.name)][...]
        c1, c2 = self.nc[
            '/data/{}/measured/radiance_to_bt_conversion_constants'
            .format(key.name)][...]

        nom = c2 * vc
        denom = a * np.log(1 + (c1 * vc**3) / Lv)

        data.data[:] = nom / denom - b / a

    def _vis_calibrate(self, data, key):
        """VIS channel calibration."""
        # radiance to reflectance taken as in mipp/xrit/MSG.py
        # again FCI User Guide is not clear on how to do this

        sirr = self.nc[
            '/data/{}/measured/channel_effective_solar_irradiance'
            .format(key.name)][...]

        # reflectance = radiance / sirr * 100
        data.data[:] /= sirr
        data.data[:] *= 100
