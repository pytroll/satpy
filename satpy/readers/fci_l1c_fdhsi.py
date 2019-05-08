#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016.

# Author(s):

#
#   Thomas Leppelt <thomas.leppelt@gmail.com>
#   Sauli Joro <sauli.joro@icloud.com>
#   Gerrit Holl <gerrit.holl@dwd.de>

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

"""Interface to MTG-FCI-FDHSI L1C NetCDF files

This module defines the :class:`FCIFDHSIFileHandler` file handler, to
be used for reading Meteosat Third Generation (MTG) Flexible Combined
Imager (FCI) Full Disk High Spectral Imagery (FDHSI) data.  FCI will fly
on the MTG Imager (MTG-I) series of satellites, scheduled to be launched
in 2021 by the earliest.  For more information about FCI, see `EUMETSAT`_.

.. EUMETSAT: https://www.eumetsat.int/website/home/Satellites/FutureSatellites/MeteosatThirdGeneration/MTGDesign/index.html#fci
"""

import logging
import numpy as np
import xarray as xr

from pyresample import geometry
from netCDF4 import default_fillvals

from .file_handlers import BaseFileHandler
from .. import CHUNK_SIZE

logger = logging.getLogger(__name__)


class FCIFDHSIFileHandler(BaseFileHandler):
    """Class implementing the MTG FCI FDHSI File Reader

    This class implements the Meteosat Third Generation (MTG) Flexible
    Combined Imager (FCI) Full Disk High Spectral Imagery (FDHSI) reader.
    It is designed to be used through the :class:`~satpy.Scene`
    class using the :mod:`~satpy.Scene.load` method with the reader
    `"fci_l1c_fdhsi"`.

    """

    def __init__(self, filename, filename_info, filetype_info):
        super(FCIFDHSIFileHandler, self).__init__(filename, filename_info,
                                                  filetype_info)
        logger.debug('Reading: {}'.format(self.filename))
        logger.debug('Start: {}'.format(self.start_time))
        logger.debug('End: {}'.format(self.end_time))

        nc = xr.open_dataset(self.filename,
                             mask_and_scale=True,
                             decode_cf=True,
                             chunks=CHUNK_SIZE)
        processor = xr.open_dataset(self.filename,
                                    mask_and_scale=False,
                                    decode_cf=True,
                                    group='/state/processor',
                                    chunks=CHUNK_SIZE)
        self.nc = {'root': nc,
                   'processor': processor}
        self.cache = {}

    @property
    def start_time(self):
        return self.filename_info['start_time']

    @property
    def end_time(self):
        return self.filename_info['end_time']

    def get_dataset(self, key, info=None):
        """Load a dataset."""

        logger.debug('Reading {}'.format(key.name))
        # Get the dataset
        # Get metadata for given dataset
        measured, root = self.get_channel_dataset(key.name)
        radiances = measured['effective_radiance']
        radiances = radiances.where(radiances > radiances.attrs['valid_range'][0])
        radiances = radiances.where(radiances < radiances.attrs['valid_range'][1])
        radiances = radiances * radiances.attrs['scale_factor'] + radiances.attrs['add_offset']

        res = self.calibrate(radiances, key, measured, root)

        self.nlines, self.ncols = res.shape
        res.attrs.update(key.to_dict())
        return res

    def get_channel_dataset(self, channel):
        if channel not in self.nc:
            root_group = '/data/{}'.format(channel)
            group = '/data/{}/measured'.format(channel)
            measured = xr.open_dataset(self.filename,
                                       mask_and_scale=False,
                                       decode_cf=True,
                                       group=group,
                                       chunks=CHUNK_SIZE)
            root = xr.open_dataset(self.filename,
                                   mask_and_scale=False,
                                   decode_cf=True,
                                   group=root_group,
                                   chunks=CHUNK_SIZE)
            self.nc[channel] = measured, root
        return self.nc[channel]

    def calc_area_extent(self, key):
        """Calculate area extent for a dataset."""
        # Calculate the area extent of the swath based on start line and column
        # information, total number of segments and channel resolution
        # numbers from Package Description, Table 8
        xyres = {500: 22272, 1000: 11136, 2000: 5568}
        chkres = xyres[key.resolution]

        # Get metadata for given dataset
        measured, root = self.get_channel_dataset(key.name)
        variable = measured['effective_radiance']
        # Get start/end line and column of loaded swath.
        self.startline = int(measured['start_position_row'])
        self.endline = int(measured['end_position_row'])
        self.startcol = int(measured['start_position_column'])
        self.endcol = int(measured['end_position_column'])
        self.nlines, self.ncols = variable.shape

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

    _fallback_area_def = {
            "reference_altitude": 35786400, # metre
            }
    def get_area_def(self, key, info=None):
        """Calculate on-fly area definition for 0 degree geos-projection for a dataset."""
        # TODO Projection information are hard coded for 0 degree geos projection
        # Test dataset doen't provide the values in the file container.
        # Only fill values are inserted

        a = float(self.nc['processor']['earth_equatorial_radius'])
        b = float(self.nc['processor']['earth_polar_radius'])
        h = float(self.nc['processor']['reference_altitude'])
        lon_0 = float(self.nc['processor']['projection_origin_longitude'])
        if h == default_fillvals[
                self.nc["processor"]["reference_altitude"].dtype.str[1:]]:
            logger.warn(
                    "Reference altitude in {:s} set to "
                    "fill value, using {:d}".format(
                        self.filename,
                        self._fallback_area_def["reference_altitude"]))
            h = self._fallback_area_def["reference_altitude"]
        # Channel dependent swath resoultion
        area_extent = self.calc_area_extent(key)
        logger.debug('Calculated area extent: {}'
                     .format(''.join(str(area_extent))))

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

    def calibrate(self, data, key, measured, root):
        """Data calibration."""

        # logger.debug('Calibration: %s' % key.calibration)
        if key.calibration == 'brightness_temperature':
            self._ir_calibrate(data, measured, root)
            pass
        elif key.calibration == 'reflectance':
            self._vis_calibrate(data, measured)
        else:
            logger.warning('Calibration disabled!')

        return data

    def _ir_calibrate(self, radiance, measured, root):
        """IR channel calibration."""
        # Not sure if Lv is correct, FCI User Guide is a bit unclear

        Lv = radiance * measured['radiance_unit_conversion_coefficient']
        vc = root['central_wavelength_actual']
        a, b, c, d = measured['radiance_to_bt_conversion_coefficients']

        c1, c2 = measured['radiance_to_bt_conversion_constants']

        nom = c2 * vc
        denom = a * np.log(1 + (c1 * vc**3) / Lv)

        return nom / denom - b / a

    def _vis_calibrate(self, radiance, measured):
        """VIS channel calibration."""
        # radiance to reflectance taken as in mipp/xrit/MSG.py
        # again FCI User Guide is not clear on how to do this

        sirr = float(measured['channel_effective_solar_irradiance'])
        return radiance / sirr * 100
