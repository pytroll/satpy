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

.. _EUMETSAT: https://www.eumetsat.int/website/home/Satellites/FutureSatellites/MeteosatThirdGeneration/MTGDesign/index.html#fci  # noqa: E501
"""

import logging
import numpy as np

from pyresample import geometry
from netCDF4 import default_fillvals

from .netcdf_utils import NetCDF4FileHandler

logger = logging.getLogger(__name__)


class FCIFDHSIFileHandler(NetCDF4FileHandler):
    """Class implementing the MTG FCI FDHSI File Reader

    This class implements the Meteosat Third Generation (MTG) Flexible
    Combined Imager (FCI) Full Disk High Spectral Imagery (FDHSI) reader.
    It is designed to be used through the :class:`~satpy.Scene`
    class using the :mod:`~satpy.Scene.load` method with the reader
    ``"fci_l1c_fdhsi"``.

    """

    def __init__(self, filename, filename_info, filetype_info):
        super(FCIFDHSIFileHandler, self).__init__(filename, filename_info,
                                                  filetype_info)
        logger.debug('Reading: {}'.format(self.filename))
        logger.debug('Start: {}'.format(self.start_time))
        logger.debug('End: {}'.format(self.end_time))

        self.cache = {}

    @property
    def start_time(self):
        return self.filename_info['start_time']

    @property
    def end_time(self):
        return self.filename_info['end_time']

    def get_dataset(self, key, info=None):
        """Load a dataset."""

        logger.debug('Reading {} from {}'.format(key.name, self.filename))
        # Get the dataset
        # Get metadata for given dataset
        measured, root = self.get_channel_dataset(key.name)
        radlab = measured + "/effective_radiance"
        data = self[radlab]

        attrs = data.attrs.copy()
        info = info.copy()
        fv = attrs.pop(
                "FillValue",
                default_fillvals.get(data.dtype.str[1:], np.nan))
        vr = attrs.pop("valid_range", [-np.inf, np.inf])
        if key.calibration == "counts":
            attrs["_FillValue"] = fv
            nfv = fv
        else:
            nfv = np.nan
        data = data.where(data > vr[0], nfv)
        data = data.where(data < vr[1], nfv)
        if key.calibration == "counts":
            # from package description, this just means not applying add_offset
            # and scale_factor
            attrs.pop("scale_factor")
            attrs.pop("add_offset")
            data.attrs["units"] = "1"
            res = data
        else:
            data = (data * attrs.pop("scale_factor", 1) +
                    attrs.pop("add_offset", 0))

            if key.calibration in ("brightness_temperature", "reflectance"):
                res = self.calibrate(data, key, measured, root)
            else:
                res = data
                data.attrs["units"] = attrs["units"]
        # pre-calibration units no longer apply
        info.pop("units")
        attrs.pop("units")

        self.nlines, self.ncols = res.shape
        res.attrs.update({
                **key.to_dict(),
                **info,
                **attrs})
        return res

    def get_channel_dataset(self, channel):
        root_group = 'data/{}'.format(channel)
        group = 'data/{}/measured'.format(channel)

        return group, root_group

    def calc_area_extent(self, key):
        """Calculate area extent for a dataset."""
        # Calculate the area extent of the swath based on start line and column
        # information, total number of segments and channel resolution
        # numbers from Package Description, Table 8
        xyres = {500: 22272, 1000: 11136, 2000: 5568}
        chkres = xyres[key.resolution]

        # Get metadata for given dataset
        measured, root = self.get_channel_dataset(key.name)
        # Get start/end line and column of loaded swath.
        self.startline = int(self[measured + "/start_position_row"])
        self.endline = int(self[measured + "/end_position_row"])
        self.startcol = int(self[measured + "/start_position_column"])
        self.endcol = int(self[measured + "/end_position_column"])
        self.nlines, self.ncols = self[measured + "/effective_radiance/shape"]

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
            "reference_altitude": 35786400,  # metre
            }

    def get_area_def(self, key, info=None):
        """Calculate on-fly area definition for 0 degree geos-projection for a dataset."""
        # TODO Projection information are hard coded for 0 degree geos projection
        # Test dataset doen't provide the values in the file container.
        # Only fill values are inserted

        a = float(self["state/processor/earth_equatorial_radius"])
        b = float(self["state/processor/earth_polar_radius"])
        h = float(self["state/processor/reference_altitude"])
        lon_0 = float(self["state/processor/projection_origin_longitude"])
        if h == default_fillvals[
                self["state/processor/reference_altitude"].dtype.str[1:]]:
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

        if key.calibration == 'brightness_temperature':
            data = self._ir_calibrate(data, measured, root)
        elif key.calibration == 'reflectance':
            data = self._vis_calibrate(data, measured)
        else:
            raise RuntimeError("Reached unreachable code!")

        return data

    def _ir_calibrate(self, radiance, measured, root):
        """IR channel calibration."""

        Lv = radiance * self[measured + "/radiance_unit_conversion_coefficient"]
        vc = self[root + "/central_wavelength_actual"]

        a = self[measured + "/radiance_to_bt_conversion_coefficient_a"]
        b = self[measured + "/radiance_to_bt_conversion_coefficient_b"]

        c1 = self[measured + "/radiance_to_bt_conversion_constant_c1"]
        c2 = self[measured + "/radiance_to_bt_conversion_constant_c2"]

        nom = c2 * vc
        denom = a * np.log(1 + (c1 * vc**3) / Lv)

        res = nom / denom - b / a
        res.attrs["units"] = "K"
        return res

    def _vis_calibrate(self, radiance, measured):
        """VIS channel calibration."""
        # radiance to reflectance taken as in mipp/xrit/MSG.py
        # again FCI User Guide is not clear on how to do this

        sirr = float(self[measured + "/channel_effective_solar_irradiance"])
        res = radiance / sirr * 100
        res.attrs["units"] = "%"
        return res
