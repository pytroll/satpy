#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Satpy developers
#
# satpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# satpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with satpy.  If not, see <http://www.gnu.org/licenses/>.
"""EUMETSAT EPS-SG Visible/Infrared Imager (VII) Level 1B products reader.

The ``vii_l1b_nc`` reader reads and calibrates EPS-SG VII L1b image data in netCDF format. The format is explained
in the `EPS-SG VII Level 1B Product Format Specification`_.
References:
.. _EPS-SG VII Level 1B Product Format Specification: https://www.eumetsat.int/website/wcm/idc/idcplg?IdcService
   =GET_FILE&dDocName=PDF_EPSSG_VII_L1B_PFS&RevisionSelectionMethod=LatestReleased&Rendition=Web

"""

import logging
import numpy as np

from satpy.readers.vii_base_nc import ViiNCBaseFileHandler
from satpy.readers.vii_utils import C1, C2, MEAN_EARTH_RADIUS

logger = logging.getLogger(__name__)


class ViiL1bNCFileHandler(ViiNCBaseFileHandler):
    """Reader class for VII L1B products in netCDF format."""

    def __init__(self, filename, filename_info, filetype_info, **kwargs):
        """Read the calibration data and prepare the class for dataset reading."""
        super().__init__(filename, filename_info, filetype_info, **kwargs)

        # Read the variables which are required for the calibration
        self._bt_conversion_a = self['data/calibration_data/bt_conversion_a'].values
        self._bt_conversion_b = self['data/calibration_data/bt_conversion_b'].values
        self._channel_cw_thermal = self['data/calibration_data/channel_cw_thermal'].values
        self._integrated_solar_irradiance = self['data/calibration_data/integrated_solar_irradiance'].values
        # Computes the angle factor for reflectance calibration as inverse of cosine of solar zenith angle
        # (the values in the product file are on tie points and in degrees,
        # therefore interpolation and conversion to radians are required)
        solar_zenith_angle = self['data/measurement_data/solar_zenith']
        solar_zenith_angle_on_pixels = self._perform_interpolation(solar_zenith_angle)
        solar_zenith_angle_on_pixels_radians = np.radians(solar_zenith_angle_on_pixels)
        self.angle_factor = 1.0 / (np.cos(solar_zenith_angle_on_pixels_radians))

    def _perform_calibration(self, variable, dataset_info):
        """Perform the calibration.

        Args:
            variable: xarray DataArray containing the dataset to calibrate.
            dataset_info: dictionary of information about the dataset.

        Returns:
            DataArray: array containing the calibrated values and all the original metadata.

        """
        calibration_name = dataset_info['calibration']
        if calibration_name == 'brightness_temperature':
            # Extract the values of calibration coefficients for the current channel
            chan_index = dataset_info['chan_thermal_index']
            cw = self._channel_cw_thermal[chan_index]
            a = self._bt_conversion_a[chan_index]
            b = self._bt_conversion_b[chan_index]
            # Perform the calibration
            calibrated_variable = self._calibrate_bt(variable, cw, a, b)
            calibrated_variable.attrs = variable.attrs
        elif calibration_name == 'reflectance':
            # Extract the values of calibration coefficients for the current channel
            chan_index = dataset_info['chan_solar_index']
            isi = self._integrated_solar_irradiance[chan_index]
            # Perform the calibration
            calibrated_variable = self._calibrate_refl(variable, self.angle_factor, isi)
            calibrated_variable.attrs = variable.attrs
        elif calibration_name == 'radiance':
            calibrated_variable = variable
        else:
            raise ValueError("Unknown calibration %s for dataset %s" % (calibration_name, dataset_info['name']))

        return calibrated_variable

    def _perform_orthorectification(self, variable, orthorect_data_name):
        """Perform the orthorectification.

        Args:
            variable: xarray DataArray containing the dataset to correct for orthorectification.
            orthorect_data_name: name of the orthorectification correction data in the product.

        Returns:
            DataArray: array containing the corrected values and all the original metadata.

        """
        try:
            orthorect_data = self[orthorect_data_name]
            # Convert the orthorectification delta values from meters to degrees
            # based on the simplified formula using mean Earth radius
            variable += np.degrees(orthorect_data / MEAN_EARTH_RADIUS)
        except KeyError:
            logger.warning('Required dataset %s for orthorectification not available, skipping', orthorect_data_name)
        return variable

    @staticmethod
    def _calibrate_bt(radiance, cw, a, b):
        """Perform the calibration to brightness temperature.

        Args:
            radiance: numpy ndarray containing the radiance values.
            cw: center wavelength [μm].
            a: temperature coefficient [-].
            b: temperature coefficient [K].

        Returns:
            numpy ndarray: array containing the calibrated brightness temperature values.

        """
        log_expr = np.log(1.0 + C1 / ((cw ** 5) * radiance))
        bt_values = b + (a * C2 / (cw * log_expr))
        return bt_values

    @staticmethod
    def _calibrate_refl(radiance, angle_factor, isi):
        """Perform the calibration to reflectance.

        Args:
            radiance: numpy ndarray containing the radiance values.
            angle_factor: numpy ndarray containing the inverse of cosine of solar zenith angle [-].
            isi: integrated solar irradiance [W/(m2 * μm)].

        Returns:
            numpy ndarray: array containing the calibrated reflectance values.

        """
        refl_values = (np.pi / isi) * angle_factor * radiance
        return refl_values
