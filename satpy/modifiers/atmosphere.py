#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 Satpy developers
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
"""Modifiers related to atmospheric corrections or adjustments."""

import logging
from weakref import WeakValueDictionary

import numpy as np
import dask.array as da
import xarray as xr

from satpy.modifiers import ModifierBase
from satpy.utils import get_satpos

logger = logging.getLogger(__name__)


class PSPRayleighReflectance(ModifierBase):
    """Pyspectral-based rayleigh corrector for visible channels."""

    _rayleigh_cache = WeakValueDictionary()

    def get_angles(self, vis):
        """Get the sun and satellite angles from the current dataarray."""
        from pyorbital.astronomy import get_alt_az, sun_zenith_angle
        from pyorbital.orbital import get_observer_look

        lons, lats = vis.attrs['area'].get_lonlats(chunks=vis.data.chunks)
        lons = da.where(lons >= 1e30, np.nan, lons)
        lats = da.where(lats >= 1e30, np.nan, lats)
        sunalt, suna = get_alt_az(vis.attrs['start_time'], lons, lats)
        suna = np.rad2deg(suna)
        sunz = sun_zenith_angle(vis.attrs['start_time'], lons, lats)

        sat_lon, sat_lat, sat_alt = get_satpos(vis)
        sata, satel = get_observer_look(
            sat_lon,
            sat_lat,
            sat_alt / 1000.0,  # km
            vis.attrs['start_time'],
            lons, lats, 0)
        satz = 90 - satel
        return sata, satz, suna, sunz

    def __call__(self, projectables, optional_datasets=None, **info):
        """Get the corrected reflectance when removing Rayleigh scattering.

        Uses pyspectral.
        """
        from pyspectral.rayleigh import Rayleigh
        if not optional_datasets or len(optional_datasets) != 4:
            vis, red = self.match_data_arrays(projectables)
            sata, satz, suna, sunz = self.get_angles(vis)
            red.data = da.rechunk(red.data, vis.data.chunks)
        else:
            vis, red, sata, satz, suna, sunz = self.match_data_arrays(
                projectables + optional_datasets)
            sata, satz, suna, sunz = optional_datasets
            # get the dask array underneath
            sata = sata.data
            satz = satz.data
            suna = suna.data
            sunz = sunz.data

        # First make sure the two azimuth angles are in the range 0-360:
        sata = sata % 360.
        suna = suna % 360.
        ssadiff = da.absolute(suna - sata)
        ssadiff = da.minimum(ssadiff, 360 - ssadiff)
        del sata, suna

        atmosphere = self.attrs.get('atmosphere', 'us-standard')
        aerosol_type = self.attrs.get('aerosol_type', 'marine_clean_aerosol')
        rayleigh_key = (vis.attrs['platform_name'],
                        vis.attrs['sensor'], atmosphere, aerosol_type)
        logger.info("Removing Rayleigh scattering with atmosphere '%s' and "
                    "aerosol type '%s' for '%s'",
                    atmosphere, aerosol_type, vis.attrs['name'])
        if rayleigh_key not in self._rayleigh_cache:
            corrector = Rayleigh(vis.attrs['platform_name'], vis.attrs['sensor'],
                                 atmosphere=atmosphere,
                                 aerosol_type=aerosol_type)
            self._rayleigh_cache[rayleigh_key] = corrector
        else:
            corrector = self._rayleigh_cache[rayleigh_key]

        try:
            refl_cor_band = corrector.get_reflectance(sunz, satz, ssadiff,
                                                      vis.attrs['name'],
                                                      red.data)
        except (KeyError, IOError):
            logger.warning("Could not get the reflectance correction using band name: %s", vis.attrs['name'])
            logger.warning("Will try use the wavelength, however, this may be ambiguous!")
            refl_cor_band = corrector.get_reflectance(sunz, satz, ssadiff,
                                                      vis.attrs['wavelength'][1],
                                                      red.data)
        proj = vis - refl_cor_band
        proj.attrs = vis.attrs
        self.apply_modifier_info(vis, proj)
        return proj


class PSPAtmosphericalCorrection(ModifierBase):
    """Correct for atmospheric effects."""

    def __call__(self, projectables, optional_datasets=None, **info):
        """Get the atmospherical correction.

        Uses pyspectral.
        """
        from pyspectral.atm_correction_ir import AtmosphericalCorrection

        band = projectables[0]

        if optional_datasets:
            satz = optional_datasets[0]
        else:
            from pyorbital.orbital import get_observer_look
            lons, lats = band.attrs['area'].get_lonlats(chunks=band.data.chunks)
            sat_lon, sat_lat, sat_alt = get_satpos(band)
            try:
                dummy, satel = get_observer_look(sat_lon,
                                                 sat_lat,
                                                 sat_alt / 1000.0,  # km
                                                 band.attrs['start_time'],
                                                 lons, lats, 0)
            except KeyError:
                raise KeyError(
                    'Band info is missing some meta data!')
            satz = 90 - satel
            del satel

        logger.info('Correction for limb cooling')
        corrector = AtmosphericalCorrection(band.attrs['platform_name'],
                                            band.attrs['sensor'])

        atm_corr = corrector.get_correction(satz, band.attrs['name'], band)
        proj = xr.DataArray(atm_corr, attrs=band.attrs,
                            dims=band.dims, coords=band.coords)
        self.apply_modifier_info(band, proj)

        return proj


class CO2Corrector(ModifierBase):
    """CO2 correction of the brightness temperature of the MSG 3.9um channel.

    .. math::

      T4_CO2corr = (BT(IR3.9)^4 + Rcorr)^0.25
      Rcorr = BT(IR10.8)^4 - (BT(IR10.8)-dt_CO2)^4
      dt_CO2 = (BT(IR10.8)-BT(IR13.4))/4.0

    Derived from D. Rosenfeld, "CO2 Correction of Brightness Temperature of Channel IR3.9"
    References:
        - http://www.eumetrain.org/IntGuide/PowerPoints/Channels/conversion.ppt
    """

    def __call__(self, projectables, optional_datasets=None, **info):
        """Apply correction."""
        ir_039, ir_108, ir_134 = projectables
        logger.info('Applying CO2 correction')
        dt_co2 = (ir_108 - ir_134) / 4.0
        rcorr = ir_108 ** 4 - (ir_108 - dt_co2) ** 4
        t4_co2corr = (ir_039 ** 4 + rcorr).clip(0.0) ** 0.25

        t4_co2corr.attrs = ir_039.attrs.copy()

        self.apply_modifier_info(ir_039, t4_co2corr)

        return t4_co2corr
