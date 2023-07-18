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

import dask.array as da
import numpy as np
import xarray as xr

from satpy.modifiers import ModifierBase
from satpy.modifiers._crefl import ReflectanceCorrector  # noqa
from satpy.modifiers.angles import compute_relative_azimuth, get_angles, get_satellite_zenith_angle

logger = logging.getLogger(__name__)


class PSPRayleighReflectance(ModifierBase):
    """Pyspectral-based rayleigh corrector for visible channels.

    It is possible to use ``reduce_lim_low``, ``reduce_lim_high`` and ``reduce_strength``
    together to reduce rayleigh correction at high solar zenith angle and make the image
    transition from rayleigh-corrected to partially/none rayleigh-corrected at day/night edge,
    therefore producing a more natural look, which could be especially helpful for geostationary
    satellites. This reduction starts at solar zenith angle of ``reduce_lim_low``, and ends in
    ``reduce_lim_high``. It's linearly scaled between these two angles. The ``reduce_strength``
    controls the amount of the reduction. When the solar zenith angle reaches ``reduce_lim_high``,
    the rayleigh correction will remain ``(1 - reduce_strength)`` of its initial reduce_strength
    at ``reduce_lim_high``.

    To use this function in a YAML configuration file:

    .. code-block:: yaml

      rayleigh_corrected_reduced:
        modifier: !!python/name:satpy.modifiers.PSPRayleighReflectance
        atmosphere: us-standard
        aerosol_type: rayleigh_only
        reduce_lim_low: 70
        reduce_lim_high: 95
        reduce_strength: 0.6
        prerequisites:
          - name: B03
            modifiers: [sunz_corrected]
        optional_prerequisites:
          - satellite_azimuth_angle
          - satellite_zenith_angle
          - solar_azimuth_angle
          - solar_zenith_angle

    In the case above, rayleigh correction is reduced gradually starting at solar zenith angle 70°.
    When reaching 95°, the correction will only remain 40% its initial strength at 95°.

    """

    def __call__(self, projectables, optional_datasets=None, **info):
        """Get the corrected reflectance when removing Rayleigh scattering.

        Uses pyspectral.
        """
        from pyspectral.rayleigh import Rayleigh
        projectables = projectables + (optional_datasets or [])
        if len(projectables) != 6:
            vis, red = self.match_data_arrays(projectables)
            sata, satz, suna, sunz = get_angles(vis)
        else:
            vis, red, sata, satz, suna, sunz = self.match_data_arrays(projectables)
            # First make sure the two azimuth angles are in the range 0-360:
            sata = sata % 360.
            suna = suna % 360.

        # get the dask array underneath
        sata = sata.data
        satz = satz.data
        suna = suna.data
        sunz = sunz.data

        ssadiff = compute_relative_azimuth(sata, suna)
        del sata, suna

        atmosphere = self.attrs.get('atmosphere', 'us-standard')
        aerosol_type = self.attrs.get('aerosol_type', 'marine_clean_aerosol')
        reduce_lim_low = abs(self.attrs.get('reduce_lim_low', 70))
        reduce_lim_high = abs(self.attrs.get('reduce_lim_high', 105))
        reduce_strength = np.clip(self.attrs.get('reduce_strength', 0), 0, 1)

        logger.info("Removing Rayleigh scattering with atmosphere '%s' and "
                    "aerosol type '%s' for '%s'",
                    atmosphere, aerosol_type, vis.attrs['name'])
        corrector = Rayleigh(vis.attrs['platform_name'], vis.attrs['sensor'],
                             atmosphere=atmosphere,
                             aerosol_type=aerosol_type)

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

        if reduce_strength > 0:
            if reduce_lim_low > reduce_lim_high:
                reduce_lim_low = reduce_lim_high
            refl_cor_band = corrector.reduce_rayleigh_highzenith(sunz, refl_cor_band,
                                                                 reduce_lim_low, reduce_lim_high, reduce_strength)

        proj = vis - refl_cor_band
        proj.attrs = vis.attrs
        self.apply_modifier_info(vis, proj)
        return proj


def _call_mapped_correction(satz, band_data, corrector, band_name):
    # need to convert to masked array
    orig_dtype = band_data.dtype
    band_data = np.ma.masked_where(np.isnan(band_data), band_data)
    res = corrector.get_correction(satz, band_name, band_data)
    return res.filled(np.nan).astype(orig_dtype, copy=False)


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
            satz = get_satellite_zenith_angle(band)
        satz = satz.data  # get dask array underneath

        logger.info('Correction for limb cooling')
        corrector = AtmosphericalCorrection(band.attrs['platform_name'],
                                            band.attrs['sensor'])

        atm_corr = da.map_blocks(_call_mapped_correction, satz, band.data,
                                 corrector=corrector,
                                 band_name=band.attrs['name'],
                                 meta=np.array((), dtype=band.dtype))
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
        - https://resources.eumetrain.org/IntGuide/PowerPoints/Channels/conversion.ppt
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
