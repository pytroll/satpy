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
"""Modifier classes for corrections based on sun and other angles."""

import logging
import time
from weakref import WeakValueDictionary

import numpy as np
import xarray as xr

from satpy.modifiers import ModifierBase
from satpy.utils import sunzen_corr_cos, atmospheric_path_length_correction

logger = logging.getLogger(__name__)


class SunZenithCorrectorBase(ModifierBase):
    """Base class for sun zenith correction modifiers."""

    coszen = WeakValueDictionary()

    def __init__(self, max_sza=95.0, **kwargs):
        """Collect custom configuration values.

        Args:
            max_sza (float): Maximum solar zenith angle in degrees that is
                considered valid and correctable. Default 95.0.

        """
        self.max_sza = max_sza
        self.max_sza_cos = np.cos(np.deg2rad(max_sza)) if max_sza is not None else None
        super(SunZenithCorrectorBase, self).__init__(**kwargs)

    def __call__(self, projectables, **info):
        """Generate the composite."""
        projectables = self.match_data_arrays(list(projectables) + list(info.get('optional_datasets', [])))
        vis = projectables[0]
        if vis.attrs.get("sunz_corrected"):
            logger.debug("Sun zen correction already applied")
            return vis

        area_name = hash(vis.attrs['area'])
        key = (vis.attrs["start_time"], area_name)
        tic = time.time()
        logger.debug("Applying sun zen correction")
        coszen = self.coszen.get(key)
        if coszen is None and not info.get('optional_datasets'):
            # we were not given SZA, generate SZA then calculate cos(SZA)
            from pyorbital.astronomy import cos_zen
            logger.debug("Computing sun zenith angles.")
            lons, lats = vis.attrs["area"].get_lonlats(chunks=vis.data.chunks)

            coords = {}
            if 'y' in vis.coords and 'x' in vis.coords:
                coords['y'] = vis['y']
                coords['x'] = vis['x']
            coszen = xr.DataArray(cos_zen(vis.attrs["start_time"], lons, lats),
                                  dims=['y', 'x'], coords=coords)
            if self.max_sza is not None:
                coszen = coszen.where(coszen >= self.max_sza_cos)
            self.coszen[key] = coszen
        elif coszen is None:
            # we were given the SZA, calculate the cos(SZA)
            coszen = np.cos(np.deg2rad(projectables[1]))
            self.coszen[key] = coszen

        proj = self._apply_correction(vis, coszen)
        proj.attrs = vis.attrs.copy()
        self.apply_modifier_info(vis, proj)
        logger.debug("Sun-zenith correction applied. Computation time: %5.1f (sec)", time.time() - tic)
        return proj

    def _apply_correction(self, proj, coszen):
        raise NotImplementedError("Correction method shall be defined!")


class SunZenithCorrector(SunZenithCorrectorBase):
    """Standard sun zenith correction using ``1 / cos(sunz)``.

    In addition to adjusting the provided reflectances by the cosine of the
    solar zenith angle, this modifier forces all reflectances beyond a
    solar zenith angle of ``max_sza`` to 0. It also gradually reduces the
    amount of correction done between ``correction_limit`` and ``max_sza``. If
    ``max_sza`` is ``None`` then a constant correction is applied to zenith
    angles beyond ``correction_limit``.

    To set ``max_sza`` to ``None`` in a YAML configuration file use:

    .. code-block:: yaml

      sunz_corrected:
        compositor: !!python/name:satpy.composites.SunZenithCorrector
        max_sza: !!null
        optional_prerequisites:
        - solar_zenith_angle

    """

    def __init__(self, correction_limit=88., **kwargs):
        """Collect custom configuration values.

        Args:
            correction_limit (float): Maximum solar zenith angle to apply the
                correction in degrees. Pixels beyond this limit have a
                constant correction applied. Default 88.
            max_sza (float): Maximum solar zenith angle in degrees that is
                considered valid and correctable. Default 95.0.

        """
        self.correction_limit = correction_limit
        super(SunZenithCorrector, self).__init__(**kwargs)

    def _apply_correction(self, proj, coszen):
        logger.debug("Apply the standard sun-zenith correction [1/cos(sunz)]")
        return sunzen_corr_cos(proj, coszen, limit=self.correction_limit, max_sza=self.max_sza)


class EffectiveSolarPathLengthCorrector(SunZenithCorrectorBase):
    """Special sun zenith correction with the method proposed by Li and Shibata.

    (2006): https://doi.org/10.1175/JAS3682.1

    In addition to adjusting the provided reflectances by the cosine of the
    solar zenith angle, this modifier forces all reflectances beyond a
    solar zenith angle of `max_sza` to 0 to reduce noise in the final data.
    It also gradually reduces the amount of correction done between
    ``correction_limit`` and ``max_sza``. If ``max_sza`` is ``None`` then a
    constant correction is applied to zenith angles beyond
    ``correction_limit``.

    To set ``max_sza`` to ``None`` in a YAML configuration file use:

    .. code-block:: yaml

      effective_solar_pathlength_corrected:
        compositor: !!python/name:satpy.composites.EffectiveSolarPathLengthCorrector
        max_sza: !!null
        optional_prerequisites:
        - solar_zenith_angle

    """

    def __init__(self, correction_limit=88., **kwargs):
        """Collect custom configuration values.

        Args:
            correction_limit (float): Maximum solar zenith angle to apply the
                correction in degrees. Pixels beyond this limit have a
                constant correction applied. Default 88.
            max_sza (float): Maximum solar zenith angle in degrees that is
                considered valid and correctable. Default 95.0.

        """
        self.correction_limit = correction_limit
        super(EffectiveSolarPathLengthCorrector, self).__init__(**kwargs)

    def _apply_correction(self, proj, coszen):
        logger.debug("Apply the effective solar atmospheric path length correction method by Li and Shibata")
        return atmospheric_path_length_correction(proj, coszen, limit=self.correction_limit, max_sza=self.max_sza)
