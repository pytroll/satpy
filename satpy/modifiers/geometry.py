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

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from satpy.modifiers import ModifierBase
from satpy.modifiers.angles import sunzen_corr_cos, sunzen_reduction
from satpy.utils import atmospheric_path_length_correction

logger = logging.getLogger(__name__)


class SunZenithCorrectorBase(ModifierBase):
    """Base class for sun zenith correction modifiers."""

    def __init__(self, **kwargs):  # noqa: D417
        """Collect custom configuration values."""
        super(SunZenithCorrectorBase, self).__init__(**kwargs)

    def __call__(self, projectables, **info):
        """Generate the composite."""
        projectables = self.match_data_arrays(list(projectables) + list(info.get("optional_datasets", [])))
        vis = projectables[0]
        if vis.attrs.get("sunz_corrected"):
            logger.debug("Sun zenith correction already applied")
            return vis

        logger.debug("Applying Sun zenith angle correction")
        if not info.get("optional_datasets"):
            # we were not given SZA, generate cos(SZA)
            logger.debug("Computing sun zenith angles.")
            from .angles import get_cos_sza
            coszen = get_cos_sza(vis)
        else:
            # we were given the SZA, calculate the cos(SZA)
            coszen = np.cos(np.deg2rad(projectables[1]))

        proj = self._apply_correction(vis, coszen)
        proj.attrs = vis.attrs.copy()
        self.apply_modifier_info(vis, proj)
        return proj

    def _apply_correction(self, proj, coszen):
        raise NotImplementedError("Correction method shall be defined!")


class SunZenithCorrector(SunZenithCorrectorBase):
    """Standard Sun zenith angle correction using ``1 / cos(sunz)``.

    Modes
    -----
    The behavior of the correction depends on the combination of ``correction_limit`` and
    ``max_sza``:

    * ``correction_limit=None, max_sza=None``:
        Apply pure ``1 / cos(sunz)`` correction everywhere.

    * ``correction_limit=None, max_sza=<float>``:
        Apply ``1 / cos(sunz)`` correction up to ``max_sza``.
        Pixels with solar zenith angle > ``max_sza`` are set to 0.

    * ``correction_limit=<float>, max_sza=None``:
        Apply ``1 / cos(sunz)`` up to ``correction_limit``.
        Beyond this limit, the correction is clamped to the value at
        ``correction_limit`` (constant correction).

    * ``correction_limit=<float>, max_sza=<float>``:
        Apply ``1 / cos(sunz)`` up to ``correction_limit``.
        Between ``correction_limit`` and ``max_sza``, the correction is
        gradually reduced to 0.
        Pixels with solar zenith angle > ``max_sza`` are set to 0.

    Note that all corrections are undefined for ``cos(sunz) <= 0`` meaning that
    the reflectance data are forced to zero.

    To configure this in a YAML configuration file setting e.g. ``max_sza`` to ``None`` use:

    .. code-block:: yaml

      sunz_corrected:
        modifier: !!python/name:satpy.modifiers.SunZenithCorrector
        correction_limit: 88
        max_sza: !!null
        optional_prerequisites:
        - solar_zenith_angle

    """

    def __init__(
        self,
        correction_limit: Optional[float] = 88.0,
        max_sza: Optional[float] = 95.0,
        **kwargs,
    ):
        """Collect custom configuration values.

        Args:
            correction_limit:
                Solar zenith angle in degrees where correction limiting
                begins.

            max_sza:
                Maximum valid angle in degrees for solar zenith angle correction.

                Pixels with solar zenith angles greater than
                ``max_sza`` are set to 0.

            **kwargs:
                Additional keyword arguments passed to the parent class.

        """
        self.correction_limit = correction_limit
        self.max_sza = max_sza
        super(SunZenithCorrector, self).__init__(**kwargs)

    def _apply_correction(self, proj, coszen):
        res = proj.copy()
        res.data = sunzen_corr_cos(proj.data, coszen.data, correction_limit=self.correction_limit, max_sza=self.max_sza)
        return res


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
        modifier: !!python/name:satpy.modifiers.EffectiveSolarPathLengthCorrector
        max_sza: !!null
        optional_prerequisites:
        - solar_zenith_angle

    """

    def __init__(self, correction_limit=88., **kwargs):  # noqa: D417
        """Collect custom configuration values.

        Args:
            correction_limit (float): Maximum solar zenith angle to apply the
                correction in degrees. If ``max_sza`` is ``None``, pixels beyond this limit have a
                constant correction applied. Otherwise, the correction is gradually reduced to 0 at
                ``max_sza``. Default 88.
            max_sza (float): Maximum solar zenith angle in degrees that is
                considered valid and correctable. Default 95.0.

        """
        self.correction_limit = correction_limit
        super(EffectiveSolarPathLengthCorrector, self).__init__(**kwargs)

    def _apply_correction(self, proj, coszen):
        logger.debug("Apply the effective solar atmospheric path length correction method by Li and Shibata")
        return atmospheric_path_length_correction(proj, coszen, limit=self.correction_limit, max_sza=self.max_sza)


class SunZenithReducer(SunZenithCorrectorBase):
    """Reduce signal strength at large sun zenith angles.

    Within a given sunz interval [correction_limit, max_sza] the strength of the signal is reduced following the
    formula:

      res = signal * reduction_factor

    where reduction_factor is a pixel-level value ranging from 0 to 1 within the sunz interval.

    The `strength` parameter can be used for a non-linear reduction within the sunz interval. A strength larger
    than 1.0 will decelerate the signal reduction towards the sunz interval extremes, whereas a strength
    smaller than 1.0 will accelerate the signal reduction towards the sunz interval extremes.

    """

    def __init__(self, correction_limit=80., max_sza=90, strength=1.3, **kwargs):  # noqa: D417
        """Collect custom configuration values.

        Args:
            correction_limit (float): Solar zenith angle in degrees where to start the signal reduction.
            max_sza (float): Maximum solar zenith angle in degrees where to apply the signal reduction. Beyond
                             this solar zenith angle the signal will become zero.
            strength (float): The strength of the non-linear signal reduction.

        """
        self.correction_limit = correction_limit
        self.strength = strength
        super(SunZenithReducer, self).__init__(max_sza=max_sza, **kwargs)
        if self.max_sza is None:
            raise ValueError("`max_sza` must be defined when using the SunZenithReducer.")

    def _apply_correction(self, proj, coszen):
        logger.debug(f"Applying sun-zenith signal reduction with correction_limit {self.correction_limit} deg,"
                     f" strength {self.strength}, and max_sza {self.max_sza} deg.")
        res = proj.copy()
        sunz = np.rad2deg(np.arccos(coszen.data))
        res.data = sunzen_reduction(proj.data, sunz,
                                    limit=self.correction_limit,
                                    max_sza=self.max_sza,
                                    strength=self.strength)
        return res
