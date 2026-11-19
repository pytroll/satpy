"""Modifier classes for corrections based on sun and other angles."""

from __future__ import annotations

import logging
import warnings
from typing import Literal

import numpy as np

import satpy
from satpy.modifiers import ModifierBase
from satpy.modifiers.angles import atmospheric_path_length_correction, sunzen_corr_cos, sunzen_reduction

logger = logging.getLogger(__name__)


class SunZenithCorrectorBase(ModifierBase):
    """Base class for sun zenith correction modifiers."""

    method: str | None = None

    def __init__(self, **kwargs):  # noqa: D417
        """Collect custom configuration values."""
        super(SunZenithCorrectorBase, self).__init__(**kwargs)

        if self.method is None:
            raise ValueError(
                f"{self.__class__.__name__} must define a 'method'."
            )

    def __call__(self, projectables, **info):
        """Generate the composite."""
        projectables = self.match_data_arrays(list(projectables) + list(info.get("optional_datasets", [])))
        vis = projectables[0]

        # Make sure to avoid alternative but mutually exclusive sun zenith angle corrections
        sunz_correction_methods = {"sunz_corrected", "effective_solar_pathlength_corrected"}
        if self.method in sunz_correction_methods:
            for correction in sunz_correction_methods:
                if vis.attrs.get(correction) or correction in vis.attrs.get("modifiers", ()):
                    logger.debug(
                        f"Sun zenith angle correction '{correction}' already applied. "
                        f"Skipping correction '{self.method}'."
                    )
                    return vis

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

    method = "sunz_corrected"

    def __init__(
        self,
        # 8< v1.1, change defaults to None and remove Literal["__default__"] as accepted type
        correction_limit: int | float | Literal["__default__"] | None = "__default__",
        max_sza: int | float | Literal["__default__"] | None = "__default__",
        # >8 v1.1
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

        # 8< v1.1
        use_legacy = satpy.config.get("use_legacy_sunz_correction")
        if use_legacy:
            self._warn_legacy_sunz_correction_with_defaults()
        # >8 v1.1

        # 8< v1.1, change to foo = self.foo instead
        correction_limit = self._get_correction_limit(use_legacy)
        max_sza = self._get_max_sza(use_legacy)
        # >8 v1.1

        res = proj.copy()
        res.data = sunzen_corr_cos(proj.data, coszen.data, correction_limit=correction_limit, max_sza=max_sza)
        return res

    # 8< v1.1
    def _warn_legacy_sunz_correction_with_defaults(self):
        if self.correction_limit == "__default__" or self.max_sza == "__default__":
            warnings.warn(
                "The default reduction of the standard Sun zenith angle correction above 88 degrees will "
                "be removed in Satpy v1.0 to allow computation of the true reflectance with the "
                "'sunz_corrected' modifier. To avoid overcorrection at high angles in (RGB) imagery, it is "
                "recommended to use the 'effective_solar_pathlength_corrected' modifier instead. "
                "To retain the current reduction after upgrading to Satpy v1.0, either set "
                "`satpy.config.set(use_legacy_sunz_correction=True)` or explicitly set `correction_limit: 88.0` "
                "and `max_sza: 95.0` in a local definition of the modifier. To opt in to the new "
                "behaviour before Satpy v1.0 and disable this warning, set "
                "`satpy.config.set(use_legacy_sunz_correction=False)`. The 'use_legacy_sunz_correction' configuration" \
                "key will be removed in Satpy 1.1.",
                UserWarning,
                stacklevel=2,
            )

    def _get_correction_limit(self, use_legacy):
        if self.correction_limit == "__default__":
            return 88.0 if use_legacy else None
        return self.correction_limit

    def _get_max_sza(self, use_legacy):
        if self.max_sza == "__default__":
            return 95.0 if use_legacy else None
        return self.max_sza
    # >8 v1.1


class EffectiveSolarPathLengthCorrector(SunZenithCorrectorBase):
    """Special sun zenith angle correction using the parameterization proposed by Li and Shibata.

    (2006): https://doi.org/10.1175/JAS3682.1

    This correction method is designed to reduce the over-correction of the standard
    sun zenith angle correction at high solar zenith angles, which is especially
    relevant for (RGB) imagery.

    In previous versions of Satpy, this correction could be capped or reduced at higher
    sun zenith angles by using the ``correction_limit`` and ``max_sza`` parameters.
    This has been disabled for this correction since the parameterization also deals with
    overcorrection at high solar zenith angles. If capping or reduction is still desireble
    it can be achieved by using the SunZenithCorrector with the same ``correction_limit``
    and ``max_sza`` parameters.

    """

    method = "effective_solar_pathlength_corrected"

    def __init__(
        self,
        # 8< v1.0
        correction_limit: int | float | None = None,
        max_sza: int | float | None = None,
        # >8 v1.0
        **kwargs,
    ):
        """Collect custom configuration values.

        Args:
            # 8< v1.0
            correction_limit:
                Solar zenith angle in degrees where correction limiting
                begins. Deprecated.

            max_sza:
                Maximum valid angle in degrees for solar zenith angle correction. Deprecated.
            # >8 v1.0

            **kwargs:
                Additional keyword arguments passed to the parent class.

        """
        # 8< v1.0
        if correction_limit is not None or max_sza is not None:
            msg = "The ``correction_limit`` and ``max_sza`` parameters have been deprecated and are no " \
                "longer used for the EffectiveSolarPathLengthCorrector and will be fully removed " \
                "in satpy v1.0. This is done since the parameterization by Li and Shibata (2006) " \
                "already accounts for overcorrection at high solar zenith angles. If capping or " \
                "reduction of the correction is still desirable it can be achieved by using the " \
                "``SunZenithCorrector`` with the same ``correction_limit`` and ``max_sza`` parameters."
            warnings.warn(msg, UserWarning, stacklevel=2)
        # >8 v1.0

        super(EffectiveSolarPathLengthCorrector, self).__init__(**kwargs)

    def _apply_correction(self, proj, coszen):
        logger.debug("Applying the effective solar atmospheric path length correction using the parameterization "
                     "by Li and Shibata (2006)")
        res = proj.copy()
        res.data = atmospheric_path_length_correction(proj.data, coszen.data)
        return res


class SunZenithReducer(SunZenithCorrectorBase):
    """Reduce signal strength at large sun zenith angles.

    Within a given sunz interval [correction_limit, max_sza] the strength of the signal is reduced following the
    formula:

      res = signal * reduction_factor

    where reduction_factor is a pixel-level value ranging from 0 to 1 within the sunz interval.

    The ``strength`` parameter can be used for a non-linear reduction within the sunz interval. A strength larger
    than 1.0 will decelerate the signal reduction towards the sunz interval extremes, whereas a strength
    smaller than 1.0 will accelerate the signal reduction towards the sunz interval extremes.

    """

    method = "sunz_reduced"

    def __init__(self, correction_limit=80., max_sza=90, strength=1.3, **kwargs):  # noqa: D417
        """Collect custom configuration values.

        Args:
            correction_limit (float): Solar zenith angle in degrees where to start the signal reduction.
            max_sza (float): Maximum solar zenith angle in degrees where to apply the signal reduction. Beyond
                             this solar zenith angle the signal will become zero.
            strength (float): The strength of the non-linear signal reduction.

        """
        self.correction_limit = correction_limit
        self.max_sza = max_sza
        self.strength = strength
        super(SunZenithReducer, self).__init__(**kwargs)
        if self.max_sza is None:
            raise ValueError("`max_sza` must be defined when using the SunZenithReducer.")

    def _apply_correction(self, proj, coszen):
        logger.debug(f"Applying sun-zenith signal reduction with correction_limit {self.correction_limit} deg,"
                     f" strength {self.strength}, and max_sza {self.max_sza} deg.")
        res = proj.copy()
        res.data = sunzen_reduction(proj.data, coszen.data,
                                    limit=self.correction_limit,
                                    max_sza=self.max_sza,
                                    strength=self.strength)
        return res
