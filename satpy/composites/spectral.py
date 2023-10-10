# Copyright (c) 2015-2022 Satpy developers
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
"""Composite classes for spectral adjustments."""

import logging
import warnings

import dask.array as da

from satpy.composites import GenericCompositor
from satpy.dataset import combine_metadata

LOG = logging.getLogger(__name__)


class SpectralBlender(GenericCompositor):
    """Construct new channel by blending contributions from a set of channels.

    This class can be used to compute weighted average of different channels.
    Primarily it's used to correct the green band of AHI and FCI in order to
    allow for proper true color imagery.

    Below is an example used to generate a corrected green channel for AHI using a weighted average from
    three channels, with 63% contribution from the native green channel (B02), 29% from the red channel (B03)
    and 8% from the near-infrared channel (B04)::

      corrected_green:
        compositor: !!python/name:satpy.composites.spectral.SpectralBlender
        fractions: [0.63, 0.29, 0.08]
        prerequisites:
          - name: B02
            modifiers: [sunz_corrected, rayleigh_corrected]
          - name: B03
            modifiers: [sunz_corrected, rayleigh_corrected]
          - name: B04
            modifiers: [sunz_corrected, rayleigh_corrected]
        standard_name: toa_bidirectional_reflectance

    Other examples can be found in the``ahi.yaml`` composite file in the satpy distribution.
    """

    def __init__(self, *args, fractions=(), **kwargs):
        """Set default keyword argument values."""
        self.fractions = fractions
        super().__init__(*args, **kwargs)

    def __call__(self, projectables, optional_datasets=None, **attrs):
        """Blend channels in projectables using the weights in self.fractions."""
        if len(self.fractions) != len(projectables):
            raise ValueError("fractions and projectables must have the same length.")

        projectables = self.match_data_arrays(projectables)
        new_channel = sum(fraction * value for fraction, value in zip(self.fractions, projectables))
        new_channel.attrs = combine_metadata(*projectables)
        return super().__call__((new_channel,), **attrs)


class HybridGreen(SpectralBlender):
    """Corrector of the FCI or AHI green band.

    The green band in FCI and AHI (and other bands centered at 0.51 microns) deliberately
    misses the chlorophyll spectral reflectance local maximum at 0.55 microns
    in order to focus on aerosol and ash rather than on vegetation. This
    affects true colour RGBs, because vegetation looks brown rather than green
    and barren surface types typically gets a reddish hue.

    To correct for this the hybrid green approach proposed by Miller et al. (2016, :doi:`10.1175/BAMS-D-15-00154.2`)
    is used. The basic idea is to include some contribution also from the 0.86 micron
    channel, which is known for its sensitivity to vegetation. The formula used for this is::

      hybrid_green = (1 - F) * R(0.51) + F * R(0.86)

    where F is a constant value, that is set to 0.15 by default in Satpy.

    For example, the HybridGreen compositor can be used as follows to construct a hybrid green channel for
    AHI, with 15% contibution from the near-infrared 0.85 µm band (B04) and the remaining 85% from the native
    green 0.51 µm band (B02)::

      hybrid_green:
        compositor: !!python/name:satpy.composites.spectral.HybridGreen
        fraction: 0.15
        prerequisites:
          - name: B02
            modifiers: [sunz_corrected, rayleigh_corrected]
          - name: B04
            modifiers: [sunz_corrected, rayleigh_corrected]
        standard_name: toa_bidirectional_reflectance

    Other examples can be found in the ``ahi.yaml`` and ``ami.yaml`` composite
    files in the satpy distribution.
    """

    def __init__(self, *args, fraction=0.15, **kwargs):
        """Set default keyword argument values."""
        fractions = (1 - fraction, fraction)
        super().__init__(fractions=fractions, *args, **kwargs)


class NDVIHybridGreen(SpectralBlender):
    """Construct a NDVI-weighted hybrid green channel.

    This green band correction follows the same approach as the HybridGreen compositor, but with a dynamic blend
    factor `f` that depends on the pixel-level Normalized Differece Vegetation Index (NDVI). The higher the NDVI, the
    smaller the contribution from the nir channel will be, following a liner (default) or non-linear relationship
    between the two ranges `[ndvi_min, ndvi_max]` and `limits`.

    As an example, a new green channel using e.g. FCI data and the NDVIHybridGreen compositor can be defined like::

      ndvi_hybrid_green:
        compositor: !!python/name:satpy.composites.spectral.NDVIHybridGreen
        ndvi_min: 0.0
        ndvi_max: 1.0
        limits: [0.15, 0.05]
        strength: 1.0
        prerequisites:
          - name: vis_05
            modifiers: [sunz_corrected, rayleigh_corrected]
          - name: vis_06
            modifiers: [sunz_corrected, rayleigh_corrected]
          - name: vis_08
            modifiers: [sunz_corrected ]
        standard_name: toa_bidirectional_reflectance

    In this example, pixels with NDVI=0.0 will be a weighted average with 15% contribution from the
    near-infrared vis_08 channel and the remaining 85% from the native green vis_05 channel, whereas
    pixels with NDVI=1.0 will be a weighted average with 5% contribution from the near-infrared
    vis_08 channel and the remaining 95% from the native green vis_05 channel. For other values of
    NDVI a linear interpolation between these values will be performed.

    A strength larger or smaller than 1.0 will introduce a non-linear relationship between the two ranges
    `[ndvi_min, ndvi_max]` and `limits`. Hence, a higher strength (> 1.0) will result in a slower transition
    to higher/lower fractions at the NDVI extremes. Similarly, a lower strength (< 1.0) will result in a
    faster transition to higher/lower fractions at the NDVI extremes.
    """

    def __init__(self, *args, ndvi_min=0.0, ndvi_max=1.0, limits=(0.15, 0.05), strength=1.0, **kwargs):
        """Initialize class and set the NDVI limits, blending fraction limits and strength."""
        if strength <= 0.0:
            raise ValueError(f"Expected stength greater than 0.0, got {strength}.")

        self.ndvi_min = ndvi_min
        self.ndvi_max = ndvi_max
        self.limits = limits
        self.strength = strength
        super().__init__(*args, **kwargs)

    def __call__(self, projectables, optional_datasets=None, **attrs):
        """Construct the hybrid green channel weighted by NDVI."""
        LOG.info(f"Applying NDVI-weighted hybrid-green correction with limits [{self.limits[0]}, "
                 f"{self.limits[1]}] and strength {self.strength}.")

        ndvi_input = self.match_data_arrays([projectables[1], projectables[2]])

        ndvi = (ndvi_input[1] - ndvi_input[0]) / (ndvi_input[1] + ndvi_input[0])

        ndvi.data = da.where(ndvi > self.ndvi_min, ndvi, self.ndvi_min)
        ndvi.data = da.where(ndvi < self.ndvi_max, ndvi, self.ndvi_max)

        # Introduce non-linearity to ndvi for non-linear scaling to NIR blend fraction
        if self.strength != 1.0:  # self._apply_strength() has no effect if strength = 1.0 -> no non-linear behaviour
            ndvi = self._apply_strength(ndvi)

        # Compute pixel-level NIR blend fractions from ndvi
        fraction = self._compute_blend_fraction(ndvi)

        # Prepare input as required by parent class (SpectralBlender)
        self.fractions = (1 - fraction, fraction)

        return super().__call__([projectables[0], projectables[2]], **attrs)

    def _apply_strength(self, ndvi):
        """Introduce non-linearity by applying strength factor.

        The method introduces non-linearity to the ndvi for a non-linear scaling from ndvi to blend fraction in
        `_compute_blend_fraction`. This can be used for a slower or faster transision to higher/lower fractions
        at the ndvi extremes. If strength equals 1.0, this operation has no effect on the ndvi.
        """
        ndvi = ndvi ** self.strength / (ndvi ** self.strength + (1 - ndvi) ** self.strength)

        return ndvi

    def _compute_blend_fraction(self, ndvi):
        """Compute pixel-level fraction of NIR signal to blend with native green signal.

        This method linearly scales the input ndvi values to pixel-level blend fractions within the range
        `[limits[0], limits[1]]` following this implementation <https://stats.stackexchange.com/a/281164>.
        """
        fraction = (ndvi - self.ndvi_min) / (self.ndvi_max - self.ndvi_min) * (self.limits[1] - self.limits[0]) \
            + self.limits[0]

        return fraction


class GreenCorrector(SpectralBlender):
    """Previous class used to blend channels for green band corrections.

    This method has been refactored to make it more generic. The replacement class is 'SpectralBlender' which computes
    a weighted average based on N number of channels and N number of corresponding weights/fractions. A new class
    called 'HybridGreen' has been created, which performs a correction of green bands centered at 0.51 microns
    following Miller et al. (2016, :doi:`10.1175/BAMS-D-15-00154.2`) in order to improve true color imagery.
    """

    def __init__(self, *args, fractions=(0.85, 0.15), **kwargs):
        """Set default keyword argument values."""
        warnings.warn(
            "'GreenCorrector' is deprecated, use 'SpectralBlender' instead, or 'HybridGreen' for hybrid green"
            " correction following Miller et al. (2016).",
            UserWarning,
            stacklevel=2
        )
        super().__init__(fractions=fractions, *args, **kwargs)
