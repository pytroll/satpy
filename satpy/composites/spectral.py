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
import numpy as np

from satpy.composites import GenericCompositor
from satpy.dataset import combine_metadata

LOG = logging.getLogger(__name__)


def _calc_norm_index(band1, band2):
    """Calculate an index (such as NDVI) from two input bands: `band2` will be subtracted from `band1`.

    For example, to calculate NDVI:
    `ndvi = _calc_norm_index(nir_band, red_band)`
    """
    return (band1 - band2) / (band1 + band2)


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
    smaller the contribution from the nir channel will be, following a liner relationship between the two ranges
    `[ndvi_min, ndvi_max]` and `limits`.

    As an example, a new green channel using e.g. FCI data and the NDVIHybridGreen compositor can be defined like::

      ndvi_hybrid_green:
        compositor: !!python/name:satpy.composites.spectral.NDVIHybridGreen
        ndvi_min: 0.0
        ndvi_max: 1.0
        limits: [0.15, 0.05]
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
    """

    def __init__(self, *args, ndvi_min=0.0, ndvi_max=1.0, limits=(0.15, 0.05), **kwargs):
        """Initialize class and set the NDVI limits and the corresponding blending fraction limits."""
        self.ndvi_min = ndvi_min
        self.ndvi_max = ndvi_max
        self.limits = limits
        super().__init__(*args, **kwargs)

    def __call__(self, projectables, optional_datasets=None, **attrs):
        """Construct the hybrid green channel weighted by NDVI."""
        ndvi_input = self.match_data_arrays([projectables[1], projectables[2]])

        ndvi = _calc_norm_index(ndvi_input[1], ndvi_input[0])

        ndvi.data = da.where(ndvi > self.ndvi_min, ndvi, self.ndvi_min)
        ndvi.data = da.where(ndvi < self.ndvi_max, ndvi, self.ndvi_max)

        fraction = (ndvi - self.ndvi_min) / (self.ndvi_max - self.ndvi_min) * (self.limits[1] - self.limits[0]) \
            + self.limits[0]
        self.fractions = (1 - fraction, fraction)

        return super().__call__([projectables[0], projectables[2]], **attrs)


class IndexedGreen(SpectralBlender):
    """Construct an index-weighted hybrid green channel.

    This green band correction follows the same approach as the HybridGreen compositor, but with a dynamic blend
    factor `f` that depends on the pixel-level vegetation (NDVI) and soil parameters (NDBI). In addition, two
    sensor-specific constants, veg_fac and soil_fac, are used that define the difference between the sensor spectral
    response and an idealised spectral response suited for vegetation.
    NDVI is calculated via: `(NIR - RED) / (NIR + RED)`
    NDBI is calculated via: `(RED - GRN) / (RED + GRN)`
    The final weighting factor is then found via:
    `f = max( ndvi * veg_fac, ndbi * soil_fac)`
    If `soil_fac` and `veg_fac` are not supplied, this method will produce results equivalent to `NDVIHybridGreen`
    """

    def __init__(self, *args,
                 ndvi_min=0.0, ndvi_max=0.7,
                 ndbi_min=0.0, ndbi_max=1.0,
                 veg_fac=1., soil_fac=0.,
                 limits=(0.15, 0.05), **kwargs):
        """Initialize class and set the index limits and the corresponding blending fraction limits."""
        self.ndvi_min = ndvi_min
        self.ndvi_max = ndvi_max
        self.ndbi_min = ndbi_min
        self.ndbi_max = ndbi_max
        self.veg_fac = veg_fac
        self.soil_fac = soil_fac
        self.limits = limits
        super().__init__(*args, **kwargs)

    def __call__(self, projectables, optional_datasets=None, **attrs):
        """Construct the hybrid green channel weighted by NDVI."""
        ndvi_input = self.match_data_arrays([projectables[1], projectables[2]])
        ndbi_input = self.match_data_arrays([projectables[0], projectables[1]])

        ndvi = _calc_norm_index(ndvi_input[1], ndvi_input[0])
        ndvi.data = np.clip(ndvi.data, self.ndvi_min, self.ndvi_max, ndvi.data)

        ndbi = _calc_norm_index(ndbi_input[1], ndbi_input[0])
        ndbi.data = np.clip(ndbi.data, self.ndbi_min, self.ndbi_max, ndbi.data)

        ndvi = ndvi * self.veg_fac
        ndbi = ndbi * self.soil_fac

        fraction = 0.5 * (ndvi + ndbi)
        self.fractions = (1 - fraction, fraction)

        tmp = super().__call__([projectables[0], projectables[2]], **attrs)

        return tmp


class IndexedRed(SpectralBlender):
    """Construct an index-weighted hybrid red channel.

    This red band correction follows the same approach as the IndexedGreen compositor, but with a dynamic blend
    factor `f` that depends on the pixel-level vegetation (NDVI) and soil parameters (NDBI). In addition, two
    sensor-specific constants, veg_fac and soil_fac, are used that define the difference between the sensor spectral
    response and an idealised spectral response suited for vegetation.
    NDVI is calculated via: `(NIR - RED) / (NIR + RED)`
    NDBI is calculated via: `(RED - GRN) / (RED + GRN)`
    The final weighting factor is then found via:
    `f = ndvi * blue * veg_frac + NDBI * nir * soil_frac + [1 - (NDVI * veg_frac + NDVI * soil_frac) * red`
    """

    def __init__(self, *args,
                 ndvi_min=0.2, ndvi_max=0.7,
                 ndbi_min=0.2, ndbi_max=0.7,
                 veg_fac=1., soil_fac=0.,
                 limits=(0.15, 0.05), **kwargs):
        """Initialize class and set the index limits and the corresponding blending fraction limits."""
        self.ndvi_min = ndvi_min
        self.ndvi_max = ndvi_max
        self.ndbi_min = ndbi_min
        self.ndbi_max = ndbi_max
        self.veg_fac = veg_fac
        self.soil_fac = soil_fac
        self.limits = limits
        super().__init__(*args, **kwargs)

    def __call__(self, projectables, optional_datasets=None, **attrs):
        """Construct the hybrid green channel weighted by NDVI."""
        ndvi_input = self.match_data_arrays([projectables[1], projectables[2]])
        ndbi_input = self.match_data_arrays([projectables[0], projectables[1]])

        ndvi = _calc_norm_index(ndvi_input[1], ndvi_input[0])
        ndvi.data = np.clip(ndvi.data, self.ndvi_min, self.ndvi_max, ndvi.data)

        ndbi = _calc_norm_index(ndbi_input[1], ndbi_input[0])
        ndbi.data = np.clip(ndbi.data, self.ndbi_min, self.ndbi_max, ndbi.data)

        ndvi = ndvi * self.veg_fac
        ndbi = ndbi * self.soil_fac

        fraction = da.where(ndvi > ndbi, ndvi, ndbi)
        self.fractions = (1 - fraction, fraction)

        tmp = super().__call__([projectables[0], projectables[2]], **attrs)

        return tmp


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
            " correction following Miller et al. (2016).", RuntimeWarning)
        super().__init__(fractions=fractions, *args, **kwargs)
