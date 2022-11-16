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

import dask.array as da

from satpy.composites import GenericCompositor
from satpy.dataset import combine_metadata

LOG = logging.getLogger(__name__)


class GreenCorrector(GenericCompositor):
    """Corrector of the FCI or AHI green band.

    The green band in FCI and AHI deliberately misses the chlorophyll peak
    in order to focus on aerosol and ash rather than on vegetation.  This
    affects true colour RGBs, because vegetation looks brown rather than green.
    To make vegetation look greener again, this corrector allows
    to simulate the green band as a fraction of two or more other channels.

    To be used, the composite takes two or more input channels and a parameter
    ``fractions`` that should be a list of floats with the same length as the
    number of channels.

    For example, to simulate an FCI corrected green composite, one could use
    a combination of 93% from the green band (vis_05) and 7% from the
    near-infrared 0.8 µm band (vis_08)::

      corrected_green:
        compositor: !!python/name:satpy.composites.ahi.GreenCorrector
        fractions: [0.93, 0.07]
        prerequisites:
          - name: vis_05
            modifiers: [sunz_corrected, rayleigh_corrected]
          - name: vis_08
            modifiers: [sunz_corrected, rayleigh_corrected]
        standard_name: toa_bidirectional_reflectance

    Other examples can be found in the ``fci.yaml`` and ``ahi.yaml`` composite
    files in the satpy distribution.
    """

    def __init__(self, *args, fractions=(0.85, 0.15), **kwargs):
        """Set default keyword argument values."""
        # XXX: Should this be 0.93 and 0.07
        self.fractions = fractions
        super(GreenCorrector, self).__init__(*args, **kwargs)

    def __call__(self, projectables, optional_datasets=None, **attrs):
        """Boost vegetation effect thanks to NIR (0.8µm) band."""
        LOG.info('Boosting vegetation on green band')

        projectables = self.match_data_arrays(projectables)
        new_green = sum(fraction * value for fraction, value in zip(self.fractions, projectables))
        new_green.attrs = combine_metadata(*projectables)
        return super(GreenCorrector, self).__call__((new_green,), **attrs)


class NDVIHybridGreen(GenericCompositor):
    """Construct a hybrid green channel weigthed by NDVI.

    This NDVIHybridGreen correction follows the same approach as the HybridGreen compositor, but with a dynamic blend
    factor `f` that depends on the pixel-level Normalized Differece Vegetation Index (NDVI). The higher the NDVI, the
    smaller the contribution from the nir channel will be, following a liner relationship between the two ranges
    [ndvi_min, ndvi_max] and `fractions`.

    A new green channel using e.g. FCI data and the NDVIHybridGreen compositor can be defined like:

    ndvi_hybrid_green:
        compositor: !!python/name:satpy.composites.spectral.NDVIHybridGreen
        fractions: [0.15, 0.05]
        prerequisites:
          - name: vis_05
            modifiers: [sunz_corrected, rayleigh_corrected]
          - name: vis_08
            modifiers: [sunz_corrected, rayleigh_corrected]
        standard_name: toa_bidirectional_reflectance

    In this example, pixels with NDVI=0.0 (default `ndvi_min`) will be a weighted average with 85% contribution from the
    native green vis_05 channel and 15% from the near-infrared vis_08 channel, whereas pixels with an NDVI=1.0 (default
    `ndvi_max`) will be a weighted average with 95% contribution from the native green vis_05 channel and 5% from the
    near-infrared vis_08 channel. For other values of NDVI (within this range) a linear interpolation will be performed.
    """

    def __init__(self, *args, ndvi_min=0.0, ndvi_max=1.0, fractions=(0.15, 0.05), **kwargs):
        """Initialize class and set the NDVI limits and the corresponding blending fraction limits."""
        self.ndvi_min = ndvi_min
        self.ndvi_max = ndvi_max
        self.fractions = fractions
        super(NDVIHybridGreen, self).__init__(*args, **kwargs)

    def __call__(self, projectables, optional_datasets=None, **attrs):
        """Construct the hybrid green channel weighted  by NDVI."""
        projectables = self.match_data_arrays(projectables)

        ndvi = (projectables[2] - projectables[1]) / (projectables[2] + projectables[1])

        ndvi.data = da.where(ndvi > self.ndvi_min, ndvi, self.ndvi_min)
        ndvi.data = da.where(ndvi < self.ndvi_max, ndvi, self.ndvi_max)

        f = (ndvi - self.ndvi_min) / (self.ndvi_max - self.ndvi_min) * (self.fractions[1] - self.fractions[0]) \
            + self.fractions[0]

        output = (1 - f) * projectables[0] + f * projectables[2]
        output.attrs = combine_metadata(*projectables)

        return super(NDVIHybridGreen, self).__call__((output,), **attrs)
