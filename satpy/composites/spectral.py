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
