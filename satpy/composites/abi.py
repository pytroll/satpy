#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017 Satpy developers
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
"""Composite classes for the ABI instrument."""

import logging

from satpy.composites import GenericCompositor

LOG = logging.getLogger(__name__)


class SimulatedGreen(GenericCompositor):
    """A single-band dataset resembling a Green (0.55 µm) band.

    This compositor creates a single band product by combining three
    other bands in various amounts. The general formula with
    dependencies (d) and fractions (f) is::

        result = d1 * f1 + d2 * f2 + d3 * f3

    See the `fractions` keyword argument for more information.
    Common used fractions for ABI data with C01, C02, and C03 inputs include:

    - SatPy default (historical): (0.465, 0.465, 0.07)
    - `CIMSS (Kaba) <https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2018EA000379>`_: (0.45, 0.45, 0.10)
    - `EDC <http://edc.occ-data.org/goes16/python/>`_: (0.45706946, 0.48358168, 0.06038137)

    """

    def __init__(self, name, fractions=(0.465, 0.465, 0.07), **kwargs):
        """Initialize fractions for input channels.

        Args:
            name (str): Name of this composite
            fractions (iterable): Fractions of each input band to include in the result.

        """
        self.fractions = fractions
        super(SimulatedGreen, self).__init__(name, **kwargs)

    def __call__(self, projectables, optional_datasets=None, **attrs):
        """Generate the single band composite."""
        c01, c02, c03 = self.match_data_arrays(projectables)
        res = c01 * self.fractions[0] + c02 * self.fractions[1] + c03 * self.fractions[2]
        res.attrs = c03.attrs.copy()
        return super(SimulatedGreen, self).__call__((res,), **attrs)
