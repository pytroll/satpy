#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
"""Composite classes for the AGRI instrument."""

import logging

from satpy.composites import GenericCompositor

LOG = logging.getLogger(__name__)


class SimulatedRed(GenericCompositor):
    """A single-band dataset resembling a Red (0.64 µm) band.

    This compositor creates a single band product by combining two
    other bands by preset amounts. The general formula with
    dependencies (d) and fractions (f) is::

        result = (f1 * d1 - f2 * d2) / f3

    See the `fractions` keyword argument for more information.
    The default setup is to use:

     - f1 = 1.0
     - f2 = 0.13
     - f3 = 0.87

    """

    def __init__(self, name, fractions=(1.0, 0.13, 0.87), **kwargs):
        """Initialize fractions for input channels.

        Args:
            name (str): Name of this composite
            fractions (iterable): Fractions of each input band to include in the result.

        """
        self.fractions = fractions
        super(SimulatedRed, self).__init__(name, **kwargs)

    def __call__(self, projectables, optional_datasets=None, **attrs):
        """Generate the single band composite."""
        c1, c2 = self.match_data_arrays(projectables)
        res = (c1 * self.fractions[0] - c2 * self.fractions[1]) / self.fractions[2]
        res.attrs = c1.attrs.copy()
        return super(SimulatedRed, self).__call__((res,), **attrs)
