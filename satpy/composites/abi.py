#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2015-2017

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Composite classes for the AHI instrument.
"""

import logging
from satpy.composites import GenericCompositor

LOG = logging.getLogger(__name__)


class SimulatedGreen(GenericCompositor):

    """A single-band dataset resembles a Green (0.55 µm)."""

    def __call__(self, projectables, optional_datasets=None, **attrs):
        c01, c02, c03 = self.check_areas(projectables)

        # Kaba:
        # res = (c01 + c02) * 0.45 + 0.1 * c03
        # EDC:
        # res = c01 * 0.45706946 + c02 * 0.48358168 + 0.06038137 * c03
        # Original attempt:
        res = (c01 + c02) / 2 * 0.93 + 0.07 * c03
        res.attrs = c03.attrs.copy()

        return super(SimulatedGreen, self).__call__((res,), **attrs)
