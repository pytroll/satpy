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

from pyresample.geometry import AreaDefinition

from satpy.composites import RGBCompositor
from satpy.dataset import Dataset

LOG = logging.getLogger(__name__)


class TrueColor(RGBCompositor):

    def __call__(self, projectables, **info):

        c01, c02, c03 = projectables

        r = c02
        b = c01
        g = (c01 + c02) / 2 * 0.93 + 0.07 * c03

        return super(TrueColor, self).__call__((r, g, b), **info)
