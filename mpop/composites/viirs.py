#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2015

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   David Hoese <david.hoese@ssec.wisc.edu>

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

"""Composite classes for the VIIRS instrument.
"""

from mpop.composites import CompositeBase
from mpop.projectable import Projectable
import numpy as np


class VIIRSFog(CompositeBase):
    def __init__(self, *args, **kwargs):
        CompositeBase.__init__(self, *args, **kwargs)

    def __call__(self, projectables, nonprojectables=None, **info):
        if len(projectables) != 2:
            raise ValueError("Expected 2 projectables, got %d" % (len(projectables),))

        p1, p2 = projectables
        fog = p1 - p2
        fog.info.update(self.info)
        fog.info["area"] = p1.info["area"]
        fog.info["start_time"] = p1.info["start_time"]
        fog.info["end_time"] = p1.info["end_time"]
        fog.info["name"] = self.info["name"]
        fog.info.setdefault("mode", "L")
        return fog


class VIIRSTrueColor(CompositeBase):
    def __init__(self, *args, **kwargs):
        CompositeBase.__init__(self, *args, **kwargs)

    def __call__(self, projectables, nonprojectables=None, **info):
        if len(projectables) != 3:
            raise ValueError("Expected 3 projectables, got %d" % (len(projectables),))

        # raise IncompatibleAreas
        p1, p2, p3 = projectables
        info = p1.info.copy()
        info.update(**self.info)
        info["name"] = self.info["name"]
        info.setdefault("mode", "RGB")
        return Projectable(
                           data=np.concatenate(
                               ([p1.data], [p2.data], [p2.data]), axis=0),
                           **info)


