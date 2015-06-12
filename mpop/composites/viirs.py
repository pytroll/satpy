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

    def __call__(self, scene):
        p1 = self.info["prerequisites"][0]
        p2 = self.info["prerequisites"][1]
        fog = scene[p1] - scene[p2]
        fog.info.update(self.info)
        fog.info["area"] = scene[p1].info["area"]
        fog.info["start_time"] = scene[p1].info["start_time"]
        fog.info["end_time"] = scene[p1].info["end_time"]
        fog.info["uid"] = self.info["uid"]
        return fog


class VIIRSTrueColor(CompositeBase):
    def __init__(self, *args, **kwargs):
        CompositeBase.__init__(self, *args, **kwargs)
        default_image_config={"mode": "RGB",
                              "stretch": "log"}
        # if image_config is not None:
        #     default_image_config.update(image_config)

        # self.uid = uid
        # self.prerequisites = ["M02", "M04", "M05"]
        # self.info["image_config"] = default_image_config

    def __call__(self, scene):
        # raise IncompatibleAreas
        return Projectable(uid=self.info["uid"],
                           data=np.concatenate(
                               ([scene["M05"].data], [scene["M04"].data], [scene["M02"].data]), axis=0),
                           area=scene["M05"].info["area"],
                           time_slot=scene.info["start_time"],
                           **self.info)


