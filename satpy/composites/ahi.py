#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2015

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#
#   AHI specific composites:
#   Balthasar Indermühle <balt@inside.net>

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

import logging

import numpy as np
import six

from pyresample.geometry import AreaDefinition
from satpy.composites import CompositeBase, IncompatibleAreas
from satpy.projectable import Projectable, combine_info
from satpy.readers import DatasetID
from satpy.composites import RGBCompositor

LOG = logging.getLogger(__name__)


class GreenCorrector(CompositeBase):
    """Corrector of the AHI green band to compensate for the deficit of
    chlorophyl signal.
    """

    def __call__(self, projectables, optional_datasets=None, **info):
        """Boost vegetation effect thanks to NIR (0.8µm) band."""

        (green, nir) = projectables

        LOG.info('Boosting vegetation on green band')

        proj = Projectable(green * 0.85 + nir * 0.15,
                           copy=False,
                           **green.info)
        self.apply_modifier_info(green, proj)

        return proj


class Reducer2(CompositeBase):
    """Reduce the size of the composite."""

    def __call__(self, projectables, optional_datasets=None, **info):
        (band,) = projectables

        factor = 2

        # proj = Projectable(band[::factor, ::factor], copy=False, **band.info)
        newshape = (band.shape[0] / factor, factor,
                    band.shape[1] / factor, factor)
        proj = Projectable(band.reshape(newshape).mean(axis=3).mean(axis=1),
                           copy=False, **band.info)

        old_area = proj.info['area']
        proj.info['area'] = AreaDefinition(old_area.area_id,
                                           old_area.name,
                                           old_area.proj_id,
                                           old_area.proj_dict,
                                           old_area.x_size / factor,
                                           old_area.y_size / factor,
                                           old_area.area_extent)
        proj.info['resolution'] *= factor
        self.apply_modifier_info(band, proj)
        return proj


class Reducer4(CompositeBase):
    """Reduce the size of the composite."""

    def __call__(self, projectables, optional_datasets=None, **info):
        (band,) = projectables

        factor = 4

        #proj = Projectable(band[::factor, ::factor], copy=False, **band.info)
        newshape = (band.shape[0] / factor, factor,
                    band.shape[1] / factor, factor)
        proj = Projectable(band.reshape(newshape).mean(axis=3).mean(axis=1),
                           copy=False, **band.info)

        old_area = proj.info['area']
        proj.info['area'] = AreaDefinition(old_area.area_id,
                                           old_area.name,
                                           old_area.proj_id,
                                           old_area.proj_dict,
                                           old_area.x_size / factor,
                                           old_area.y_size / factor,
                                           old_area.area_extent)
        proj.info['resolution'] *= factor
        self.apply_modifier_info(band, proj)
        return proj

class Airmass(RGBCompositor):

    def __call__(self, projectables, *args, **kwargs):
        """Make an airmass RGB image composite (Himawari 8 version MSC JMA)

        +--------------------+--------------------+--------------------+
        | Channels           | Temp               | Gamma              |
        +====================+====================+====================+
        | WV6.2 - WV7.3      |     -25 to 0 K     | gamma 1            |
        +--------------------+--------------------+--------------------+
        | IR9.6 - IR10.4     |     -40 to 5 K     | gamma 1            |
        +--------------------+--------------------+--------------------+
        | WV6.2              |   243 to 208 K     | gamma 1            |
        +--------------------+--------------------+--------------------+
        """
        try:
            res = RGBCompositor.__call__(self, (projectables[0] - projectables[1],
                                                projectables[2] - projectables[3],
                                                projectables[0]), *args, **kwargs)
        except ValueError:
            raise IncompatibleAreas
        return res

class Dayconvective(RGBCompositor):

    def __call__(self, projectables, *args, **kwargs):
        """Make a day convective RGB image composite (Himawari 8 version MSC JMA)

        +--------------------+--------------------+--------------------+
        | Channels           | Temp               | Gamma              |
        +====================+====================+====================+
        | WV6.2 - WV7.3      |     -35 to 5 K     | gamma 1            |
        +--------------------+--------------------+--------------------+
        | IR3.9 - IR10.4     |     -5 to 60 K     | gamma 0.5          |
        +--------------------+--------------------+--------------------+
        | NIR1.6 - VIS0.6    |    -75 to 25 K     | gamma 1            |
        +--------------------+--------------------+--------------------+
        """
        try:
            res = RGBCompositor.__call__(self, (projectables[3] - projectables[4],
                                                projectables[2] - projectables[5],
                                                projectables[1] - projectables[0]), *args, **kwargs)
        except ValueError:
            raise IncompatibleAreas
        return res

class Ash(RGBCompositor):

    def __call__(self, projectables, *args, **kwargs):
        """Make a day convective RGB image composite (Himawari 8 version MSC JMA)

        +--------------------+--------------------+--------------------+
        | Channels           | Temp               | Gamma              |
        +====================+====================+====================+
        | IR12.3 - IR10.4    |     -4 to 2 K      | gamma 1            |
        +--------------------+--------------------+--------------------+
        | IR10.4 - IR8.6     |     -4 to 5 K      | gamma 1            |
        +--------------------+--------------------+--------------------+
        |       IR10.4       |    243 to 208 K    | gamma 1            |
        +--------------------+--------------------+--------------------+
        """
        try:
            res = RGBCompositor.__call__(self, (projectables[2] - projectables[1],
                                                projectables[1] - projectables[0],
                                                projectables[1]), *args, **kwargs)
        except ValueError:
            raise IncompatibleAreas
        return res

class MicrophysicsDay(RGBCompositor):

    def __call__(self, projectables, *args, **kwargs):
        """Make a day Microphysics/Nephanalysis RGB image composite (Himawari 8 version MSC JMA)

        +--------------------+--------------------+--------------------+
        | Channels           | Temp               | Gamma              |
        +====================+====================+====================+
        |     NIR0.86        |     0 - 100 %      | gamma 1            |
        +--------------------+--------------------+--------------------+
        |      IR3.9         |  Summer: 0 - 60 %  | gamma 2.5          |
        |                    |  Winter: 0 - 25 %  | gamma 1.5          |
        +--------------------+--------------------+--------------------+
        |      IR10.4        |    203 to 323 K    | gamma 1, reverse   |
        +--------------------+--------------------+--------------------+
        """
        try:
            res = RGBCompositor.__call__(self, (projectables[0],
                                                projectables[1],
                                                projectables[2]), *args, **kwargs)
        except ValueError:
            raise IncompatibleAreas
        return res

class MicrophysicsNight(RGBCompositor):

    def __call__(self, projectables, *args, **kwargs):
        """Make a night microphysics/nephanalysis RGB image composite (Himawari 8 version MSC JMA)

        +--------------------+--------------------+--------------------+
        | Channels           | Temp               | Gamma              |
        +====================+====================+====================+
        |  IR12.4 - IR10.4   |    -4 to 2 K       | gamma 1            |
        +--------------------+--------------------+--------------------+
        |   IR10.4 - IR3.9   |     0 to 10 K      | gamma 1            |
        +--------------------+--------------------+--------------------+
        |      IR10.4        |    243 to 293 K    | gamma 1            |
        +--------------------+--------------------+--------------------+
        """
        try:
            res = RGBCompositor.__call__(self, (projectables[2] - projectables[1],
                                                projectables[1] - projectables[0],
                                                projectables[2]), *args, **kwargs)
        except ValueError:
            raise IncompatibleAreas
        return res
