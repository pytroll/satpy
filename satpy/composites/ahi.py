#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2015

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
"""Composite classes for the VIIRS instrument.
"""

import logging

from pyresample.geometry import AreaDefinition
from satpy.composites import CompositeBase, IncompatibleAreas
from satpy.projectable import Projectable, combine_info
from satpy.readers import DatasetID

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
        self.apply_modifier_info(band, proj)

        old_area = proj.info['area']
        proj.info['area'] = AreaDefinition(old_area.area_id,
                                           old_area.name,
                                           old_area.proj_id,
                                           old_area.proj_dict,
                                           old_area.x_size / factor,
                                           old_area.y_size / factor,
                                           old_area.area_extent)
        proj.info['resolution'] *= factor
        proj.info['id'] = DatasetID(name=proj.info['id'].name,
                                    resolution=proj.info['resolution'],
                                    wavelength=proj.info['id'].wavelength,
                                    polarization=proj.info['id'].polarization,
                                    calibration=proj.info['id'].calibration,
                                    modifiers=proj.info['id'].modifiers)

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
        self.apply_modifier_info(band, proj)

        old_area = proj.info['area']
        proj.info['area'] = AreaDefinition(old_area.area_id,
                                           old_area.name,
                                           old_area.proj_id,
                                           old_area.proj_dict,
                                           old_area.x_size / factor,
                                           old_area.y_size / factor,
                                           old_area.area_extent)
        proj.info['resolution'] *= factor
        proj.info['id'] = DatasetID(name=proj.info['id'].name,
                                    resolution=proj.info['resolution'],
                                    wavelength=proj.info['id'].wavelength,
                                    polarization=proj.info['id'].polarization,
                                    calibration=proj.info['id'].calibration,
                                    modifiers=proj.info['id'].modifiers)

        return proj
