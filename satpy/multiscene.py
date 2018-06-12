#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016

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

"""MultiScene object to blend satellite data.
"""

import numpy as np

from satpy.dataset import Dataset
from satpy.scene import Scene


def stack(datasets):
    """First dataset at the bottom."""
    base = datasets[0].copy()
    for dataset in datasets[1:]:
        base = base.where(dataset.isnull(), dataset)
    return base


class MultiScene(object):
    """Container for multiple `Scene` objects."""

    def __init__(self, layers):
        """Initialize MultiScene and validate sub-scenes"""
        self.scenes = layers

    @property
    def loaded_dataset_ids(self):
        """Union of all Dataset IDs loaded by all children."""
        return set(ds_id for scene in self.scenes for ds_id in scene.keys())

    @property
    def shared_dataset_ids(self):
        """Dataset IDs shared by all children."""
        shared_ids = set(self.scenes[0].keys())
        for scene in self.scenes[1:]:
            shared_ids &= set(scene.keys())
        return shared_ids

    def load(self, *args, **kwargs):
        """Load the required datasets from the multiple scenes."""
        for layer in self.scenes:
            layer.load(*args, **kwargs)

    def resample(self, destination, **kwargs):
        """Resample the multiscene."""
        return self.__class__([scn.resample(destination, **kwargs)
                               for scn in self.scenes])

    def blend(self, blend_function=stack):
        """Blend the datasets into one scene."""
        new_scn = Scene()
        common_datasets = self.shared_dataset_ids
        for ds_id in common_datasets:
            datasets = [scn[ds_id] for scn in self.scenes if ds_id in scn]
            new_scn[ds_id] = blend_function(datasets)

        return new_scn
