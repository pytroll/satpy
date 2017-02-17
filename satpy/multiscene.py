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

    base = Dataset(datasets[0], copy=True)
    for dataset in datasets[1:]:
        base_mask = np.ma.getmaskarray(base)
        other_mask = np.ma.getmaskarray(dataset)
        base.mask = np.logical_and(base_mask, other_mask)
        not_masked = np.logical_not(other_mask)
        base[not_masked] = dataset[not_masked]

    return base


def stack_time(datasets):
    """Oldest time at the bottom."""

    return stack(sorted(datasets, key=lambda x: x.info['start_time']))


class MultiScene(object):

    def __init__(self, layers):
        self.layers = layers

    def load(self, *args, **kwargs):
        """Load the required datasets from the multiple scenes."""
        for layer in self.layers:
            layer.load(*args, **kwargs)

    def resample(self, destination, **kwargs):
        """Resample the multiscene."""
        return MultiScene([layer.resample(destination, **kwargs) for layer in self.layers])

    def blend(self, blend_function=stack):
        """Blend the datasets into one scene."""
        scn = Scene()
        common_datasets = None
        for layer in self.layers:
            if common_datasets is None:
                common_datasets = set(
                    [dataset.id.name for dataset in layer])
            else:
                common_datasets &= set(
                    [dataset.id.name for dataset in layer])
        for dataset_id in common_datasets:
            datasets = [layer[dataset_id] for layer in self.layers]
            scn[dataset_id] = blend_function(datasets)

        return scn


if __name__ == '__main__':
    from datetime import datetime
    from satpy.utils import debug_on

    debug_on()
    scenes = [
        Scene(platform_name="Meteosat-10", sensor="seviri",
              start_time=datetime(2016, 9, 6, 11, 0),
              base_dir="/home/a001673/data/satellite/Meteosat-10/seviri/lvl1.5/2015/04/20/HRIT"),

        Scene(platform_name="SNPP", sensor="viirs",
              start_time=datetime(2016, 9, 6, 10, 51),
              end_time=datetime(2016, 9, 6, 11, 0),
              base_dir="/home/a001673/data/satellite/Suomi-NPP/viirs/lvl1b/2015/03/11/SDR")
    ]

    mscn = MultiScene(scenes)
    mscn.load(['overview_sun'])
    mscn = mscn.resample('eurol')
    scn = mscn.blend()
    scn.save_dataset('overview_sun', '/tmp/blend.png')
