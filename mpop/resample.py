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

"""Shortcuts to resampling stuff.
"""

import collections

from pyresample.kd_tree import get_neighbour_info, get_sample_from_neighbour_info

cache = {}

def resample(projectables, destination_area, **kwargs):
    area_dict = {}
    if not isinstance(projectables, collections.Iterable):
        projectables = [projectables]
    for projectable in projectables:
        area_dict.setdefault(projectable.info["area"].area_name, []).append(projectable)

    for plist in area_dict.itervalues():
        source_area = plist[0].info["area"]
        data = (projectable.data for projectable in plist)
        resample_kd_tree_nearest(source_area, data, destination_area, **kwargs)


def memoize(func, keys):
    def inner(*args, **kwargs):
        memoization_key = list(args)
        for key in sorted(kwargs):
            if key in keys:
                memoization_key.append(kwargs[key])

        if not isinstance(args, collections.Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            return func(*args, **kwargs)
        if (func, str(memoization_key)) in cache:
            return cache[func, str(memoization_key)]
        else:
            value = func(*args, **kwargs)
            cache[func, str(memoization_key)] = value
        return value

    return inner


def resample_kd_tree_nearest(source_geo_def, data, target_geo_def,
                             radius_of_influence, epsilon=0, weight_funcs=None,
                             fill_value=0, reduce_data=True, nprocs=1, segments=None, with_uncert=False,
                             precompute=False):
    """Resamples using kd-tree approach"""

    m_get_neighbour_info = memoize(get_neighbour_info, keys=("neighbours", "epsilon"))

    valid_input_index, valid_output_index, index_array, distance_array = \
        m_get_neighbour_info(source_geo_def,
                             target_geo_def,
                             radius_of_influence,
                             neighbours=1,
                             epsilon=epsilon,
                             reduce_data=reduce_data,
                             nprocs=nprocs,
                             segments=segments)

    return get_sample_from_neighbour_info('nn',
                                          target_geo_def.shape,
                                          data, valid_input_index,
                                          valid_output_index,
                                          index_array,
                                          distance_array=distance_array,
                                          weight_funcs=weight_funcs,
                                          fill_value=fill_value,
                                          with_uncert=with_uncert)
