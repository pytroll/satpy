#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016.

# Author(s):

#   David Hoese <david.hoese@ssec.wisc.edu>

# This file is part of satpy.

# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.

from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np
import six

from pyresample.geometry import SwathDefinition
from satpy.dataset import combine_info


# what about file pattern and config ?


class BaseFileHandler(six.with_metaclass(ABCMeta, object)):

    def __init__(self, filename, filename_info, filetype_info):
        self.filename = filename
        self.navigation_reader = None
        self.filename_info = filename_info
        self.filetype_info = filetype_info
        self.metadata = filename_info.copy()

    def __str__(self):
        return "<{}: '{}'>".format(self.__class__.__name__, self.filename)

    def __repr__(self):
        return str(self)

    def get_dataset(self, dataset_id, ds_info, out=None,
                    xslice=slice(None), yslice=slice(None)):
        raise NotImplementedError

    def get_shape(self, dataset_id, ds_info):
        raise NotImplementedError

    def get_area_def(self, dsid):
        raise NotImplementedError

    def get_bounding_box(self):
        raise NotImplementedError

    def get_lonlats(self, nav_id, nav_info, lon_out=None, lat_out=None):
        raise NotImplementedError

    def combine_info(self, all_infos):
        """Combine metadata for multiple datasets.

        When loading data from multiple files it can be non-trivial to combine
        things like start_time, end_time, start_orbit, end_orbit, etc.

        By default this method will produce a dictionary containing all values
        that were equal across **all** provided info dictionaries.

        Additionally it performs the logical comparisons to produce the
        following if they exist:

         - start_time
         - end_time
         - start_orbit
         - end_orbit

         Also, concatenate the areas.

        """
        combined_info = combine_info(*all_infos)
        if 'start_time' not in combined_info and 'start_time' in all_infos[0]:
            combined_info['start_time'] = min(
                i['start_time'] for i in all_infos)
        if 'end_time' not in combined_info and 'end_time' in all_infos[0]:
            combined_info['end_time'] = max(i['end_time'] for i in all_infos)
        if 'start_orbit' not in combined_info and 'start_orbit' in all_infos[0]:
            combined_info['start_orbit'] = min(
                i['start_orbit'] for i in all_infos)
        if 'end_orbit' not in combined_info and 'end_orbit' in all_infos[0]:
            combined_info['end_orbit'] = max(i['end_orbit'] for i in all_infos)

        try:
            area = SwathDefinition(lons=np.ma.vstack([info['area'].lons for info in all_infos]),
                                   lats=np.ma.vstack([info['area'].lats for info in all_infos]))
            area.name = '_'.join([info['area'].name for info in all_infos])
            combined_info['area'] = area
        except KeyError:
            pass

        return combined_info

    @property
    def start_time(self):
        return self.filename_info['start_time']

    @property
    def end_time(self):
        return self.filename_info.get('end_time', self.start_time)
