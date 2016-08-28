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

from satpy.projectable import combine_info

# what about file pattern and config ?
class BaseFileHandler(object):

    def __init__(self, filename, filename_info, **kwargs):
        self.filename = filename
        self.navigation_reader = None
        self.filename_info = filename_info

    def get_dataset(self, dataset_id, ds_info, out=None):
        raise NotImplementedError

    def get_shape(self, dataset_id):
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

        """
        combined_info = combine_info(*all_infos)
        if 'start_time' not in combined_info and 'start_time' in all_infos[0]:
            combined_info['start_time'] = min(i['start_time'] for i in all_infos)
        if 'end_time' not in combined_info and 'end_time' in all_infos[0]:
            combined_info['end_time'] = max(i['end_time'] for i in all_infos)
        if 'start_orbit' not in combined_info and 'start_orbit' in all_infos[0]:
            combined_info['start_orbit'] = min(i['start_orbit'] for i in all_infos)
        if 'end_orbit' not in combined_info and 'end_orbit' in all_infos[0]:
            combined_info['end_orbit'] = max(i['end_orbit'] for i in all_infos)
        return combined_info

    def start_time(self):
        raise NotImplementedError

    def end_time(self):
        raise NotImplementedError


class GeoFileHandler(BaseFileHandler):

    def get_area(self, nav_name, nav_info, lon_out, lat_out):
        raise NotImplementedError
