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

from abc import ABCMeta

import numpy as np
import six

from pyresample.geometry import SwathDefinition
from satpy.dataset import combine_metadata


class BaseFileHandler(six.with_metaclass(ABCMeta, object)):

    def __init__(self, filename, filename_info, filetype_info):
        self.filename = str(filename)
        self.navigation_reader = None
        self.filename_info = filename_info
        self.filetype_info = filetype_info
        self.metadata = filename_info.copy()

    def __str__(self):
        return "<{}: '{}'>".format(self.__class__.__name__, self.filename)

    def __repr__(self):
        return str(self)

    def get_dataset(self, dataset_id, ds_info):
        raise NotImplementedError

    def get_area_def(self, dsid):
        raise NotImplementedError

    def get_bounding_box(self):
        """Get the bounding box of the files, as a (lons, lats) tuple.

        The tuple return should a lons and lats list of coordinates traveling
        clockwise around the points available in the file.
        """
        raise NotImplementedError

    @staticmethod
    def _combine(infos, func, *keys):
        res = {}
        for key in keys:
            if key in infos[0]:
                res[key] = func([i[key] for i in infos])
        return res

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
         - satellite_altitude
         - satellite_latitude
         - satellite_longitude

         Also, concatenate the areas.

        """
        combined_info = combine_metadata(*all_infos)

        new_dict = self._combine(all_infos, min, 'start_time', 'start_orbit')
        new_dict.update(self._combine(all_infos, max, 'end_time', 'end_orbit'))
        new_dict.update(self._combine(all_infos, np.mean,
                                      'satellite_longitude',
                                      'satellite_latitude',
                                      'satellite_altitude'))

        try:
            area = SwathDefinition(lons=np.ma.vstack([info['area'].lons for info in all_infos]),
                                   lats=np.ma.vstack([info['area'].lats for info in all_infos]))
            area.name = '_'.join([info['area'].name for info in all_infos])
            combined_info['area'] = area
        except KeyError:
            pass

        new_dict.update(combined_info)
        return new_dict

    @property
    def start_time(self):
        return self.filename_info['start_time']

    @property
    def end_time(self):
        return self.filename_info.get('end_time', self.start_time)

    @property
    def sensor_names(self):
        """List of sensors represented in this file."""
        raise NotImplementedError

    def available_datasets(self):
        """Get information of available datasets in file.

        This is used for dynamically specifying what datasets are available
        from a file instead of those listed in a YAML configuration file.

        Returns: Iterator of (DatasetID, dict) pairs where dict is the
                 dataset's metadata, similar to that specified in the YAML
                 configuration files.

        """
        raise NotImplementedError
