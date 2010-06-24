#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
#   Martin Raspaud <martin.raspaud@smhi.se>
#   Adam Dybbroe <adam.dybbroe@smhi.se>

# This file is part of mpop.

# mpop is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# mpop is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with mpop.  If not, see <http://www.gnu.org/licenses/>.

"""This module handles coverage objects. Such objects are used to
transform area projected data by changing either the area or the
projection or both. A typical usage is to transform one large area in
satellite projection to an area of interest in polar projection for
example.
"""

import os
import numpy as np

from pp.logger import LOG
from pyresample import image, utils, swath

from pp import CONFIG_PATH

class Projector(object):
    """This class define projector objects. They contain the mapping
    information necessary for projection purposes. For efficiency reasons,
    generated projectors are saved to disk for later reuse. The *recompute*
    flag can be used to regenerate the saved projector arrays.
    """
    
    in_area = None
    out_area = None
    _lon = None
    _lat = None
    _swath = False
    _cache = None
    _filename = None
    
    area_file = os.path.join(CONFIG_PATH, "areas.def")

    def __init__(self, in_area_id, out_area_id,
                 in_latlons=None, precompute=False, mode="quick"):
        try:
            self.in_area = utils.parse_area_file(self.area_file,
                                                 in_area_id)[0]
        except utils.AreaNotFound:
            if in_latlons is not None:
                self._lat = in_latlons[0]
                self._lon = in_latlons[1]
                self._swath = True
            else:
                raise utils.AreaNotFound("Input area must be defined in " +
                                         self.area_file + " or "
                                         "longitudes/latitudes must be "
                                         "provided.")

        self.out_area = utils.parse_area_file(self.area_file,
                                              out_area_id)[0]

        folder = "/var/tmp"
        filename = ("%s2%s.npz"%(in_area_id, out_area_id))
        self._filename = os.path.join(folder, filename)

        if(not os.path.exists(self._filename)):
            LOG.info("Computing projection from %s to %s..."
                     %(in_area_id, out_area_id))

            if self._swath:
                valid_index, index_array, distance_array = \
                      swath.get_neighbour_info(self._lon.ravel(),
                                               self._lat.ravel(),
                                               self.out_area,
                                               5000,
                                               neighbours=1)
                del distance_array
                self._cache = {}
                self._cache['valid_index'] = valid_index
                self._cache['index_array'] = index_array

                if(precompute):
                    LOG.info("Saving projection from %s to %s..."
                             %(in_area_id, out_area_id))
                    np.savez(self._filename,
                             valid_index=valid_index,
                             index_array=index_array)
            else:
                if mode == "nearest":
                    ridx, cidx = \
                        utils.generate_nearest_neighbour_linesample_arrays(self.in_area,
                                                                           self.out_area,
                                                                           50000)
                else:
                    ridx, cidx = \
                        utils.generate_quick_linesample_arrays(self.in_area,
                        self.out_area)
                                                    
                self._cache = {}
                self._cache['row_idx'] = ridx
                self._cache['col_idx'] = cidx

                if(precompute):
                    LOG.info("Saving projection from %s to %s..."
                             %(in_area_id, out_area_id))
                    np.savez(self._filename, row_idx=ridx, col_idx=cidx)
            
        else:
            self._cache = np.load(self._filename)
        
        
    def project_array(self, data):
        """Project an array *data* along the given Projector object.
        """
        if self._swath:
            valid_index, index_array = (self._cache['valid_index'],
                                        self._cache['index_array'])

            res = swath.get_sample_from_neighbour_info('nn',
                                                       self.out_area.shape,
                                                       data.ravel(),
                                                       valid_index,
                                                       index_array,
                                                       fill_value = None)

            return res
        else:
            row_idx, col_idx = self._cache['row_idx'], self._cache['col_idx']
            img = image.ImageContainer(data, self.in_area)
            return np.ma.array(img.get_array_from_linesample(row_idx, col_idx,
                                                             fill_value = None),
                               dtype = data.dtype)
            
        




        
