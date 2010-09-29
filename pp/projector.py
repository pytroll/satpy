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
from pyresample import image, utils, geometry, kd_tree

from pp import CONFIG_PATH
from pp.logger import LOG


class Projector(object):
    """This class define projector objects. They contain the mapping
    information necessary for projection purposes. For efficiency reasons,
    generated projectors are saved to disk for later reuse. The *recompute*
    flag can be used to regenerate the saved projector arrays.
    """
    
    in_area = None
    out_area = None
    _swath = False
    _cache = None
    _filename = None
    mode = "quick"
    
    area_file = os.path.join(CONFIG_PATH, "areas.def")

    def __init__(self, in_area, out_area,
                 in_latlons=None, precompute=False, mode="quick"):

        self.mode = mode

        # Setting up the input area
        try:
            self.in_area = utils.parse_area_file(self.area_file,
                                                 in_area)[0]
            in_id = in_area
        except utils.AreaNotFound:
            if isinstance(in_area, (geometry.AreaDefinition,
                                    geometry.SwathDefinition)):
                self.in_area = in_area
                in_id = in_area.area_id
            elif in_latlons is not None:
                self._swath = True
                self.in_area = geometry.SwathDefinition(lons=in_latlons[0],
                                                        lats=in_latlons[1])
                in_id = in_area
            else:
                raise utils.AreaNotFound("Input area must be defined in " +
                                         self.area_file + ", be an area object"
                                         " or longitudes/latitudes must be "
                                         "provided.")


        # Setting up the output area
        try:
            self.out_area = utils.parse_area_file(self.area_file,
                                                  out_area)[0]
            out_id = out_area
        except utils.AreaNotFound:
            if isinstance(out_area, (geometry.AreaDefinition,
                                    geometry.SwathDefinition)):
                self.out_area = out_area
                out_id = out_area.area_id
            else:
                raise utils.AreaNotFound("Output area must be defined in " +
                                         self.area_file + " or "
                                         "be an area object.")

        
        if self.in_area == self.out_area:
            return

        folder = "/var/tmp"
        filename = ("%s2%s%s.npz"%(in_id, out_id, mode))
        self._filename = os.path.join(folder, filename)

        if(not os.path.exists(self._filename)):
            LOG.info("Computing projection from %s to %s..."
                     %(in_id, out_id))


            if self.mode == "nearest":
                # FIXME: these value should be dynamically computed, or in a
                # configuration file.
                if self._swath:
                    radius = 5000
                else:
                    radius = 50000

                valid_index, valid_output_index, index_array, distance_array = \
                             kd_tree.get_neighbour_info(self.in_area,
                                                        self.out_area,
                                                        radius,
                                                        neighbours=1)
                del distance_array
                self._cache = {}
                self._cache['valid_index'] = valid_index
                self._cache['valid_output_index'] = valid_output_index
                self._cache['index_array'] = index_array

                if(precompute):
                    LOG.info("Saving projection from %s to %s..."
                             %(in_id, out_id))
                    np.savez(self._filename,
                             valid_index=valid_index,
                             valid_output_index=valid_output_index,
                             index_array=index_array)

            elif self.mode == "quick":
                ridx, cidx = \
                      utils.generate_quick_linesample_arrays(self.in_area,
                                                             self.out_area)
                                                    
                self._cache = {}
                self._cache['row_idx'] = ridx
                self._cache['col_idx'] = cidx

                if(precompute):
                    LOG.info("Saving projection from %s to %s..."
                             %(in_id, out_id))
                    np.savez(self._filename, row_idx=ridx, col_idx=cidx)
            else:
                raise ValueError("Unrecognised mode " + str(mode) + ".") 
            
        else:
            self._cache = np.load(self._filename)
        
        
    def project_array(self, data):
        """Project an array *data* along the given Projector object.
        """
        
        if self.in_area == self.out_area:
            return data

        if self.mode == "nearest":

            valid_index, valid_output_index, index_array = \
                         (self._cache['valid_index'],
                          self._cache['valid_output_index'],
                          self._cache['index_array'])

            res = kd_tree.get_sample_from_neighbour_info('nn',
                                                         self.out_area.shape,
                                                         data.ravel(),
                                                         valid_index,
                                                         valid_output_index,
                                                         index_array,
                                                         fill_value = None)

        elif self.mode == "quick":
            row_idx, col_idx = self._cache['row_idx'], self._cache['col_idx']
            img = image.ImageContainer(data, self.in_area, fill_value=None)
            res = np.ma.array(img.get_array_from_linesample(row_idx, col_idx),
                              dtype = data.dtype)
            
        

        return res
    



        
