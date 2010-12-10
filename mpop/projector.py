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
import ConfigParser

import numpy as np
from pyresample import image, utils, geometry, kd_tree

from mpop import CONFIG_PATH
from mpop.logger import LOG

CONF = ConfigParser.ConfigParser()
CONF.read(os.path.join(CONFIG_PATH, "mpop.cfg"))

try:
    AREA_FILE = os.path.join(CONF.get("projector", "area_directory") or CONFIG_PATH,
                             CONF.get("projector", "area_file"))
except ConfigParser.NoSectionError:
    AREA_FILE = ""
    LOG.warning("Couldn't find the mpop.cfg file. "
                "Do you have one ? is it in $PPP_CONFIG_DIR ?")

def get_area_def(area):
    """Get the definition of *area* from file. The file is defined to use is to
    be placed in the $PPP_CONFIG_DIR directory, and its name is defined in
    mpop's configuration file.
    """
    return utils.parse_area_file(AREA_FILE, area)[0]

class Projector(object):
    """This class define projector objects. They contain the mapping
    information necessary for projection purposes. For efficiency reasons,
    generated projectors can be saved to disk for later reuse. Use the
    :meth:`save` method for this.
    """
    
    in_area = None
    out_area = None
    _swath = False
    _cache = None
    _filename = None
    mode = "quick"
    
    def __init__(self, in_area, out_area,
                 in_latlons=None, mode="quick"):

        # TODO:
        # - Rework so that in_area and out_area can be lonlats.
        # - Add a recompute flag ?

        self.mode = mode

        # Setting up the input area
        try:
            self.in_area = get_area_def(in_area)
            in_id = in_area
        except (utils.AreaNotFound, AttributeError):
            if isinstance(in_area, geometry.AreaDefinition):
                self.in_area = in_area
                in_id = in_area.area_id
            elif isinstance(in_area, geometry.SwathDefinition):
                self.in_area = in_area
                self._swath = True
                in_id = in_area.area_id
            elif in_latlons is not None:
                self._swath = True
                self.in_area = geometry.SwathDefinition(lons=in_latlons[0],
                                                        lats=in_latlons[1])
                in_id = in_area
            else:
                raise utils.AreaNotFound("Input area " +
                                         str(in_area) +
                                         " must be defined in " +
                                         AREA_FILE + ", be an area object"
                                         " or longitudes/latitudes must be "
                                         "provided.")


        # Setting up the output area
        try:
            self.out_area = get_area_def(out_area)
            out_id = out_area
        except (utils.AreaNotFound, AttributeError):
            if isinstance(out_area, (geometry.AreaDefinition,
                                    geometry.SwathDefinition)):
                self.out_area = out_area
                out_id = out_area.area_id
            else:
                raise utils.AreaNotFound("Output area " +
                                         str(out_area) +
                                         " must be defined in " +
                                         AREA_FILE + " or "
                                         "be an area object.")

        if self.in_area == self.out_area:
            return

        filename = (in_id + "2" + out_id + "_" + mode + ".npz")

        projections_directory = "/var/tmp"
        try:
            projections_directory = CONF.get("projector",
                                             "projections_directory")
        except ConfigParser.NoSectionError:
            pass
        
        self._filename = os.path.join(projections_directory, filename)

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

            elif self.mode == "quick":
                ridx, cidx = \
                      utils.generate_quick_linesample_arrays(self.in_area,
                                                             self.out_area)
                                                    
                self._cache = {}
                self._cache['row_idx'] = ridx
                self._cache['col_idx'] = cidx

            else:
                raise ValueError("Unrecognised mode " + str(mode) + ".") 
            
        else:
            self._cache = np.load(self._filename)

    def save(self, resave=False):
        """Save the precomputation to disk, and overwrite existing file in case
        *resave* is true.
        """
        if (not os.path.exists(self._filename)) or resave:
            LOG.info("Saving projection to " +
                     self._filename)
            np.savez(self._filename, **self._cache)
        
        
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
    



        
