#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009, 2010, 2011, 2012, 2013, 2014.

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
import logging

import numpy as np
from pyresample import image, utils, geometry, kd_tree
from mpop import CONFIG_PATH

logger = logging.getLogger(__name__)

area_file = None


def get_area_file():
    global area_file
    if area_file:
        return area_file

    conf = ConfigParser.ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, "mpop.cfg"))

    try:
        area_file = os.path.join(conf.get("projector",
                                          "area_directory") or
                                 CONFIG_PATH,
                                 conf.get("projector", "area_file"))
    except ConfigParser.NoSectionError:
        area_file = ""
        logger.warning("Couldn't find the mpop.cfg file. "
                       "Do you have one ? is it in $PPP_CONFIG_DIR ?")
    return area_file


def get_area_def(area_name):
    """Get the definition of *area_name* from file. The file is defined to use
    is to be placed in the $PPP_CONFIG_DIR directory, and its name is defined
    in mpop's configuration file.
    """
    return utils.parse_area_file(get_area_file(), area_name)[0]


def _get_area_hash(area):
    """Calculate a (close to) unique hash value for a given area.
    """
    try:
        return hash(area.lons.tostring() + area.lats.tostring())
    except AttributeError:
        try:
            return hash(area.tostring())
        except AttributeError:
            return hash(str(area))


class Projector(object):

    """This class define projector objects. They contain the mapping
    information necessary for projection purposes. For efficiency reasons,
    generated projectors can be saved to disk for later reuse. Use the
    :meth:`save` method for this.

    To define a projector object, on has to specify *in_area* and *out_area*,
    and can also input the *in_lonlats* or the *mode* ('quick' which works only
    if both in- and out-areas are AreaDefinitions, or 'nearest'). *radius*
    defines the radius of influence for nearest neighbour search in 'nearest'
    mode.
    """

    def __init__(self, in_area, out_area,
                 in_latlons=None, mode=None,
                 radius=10000, nprocs=1):

        if (mode is not None and
                mode not in ["quick", "nearest"]):
            raise ValueError("Projector mode must be 'nearest' or 'quick'")

        self.area_file = get_area_file()

        self.in_area = None
        self.out_area = None
        self._cache = None
        self._filename = None
        self.mode = "quick"
        self.radius = radius
        self.conf = ConfigParser.ConfigParser()
        self.conf.read(os.path.join(CONFIG_PATH, "mpop.cfg"))

        # TODO:
        # - Rework so that in_area and out_area can be lonlats.
        # - Add a recompute flag ?

        # Setting up the input area
        try:
            self.in_area = get_area_def(in_area)
            in_id = in_area
        except (utils.AreaNotFound, AttributeError):
            try:
                in_id = in_area.area_id
                self.in_area = in_area
            except AttributeError:
                try:
                    self.in_area = geometry.SwathDefinition(lons=in_latlons[0],
                                                            lats=in_latlons[1])
                    in_id = in_area
                except TypeError:
                    raise utils.AreaNotFound("Input area " +
                                             str(in_area) +
                                             " must be defined in " +
                                             self.area_file +
                                             ", be an area object"
                                             " or longitudes/latitudes must be "
                                             "provided.")

        # Setting up the output area
        try:
            self.out_area = get_area_def(out_area)
            out_id = out_area
        except (utils.AreaNotFound, AttributeError):
            try:
                out_id = out_area.area_id
                self.out_area = out_area
            except AttributeError:
                raise utils.AreaNotFound("Output area " +
                                         str(out_area) +
                                         " must be defined in " +
                                         self.area_file + " or "
                                         "be an area object.")

        if self.in_area == self.out_area:
            return

        # choosing the right mode if necessary
        if mode is None:
            try:
                dicts = in_area.proj_dict, out_area.proj_dict
                del dicts
                self.mode = "quick"
            except AttributeError:
                self.mode = "nearest"
        else:
            self.mode = mode

        filename = (in_id + "2" + out_id + "_" +
                    str(_get_area_hash(self.in_area)) + "to" +
                    str(_get_area_hash(self.out_area)) + "_" +
                    self.mode + ".npz")

        projections_directory = "/var/tmp"
        try:
            projections_directory = self.conf.get("projector",
                                                  "projections_directory")
        except ConfigParser.NoSectionError:
            pass

        self._filename = os.path.join(projections_directory, filename)

        try:
            self._cache = {}
            self._file_cache = np.load(self._filename)
        except:
            logger.info("Computing projection from %s to %s...",
                        in_id, out_id)

            if self.mode == "nearest":
                valid_index, valid_output_index, index_array, distance_array = \
                    kd_tree.get_neighbour_info(self.in_area,
                                               self.out_area,
                                               self.radius,
                                               neighbours=1,
                                               nprocs=nprocs)
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

    def save(self, resave=False):
        """Save the precomputation to disk, and overwrite existing file in case
        *resave* is true.
        """
        if (not os.path.exists(self._filename)) or resave:
            logger.info("Saving projection to " +
                        self._filename)
            np.savez(self._filename, **self._cache)

    def project_array(self, data):
        """Project an array *data* along the given Projector object.
        """

        if self.mode == "nearest":
            if not 'valid_index' in self._cache:
                self._cache['valid_index'] = self._file_cache['valid_index']
                self._cache['valid_output_index'] = \
                    self._file_cache['valid_output_index']
                self._cache['index_array'] = self._file_cache['index_array']

            valid_index, valid_output_index, index_array = \
                (self._cache['valid_index'],
                 self._cache['valid_output_index'],
                 self._cache['index_array'])

            res = kd_tree.get_sample_from_neighbour_info('nn',
                                                         self.out_area.shape,
                                                         data,
                                                         valid_index,
                                                         valid_output_index,
                                                         index_array,
                                                         fill_value=None)

        elif self.mode == "quick":
            if not 'row_idx' in self._cache:
                self._cache['row_idx'] = self._file_cache['row_idx']
                self._cache['col_idx'] = self._file_cache['col_idx']
            row_idx, col_idx = self._cache['row_idx'], self._cache['col_idx']
            img = image.ImageContainer(data, self.in_area, fill_value=None)
            res = np.ma.array(img.get_array_from_linesample(row_idx, col_idx),
                              dtype=data.dtype)

        return res
