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

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
from pyresample.kd_tree import get_neighbour_info, get_sample_from_neighbour_info
from logging import getLogger
import numpy as np
import hashlib
import json
import os
from mpop import get_config, utils
from ConfigParser import NoSectionError

LOG = getLogger(__name__)

CACHE_SIZE = 10

def get_area_file():
    conf = get_config("mpop.cfg")

    try:
        return os.path.join(conf.get("projector",
                                     "area_directory") or
                            CONFIG_PATH,
                            conf.get("projector", "area_file"))
    except NoSectionError:
        LOG.warning("Couldn't find the mpop.cfg file. "
                    "Do you have one ? is it in $PPP_CONFIG_DIR ?")


def get_area_def(area_name):
    """Get the definition of *area_name* from file. The file is defined to use
    is to be placed in the $PPP_CONFIG_DIR directory, and its name is defined
    in mpop's configuration file.
    """
    return utils.parse_area_file(get_area_file(), area_name)[0]


class BaseResampler(object):
    """
    The base resampler class. Abstract.
    """

    def __init__(self, source_geo_def, target_geo_def):
        """
        :param source_geo_def: The source area
        :param target_geo_def: The destination area
        """

        self.source_geo_def = source_geo_def
        self.target_geo_def = target_geo_def
        self.cache = {}

    def precompute(self, **kwargs):
        """Do the precomputation
        """
        pass

    def compute(self, data, **kwargs):
        """Do the actual resampling
        """
        raise NotImplementedError

    def dump(self, filename):
        """Dump the projection info to *filename*.
        """
        if os.path.exists(filename):
            LOG.debug("Projection already saved to %s", filename)
        else:
            LOG.info("Saving projection to %s", filename)
            np.savez(filename, **self.cache)


    def resample(self, data, cache_dir=False, **kwargs):
        """Resample the *data*, saving the projection info on disk if *precompute* evaluates to True.
        """
        self.precompute(cache_dir=cache_dir, **kwargs)
        return self.compute(data, **kwargs)

    def __call__(self, *args, **kwargs):
        """Shortcut for the :meth:`resample` method
        """
        self.resample(*args, **kwargs)

class KDTreeResampler(BaseResampler):
    """
    Resample using nearest neighbour.
    """

    caches = OrderedDict()

    def __init__(self, source_geo_def, target_geo_def):
        BaseResampler.__init__(self, source_geo_def, target_geo_def)

    @staticmethod
    def hash_area(area):
        """Get (and set) the hash for the *area*.
        """
        try:
            return area.kdtree_hash
        except AttributeError:
            LOG.debug("Computing kd-tree hash for area %s", area.name)
        try:
            area_hash = "".join((hashlib.sha1(json.dumps(area.proj_dict, sort_keys=True)).hexdigest(),
                                 hashlib.sha1(json.dumps(area.area_extent)).hexdigest(),
                                 hashlib.sha1(json.dumps(area.shape)).hexdigest()))
        except AttributeError:
            if not hasattr(area, "lons") or area.lons is None:
                lons, lats = area.get_lonlats()
            else:
                lons, lats = area.lons, area.lats

            try:
                mask = lons.mask
            except AttributeError:
                mask = "False"
            area_hash = "".join((hashlib.sha1(mask).hexdigest(),
                                 hashlib.sha1(lons).hexdigest(),
                                 hashlib.sha1(lats).hexdigest()))
        area.kdtree_hash = area_hash
        return area_hash


    def get_hash(self, **kwargs):
        """Get hash for the current resample with the given *kwargs*.
        """
        the_hash = "".join((self.hash_area(self.source_geo_def),
                            self.hash_area(self.target_geo_def),
                            hashlib.sha1(json.dumps(kwargs, sort_keys=True)).hexdigest()))
        return the_hash

    def precompute(self, radius_of_influence=10000, epsilon=0, reduce_data=True, nprocs=1, segments=None,
                   cache_dir=False, **kwargs):

        del kwargs

        kd_hash = self.get_hash(radius_of_influence=radius_of_influence, epsilon=epsilon)
        if isinstance(cache_dir, (str, unicode)):
            filename = os.path.join(cache_dir, hashlib.sha1(kd_hash).hexdigest() + ".npz")
        else:
            filename = os.path.join('.', hashlib.sha1(kd_hash).hexdigest() + ".npz")

        try:
            self.cache = self.caches[kd_hash]
            # trick to keep most used caches away from deletion
            del self.caches[kd_hash]
            self.caches[kd_hash] = self.cache

            if cache_dir:
                self.dump(filename)
            return self.cache
        except KeyError:
            if os.path.exists(filename):
                LOG.debug("Loading kd-tree parameters")
                self.cache = dict(np.load(filename))
                self.caches[kd_hash] = self.cache
                while len(self.caches) > CACHE_SIZE:
                    self.caches.popitem()
                if cache_dir:
                    self.dump(filename)
                return self.cache
            else:
                LOG.debug("Computing kd-tree parameters")

        valid_input_index, valid_output_index, index_array, distance_array = \
            get_neighbour_info(self.source_geo_def,
                               self.target_geo_def,
                               radius_of_influence,
                               neighbours=1,
                               epsilon=epsilon,
                               reduce_data=reduce_data,
                               nprocs=nprocs,
                               segments=segments)

        # it's important here not to modify the existing cache dictionnary.
        self.cache = {"valid_input_index": valid_input_index,
                      "valid_output_index": valid_output_index,
                      "index_array": index_array,
                      "distance_array": distance_array}

        self.caches[kd_hash] = self.cache
        while len(self.caches) > CACHE_SIZE:
            self.caches.popitem()

        if cache_dir:
            self.dump(filename)
        return self.cache

    def compute(self, data, weight_funcs=None, fill_value=None, with_uncert=False, **kwargs):

        del kwargs

        return get_sample_from_neighbour_info('nn',
                                              self.target_geo_def.shape,
                                              data,
                                              self.cache["valid_input_index"],
                                              self.cache["valid_output_index"],
                                              self.cache["index_array"],
                                              distance_array=self.cache["distance_array"],
                                              weight_funcs=weight_funcs,
                                              fill_value=fill_value,
                                              with_uncert=with_uncert)


RESAMPLERS = {"kd_tree": KDTreeResampler,
              "nearest": KDTreeResampler}

def resample(source_area, data, destination_area, resampler=KDTreeResampler, **kwargs):
    """Do the resampling
    """
    if isinstance(resampler, (str, unicode)):
        resampler_class = RESAMPLERS[resampler]
    else:
        resampler_class = resampler
    resampler = resampler_class(source_area, destination_area)
    return resampler.resample(data, **kwargs)
