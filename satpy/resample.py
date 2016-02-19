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
import six
from satpy import get_config, get_config_path, utils
try:
    import configparser
except ImportError:
    from six.moves import configparser

LOG = getLogger(__name__)

CACHE_SIZE = 10


def get_area_file():
    conf, successes = get_config("satpy.cfg")
    if conf is None or not successes:
        LOG.warning("Couldn't find the satpy.cfg file. Do you have one ? is it in $PPP_CONFIG_DIR ?")
        return None

    try:
        fn = os.path.join(conf.get("projector", "area_directory") or "",
                          conf.get("projector", "area_file"))
        return get_config_path(fn)
    except configparser.NoSectionError:
        LOG.warning("Couldn't find 'projector' section of 'satpy.cfg'")


def get_area_def(area_name):
    """Get the definition of *area_name* from file. The file is defined to use
    is to be placed in the $PPP_CONFIG_DIR directory, and its name is defined
    in satpy's configuration file.
    """
    from pyresample.utils import parse_area_file
    return parse_area_file(get_area_file(), area_name)[0]


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

    def resample(self, data, cache_dir=False, mask_area=True, **kwargs):
        """Resample the *data*, saving the projection info on disk if *precompute* evaluates to True.

        :param mask_area: Provide data mask to `precompute` method to mask invalid data values in geolocation.
        """
        if mask_area and hasattr(data, "mask"):
            kwargs.setdefault("mask", data.mask)
        cache_id = self.precompute(cache_dir=cache_dir, **kwargs)
        return self.compute(data, cache_id=cache_id, **kwargs)

    # FIXME: there should be only one obvious way to resample
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
            area_hash = "".join((hashlib.sha1(json.dumps(area.proj_dict, sort_keys=True).encode("utf-8")).hexdigest(),
                                 hashlib.sha1(json.dumps(area.area_extent).encode("utf-8")).hexdigest(),
                                 hashlib.sha1(json.dumps(area.shape).encode('utf-8')).hexdigest()))
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

    def get_hash(self, source_geo_def=None, target_geo_def=None, **kwargs):
        """Get hash for the current resample with the given *kwargs*.
        """
        if source_geo_def is None:
            source_geo_def = self.source_geo_def
        if target_geo_def is None:
            target_geo_def = self.target_geo_def

        the_hash = "".join((self.hash_area(source_geo_def),
                            self.hash_area(target_geo_def),
                            hashlib.sha1(json.dumps(kwargs, sort_keys=True).encode('utf-8')).hexdigest()))
        return the_hash

    def precompute(self, mask=None, radius_of_influence=10000, epsilon=0, reduce_data=True, nprocs=1, segments=None,
                   cache_dir=False, **kwargs):
        """Create a KDTree structure and store it for later use.

        Note: The `mask` keyword should be provided if geolocation may be valid where data points are invalid.
        This defaults to the `mask` attribute of the `data` numpy masked array passed to the `resample` method.
        """

        del kwargs

        source_geo_def = self.source_geo_def
        # the data may have additional masked pixels
        # let's compare them to see if we can use the same area
        # assume lons and lats mask are the same
        if mask is not None and np.any(((source_geo_def.lons.mask & mask) != mask)):
            LOG.debug("Copying source area to mask invalid dataset points")
            # copy the source area and use it for the rest of the calculations
            new_mask = source_geo_def.lons.mask | mask
            # use the same class as the original source area in case it's a subclass
            cls = self.source_geo_def.__class__
            # use the same data, but make a new mask (i.e. don't affect the original masked array)
            lons = np.ma.masked_array(self.source_geo_def.lons.data, new_mask)
            lats = np.ma.masked_array(self.source_geo_def.lats.data, new_mask)
            source_geo_def = cls(lons, lats, nprocs=source_geo_def.nprocs)
            # FIXME: Finalize how Area and GridDefinitions use the `name` attribute
            if hasattr(self.source_geo_def, "name"):
                source_geo_def.name = self.source_geo_def.name

        kd_hash = self.get_hash(source_geo_def=source_geo_def,
                                radius_of_influence=radius_of_influence,
                                epsilon=epsilon)
        if isinstance(cache_dir, (str, six.text_type)):
            filename = os.path.join(cache_dir, hashlib.sha1(kd_hash).hexdigest() + ".npz")
        else:
            filename = os.path.join('.', hashlib.sha1(kd_hash.encode("utf-8")).hexdigest() + ".npz")

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
                    self.caches.popitem(False)
                if cache_dir:
                    self.dump(filename)
                return self.cache
            else:
                LOG.debug("Computing kd-tree parameters")

        valid_input_index, valid_output_index, index_array, distance_array = \
            get_neighbour_info(source_geo_def,
                               self.target_geo_def,
                               radius_of_influence,
                               neighbours=1,
                               epsilon=epsilon,
                               reduce_data=reduce_data,
                               nprocs=nprocs,
                               segments=segments)

        # it's important here not to modify the existing cache dictionary.
        self.cache = {"valid_input_index": valid_input_index,
                      "valid_output_index": valid_output_index,
                      "index_array": index_array,
                      "distance_array": distance_array,
                      "source_geo_def": source_geo_def,
                     }

        self.caches[kd_hash] = self.cache
        while len(self.caches) > CACHE_SIZE:
            self.caches.popitem(False)

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
    if isinstance(resampler, (str, six.text_type)):
        resampler_class = RESAMPLERS[resampler]
    else:
        resampler_class = resampler
    resampler = resampler_class(source_area, destination_area)
    return resampler.resample(data, **kwargs)
