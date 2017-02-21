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
import hashlib
import json
import os
from copy import deepcopy
from logging import getLogger

import numpy as np
import six

from pyresample.ewa import fornav, ll2cr
from pyresample.kd_tree import (get_neighbour_info,
                                get_sample_from_neighbour_info)
from satpy.config import get_config, get_config_path
from satpy.dataset import Dataset

try:
    import configparser
except ImportError:
    from six.moves import configparser

LOG = getLogger(__name__)

CACHE_SIZE = 10


def get_area_file():
    conf, successes = get_config("satpy.cfg")
    if conf is None or not successes:
        LOG.warning(
            "Couldn't find the satpy.cfg file. Do you have one ? is it in $PPP_CONFIG_DIR ?")
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

    @staticmethod
    def hash_area(area):
        """Get (and set) the hash for the *area*.
        """
        try:
            return area.kdtree_hash
        except AttributeError:
            LOG.debug("Computing kd-tree hash for area %s",
                      getattr(area, 'name', 'swath'))
        try:
            area_hash = "".join((hashlib.sha1(json.dumps(area.proj_dict, sort_keys=True).encode("utf-8")).hexdigest(),
                                 hashlib.sha1(json.dumps(area.area_extent).encode(
                                     "utf-8")).hexdigest(),
                                 hashlib.sha1(json.dumps(area.shape).encode('utf-8')).hexdigest()))
        except AttributeError:
            if not hasattr(area, "lons") or area.lons is None:
                lons, lats = area.get_lonlats()
            else:
                lons, lats = area.lons, area.lats

            try:
                mask_hash = hashlib.sha1(area.mask).hexdigest()
            except AttributeError:
                mask_hash = "False"
            area_hash = "".join((mask_hash,
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

    def precompute(self, **kwargs):
        """Do the precomputation.

        This is an optional step if the subclass wants to implement more
        complex features like caching or can share some calculations
        between multiple datasets to be processed.
        """
        return None

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
        if np.any(mask):
            # copy the source area and use it for the rest of the calculations
            LOG.debug("Copying source area to mask invalid dataset points")
            source_geo_def = deepcopy(self.source_geo_def)
            lons, lats = source_geo_def.get_lonlats()
            if np.ndim(mask) == 3:
                # FIXME: we should treat 3d arrays (composites) layer by layer!
                mask = np.sum(mask, axis=2)
                # FIXME: pyresample doesn't seem to like this
                #lons = np.tile(lons, (1, 1, mask.shape[2]))
                #lats = np.tile(lats, (1, 1, mask.shape[2]))

            # use the same data, but make a new mask (i.e. don't affect the original masked array)
            # the ma.array function combines the undelying mask with the new
            # one (OR)
            source_geo_def.lons = np.ma.array(lons, mask=mask)
            source_geo_def.lats = np.ma.array(lats, mask=mask)

        kd_hash = self.get_hash(source_geo_def=source_geo_def,
                                radius_of_influence=radius_of_influence,
                                epsilon=epsilon)
        if isinstance(cache_dir, (str, six.text_type)):
            filename = os.path.join(
                cache_dir, hashlib.sha1(kd_hash).hexdigest() + ".npz")
        else:
            filename = os.path.join('.', hashlib.sha1(
                kd_hash.encode("utf-8")).hexdigest() + ".npz")

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
                                              distance_array=self.cache[
                                                  "distance_array"],
                                              weight_funcs=weight_funcs,
                                              fill_value=fill_value,
                                              with_uncert=with_uncert)


class EWAResampler(BaseResampler):
    caches = OrderedDict()

    def __init__(self, source_geo_def, target_geo_def, swath_usage=0, grid_coverage=0, **kwargs):
        """

        :param source_geo_def: See `BaseResampler` for details
        :param target_geo_def: See `BaseResampler` for details
        :param swath_usage: minimum ratio of number of input pixels to number of pixels used in output
        :param grid_coverage: minimum ratio of number of output grid pixels covered with swath pixels
        """
        self.swath_usage = swath_usage
        self.grid_coverage = grid_coverage
        super(EWAResampler, self).__init__(
            source_geo_def, target_geo_def, **kwargs)

    def precompute(self, mask=None,
                   # nprocs=1,
                   cache_dir=False,
                   **kwargs):
        """Generate row and column arrays and store it for later use.

        Note: The `mask` keyword should be provided if geolocation may be valid where data points are invalid.
        This defaults to the `mask` attribute of the `data` numpy masked array passed to the `resample` method.
        """

        del kwargs

        source_geo_def = self.source_geo_def

        kd_hash = self.get_hash(source_geo_def=source_geo_def)
        if isinstance(cache_dir, (str, six.text_type)):
            filename = os.path.join(
                cache_dir, hashlib.sha1(kd_hash).hexdigest() + ".npz")
        else:
            filename = os.path.join('.', hashlib.sha1(
                kd_hash.encode("utf-8")).hexdigest() + ".npz")

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
                LOG.debug("Computing ll2cr parameters")

        lons, lats = source_geo_def.get_lonlats()
        fill_in = np.nan
        grid_name = getattr(self.target_geo_def, "name", "N/A")
        p = self.target_geo_def.proj4_string
        cw = self.target_geo_def.pixel_size_x
        ch = -abs(self.target_geo_def.pixel_size_y)
        w = self.target_geo_def.x_size
        h = self.target_geo_def.y_size
        ox = self.target_geo_def.area_extent[0]
        oy = self.target_geo_def.area_extent[3]

        is_static = True  # SatPy/PyResample don't support dynamic grids out of the box yet
        if is_static:
            # we are remapping to a static unchanging grid/area with all of
            # its parameters specified
            # inplace operation so lon_arr and lat_arr are written to
            swath_points_in_grid = ll2cr(source_geo_def, self.target_geo_def)
        else:
            raise NotImplementedError(
                "Dynamic ll2cr is not supported by satpy yet")
            swath_points_in_grid = ll2cr(source_geo_def, self.target_geo_def)
            swath_points_in_grid, lon_orig, lat_orig, origin_x, origin_y, width, height = results

            # edit the grid information because now we know what it is
            # FIXME: This should be less magical to the user, maybe a separate
            # step for this operation
            self.target_geo_def.origin_x = origin_x
            self.target_geo_def.origin_y = origin_y
            self.target_geo_def.width = width
            self.target_geo_def.height = height

        # Determine if enough of the input swath was used
        fraction_in = swath_points_in_grid / float(lon_arr.size)
        swath_used = fraction_in > self.swath_usage
        if not swath_used:
            LOG.info("Data does not fit in grid %s because it only %f%% of the swath is used" %
                     (grid_name, fraction_in * 100))
            raise RuntimeError("Data does not fit in grid %s" % (grid_name,))
        else:
            LOG.debug("Data fits in grid %s and uses %f%% of the swath",
                      grid_name, fraction_in * 100)

        # it's important here not to modify the existing cache dictionary.
        self.cache = {
            "source_geo_def": source_geo_def,
            "rows": lat_arr,
            "cols": lon_arr,
        }

        self.caches[kd_hash] = self.cache
        while len(self.caches) > CACHE_SIZE:
            self.caches.popitem(False)

        if cache_dir:
            # XXX: Look in to doing memmap-able files instead
            # `arr.tofile(filename)`
            self.dump(filename)
        return self.cache

    def compute(self, data,
                weight_count=10000, weight_min=0.01, weight_distance_max=1.0,
                weight_sum_min=-1.0, maximum_weight_mode=False,
                **kwargs):
        rows = self.cache["rows"]
        cols = self.cache["cols"]
        # if the data is scan based then check its metadata or the passed kwargs
        # otherwise assume the entire input swath is one large "scanline"
        rows_per_scan = getattr(data, "info", kwargs).get(
            "rows_per_scan", data.shape[0])
        num_valid_points, out_arrs = fornav(cols, rows, self.target_area_def, data, rows_per_scan=rows_per_scan,
                                            weight_count=weight_count, weight_min=weight_min,
                                            weight_distance_max=weight_distance_max, weight_sum_min=weight_sum_min,
                                            maximum_weight_mode=maximum_weight_mode, **kwargs)
        num_valid_points = num_valid_points[0]
        out_arr = out_arrs[0]

        grid_covered_ratio = num_valid_points / float(out_arr.size)
        grid_covered = grid_covered_ratio > self.grid_coverage
        if not grid_covered:
            msg = "EWA resampling only found %f%% of the grid covered (need %f%%)" % (grid_covered_ratio * 100,
                                                                                      self.grid_coverage * 100)
            raise RuntimeError(msg)
        LOG.debug("EWA resampling found %f%% of the grid covered" %
                  (grid_covered_ratio * 100))
        info = getattr(data, "info", {})
        info["mask"] = np.isnan(out_arr) if np.issubdtype(
            data.dtype, np.floating) else out_arr == fill
        return Dataset(out_arr, **info)


RESAMPLERS = {"kd_tree": KDTreeResampler,
              "nearest": KDTreeResampler,
              "ewa": EWAResampler,
              }


def resample(source_area, data, destination_area, resampler=KDTreeResampler, **kwargs):
    """Do the resampling
    """
    if isinstance(resampler, (str, six.text_type)):
        resampler_class = RESAMPLERS[resampler]
    else:
        resampler_class = resampler
    resampler = resampler_class(source_area, destination_area)
    return resampler.resample(data, **kwargs)
