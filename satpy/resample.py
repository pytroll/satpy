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

import hashlib
import json
import os
from logging import getLogger
from collections import OrderedDict

import numpy as np
import xarray as xr
import dask.array as da
import xarray.ufuncs as xu
import six

from pyresample.bilinear import get_bil_info, get_sample_from_bil_info
from pyresample.ewa import fornav, ll2cr
from pyresample.geometry import SwathDefinition, AreaDefinition
from pyresample.kd_tree import XArrayResamplerNN
from satpy import CHUNK_SIZE
from satpy.config import config_search_paths, get_config_path

LOG = getLogger(__name__)

CACHE_SIZE = 10


def get_area_file():
    """Find area file(s) to use.

    The files are to be named `areas.yaml` or `areas.def`.
    """
    paths = config_search_paths('areas.yaml')
    if paths:
        return paths
    else:
        return get_config_path('areas.def')


def get_area_def(area_name):
    """Get the definition of *area_name* from file.

    The file is defined to use is to be placed in the $PPP_CONFIG_DIR
    directory, and its name is defined in satpy's configuration file.
    """
    from pyresample.utils import parse_area_file
    return parse_area_file(get_area_file(), area_name)[0]


def get_frozen_area(to_freeze, ref):
    """Freeze the *to_freeze* area according to *ref* if applicable, otherwise
    return *to_freeze* as an area definition instance.
    """
    if isinstance(to_freeze, (str, six.text_type)):
        to_freeze = get_area_def(to_freeze)

    try:
        return to_freeze.freeze(ref)
    except AttributeError:
        return to_freeze


class BaseResampler(object):

    """
    The base resampler class. Abstract.
    """

    caches = OrderedDict()

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
        return str(area.__hash__())

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
        """Resample data.

        If the resampler supports precomputing then that information can be
        cached on disk (if the `precompute` method returns `True`).

        Args:
            data (xarray.DataArray): Data to be resampled
            cache_dir (bool): directory to cache precomputed results
                              (default False, optional)
            mask_area (bool): Mask geolocation data where data values are
                              invalid. This should be used when data values
                              may affect what neighbors are considered valid.

        """
        if mask_area:
            flat_dims = [dim for dim in data.dims if dim not in ['x', 'y']]
            # xarray <= 0.10.1 computes dask arrays during isnull
            kwargs['mask'] = data.isnull().all(dim=flat_dims)
        cache_id = self.precompute(cache_dir=cache_dir, **kwargs)
        return self.compute(data, cache_id=cache_id, **kwargs)

    # FIXME: there should be only one obvious way to resample
    def __call__(self, *args, **kwargs):
        """Shortcut for the :meth:`resample` method
        """
        self.resample(*args, **kwargs)

    def _create_cache_filename(self, cache_dir, hash_str):
        """Create filename for the cached resampling parameters"""
        if isinstance(cache_dir, (str, six.text_type)):
            filename = os.path.join(
                cache_dir, hashlib.sha1(hash_str).hexdigest() + ".npz")
        else:
            filename = os.path.join('.', hashlib.sha1(
                hash_str.encode("utf-8")).hexdigest() + ".npz")

        return filename

    def _read_params_from_cache(self, cache_dir, hash_str, filename):
        """Read resampling parameters from cache"""
        self.cache = self.caches.pop(hash_str, None)
        if self.cache is not None and cache_dir:
            self.dump(filename)
        elif os.path.exists(filename):
            self.cache = dict(np.load(filename))
            self.caches[hash_str] = self.cache

    def _update_caches(self, hash_str, cache_dir, filename):
        """Update caches and dump new resampling parameters to disk"""
        while len(self.caches.keys()) > 2:
            self.caches.popitem()
        self.caches[hash_str] = self.cache
        if cache_dir:
            # XXX: Look in to doing memmap-able files instead
            # `arr.tofile(filename)`
            self.dump(filename)


class KDTreeResampler(BaseResampler):

    """
    Resample using nearest neighbour.
    """

    def __init__(self, source_geo_def, target_geo_def):
        super(KDTreeResampler, self).__init__(source_geo_def, target_geo_def)
        self.resampler = None

    def precompute(
            self, mask=None, radius_of_influence=10000, epsilon=0, reduce_data=True, nprocs=1, segments=None,
            cache_dir=False, **kwargs):
        """Create a KDTree structure and store it for later use.

        Note: The `mask` keyword should be provided if geolocation may be valid
        where data points are invalid. This defaults to the `mask` attribute of
        the `data` numpy masked array passed to the `resample` method.
        """

        del kwargs
        source_geo_def = mask_source_lonlats(self.source_geo_def, mask)

        kd_hash = self.get_hash(source_geo_def=source_geo_def,
                                radius_of_influence=radius_of_influence,
                                epsilon=epsilon)

        filename = self._create_cache_filename(cache_dir, kd_hash)
        self._read_params_from_cache(cache_dir, kd_hash, filename)

        if self.resampler is None:
            if self.cache is not None:
                LOG.debug("Loaded kd-tree parameters")
                return self.cache

            LOG.debug("Computing kd-tree parameters")

            self.resampler = XArrayResamplerNN(source_geo_def,
                                               self.target_geo_def,
                                               radius_of_influence,
                                               neighbours=1,
                                               epsilon=epsilon,
                                               reduce_data=reduce_data,
                                               nprocs=nprocs,
                                               segments=segments)

            valid_input_index, valid_output_index, index_array, distance_array = \
                self.resampler.get_neighbour_info()
            # reference call to pristine pyresample
            # vii, voi, ia, da = get_neighbour_info(source_geo_def,
            #                                       self.target_geo_def,
            #                                       radius_of_influence,
            #                                       neighbours=1,
            #                                       epsilon=epsilon,
            #                                       reduce_data=reduce_data,
            #                                       nprocs=nprocs,
            #                                       segments=segments)

            # it's important here not to modify the existing cache dictionary.
            if cache_dir:
                self.cache = {"valid_input_index": valid_input_index,
                              "valid_output_index": valid_output_index,
                              "index_array": index_array,
                              "distance_array": distance_array,
                              "source_geo_def": source_geo_def,
                              }

                self._update_caches(kd_hash, cache_dir, filename)

                return self.cache
            else:
                del valid_input_index, valid_output_index, index_array, distance_array

    def compute(self, data, weight_funcs=None, fill_value=None,
                with_uncert=False, **kwargs):
        del kwargs
        LOG.debug("Resampling " + str(data.name))
        if fill_value is None:
            fill_value = data.attrs.get('_FillValue')
        res = self.resampler.get_sample_from_neighbour_info(data, fill_value)
        return res


class EWAResampler(BaseResampler):

    def precompute(self, mask=None, cache_dir=False, swath_usage=0,
                   **kwargs):
        """Generate row and column arrays and store it for later use.

        :param swath_usage: minimum ratio of number of input pixels to
                            number of pixels used in output

        Note: The `mask` keyword should be provided if geolocation may be
              valid where data points are invalid. This defaults to the
              `mask` attribute of the `data` numpy masked array passed to
              the `resample` method.
        """

        del kwargs

        source_geo_def = self.source_geo_def

        ewa_hash = self.get_hash(source_geo_def=source_geo_def)

        filename = self._create_cache_filename(cache_dir, ewa_hash)
        self._read_params_from_cache(cache_dir, ewa_hash, filename)

        if self.cache is not None:
            LOG.debug("Loaded ll2cr parameters")
            return self.cache
        else:
            LOG.debug("Computing ll2cr parameters")

        lons, lats = source_geo_def.get_lonlats()
        grid_name = getattr(self.target_geo_def, "name", "N/A")

        # SatPy/PyResample don't support dynamic grids out of the box yet
        is_static = True
        if is_static:
            # we are remapping to a static unchanging grid/area with all of
            # its parameters specified
            # inplace operation so lon_arr and lat_arr are written to
            swath_points_in_grid, cols, rows = ll2cr(source_geo_def,
                                                     self.target_geo_def)
        else:
            raise NotImplementedError(
                "Dynamic ll2cr is not supported by satpy yet")

        # Determine if enough of the input swath was used
        fraction_in = swath_points_in_grid / float(lons.size)
        swath_used = fraction_in > swath_usage
        if not swath_used:
            LOG.info("Data does not fit in grid %s because it only %f%% of "
                     "the swath is used" %
                     (grid_name, fraction_in * 100))
            raise RuntimeError("Data does not fit in grid %s" % (grid_name,))
        else:
            LOG.debug("Data fits in grid %s and uses %f%% of the swath",
                      grid_name, fraction_in * 100)

        # Can't save masked arrays to npz, so remove the mask
        if hasattr(rows, 'mask'):
            rows = rows.data
            cols = cols.data

        # it's important here not to modify the existing cache dictionary.
        self.cache = {
            "source_geo_def": source_geo_def,
            "rows": rows,
            "cols": cols,
        }

        self._update_caches(ewa_hash, cache_dir, filename)

        return self.cache

    def compute(self, data, fill_value=0, weight_count=10000, weight_min=0.01,
                weight_distance_max=1.0, weight_sum_min=-1.0,
                maximum_weight_mode=False, grid_coverage=0, **kwargs):
        """Resample the data according to the precomputed X/Y coordinates.

        :param grid_coverage: minimum ratio of number of output grid pixels
                              covered with swath pixels

        """
        rows = self.cache["rows"]
        cols = self.cache["cols"]

        # if the data is scan based then check its metadata or the passed
        # kwargs otherwise assume the entire input swath is one large
        # "scanline"
        rows_per_scan = getattr(data, "info", kwargs).get(
            "rows_per_scan", data.shape[0])
        if hasattr(data, 'mask'):
            mask = data.mask
            data = data.data
            data[mask] = np.nan

        if data.ndim >= 3:
            data_in = tuple(data[..., i] for i in range(data.shape[-1]))
        else:
            data_in = data

        num_valid_points, res = fornav(cols, rows, self.target_geo_def,
                                       data_in,
                                       rows_per_scan=rows_per_scan,
                                       weight_count=weight_count,
                                       weight_min=weight_min,
                                       weight_distance_max=weight_distance_max,
                                       weight_sum_min=weight_sum_min,
                                       maximum_weight_mode=maximum_weight_mode)

        if data.ndim >= 3:
            # convert 'res' from tuple of arrays to one array
            res = np.dstack(res)
            num_valid_points = sum(num_valid_points)

        grid_covered_ratio = num_valid_points / float(res.size)
        grid_covered = grid_covered_ratio > grid_coverage
        if not grid_covered:
            msg = "EWA resampling only found %f%% of the grid covered "
            "(need %f%%)" % (grid_covered_ratio * 100,
                             grid_coverage * 100)
            raise RuntimeError(msg)
        LOG.debug("EWA resampling found %f%% of the grid covered" %
                  (grid_covered_ratio * 100))

        return np.ma.masked_invalid(res)


class BilinearResampler(BaseResampler):

    """Resample using bilinear."""

    def precompute(self, mask=None, radius_of_influence=50000,
                   reduce_data=True, nprocs=1, segments=None,
                   cache_dir=False, **kwargs):
        """Create bilinear coefficients and store them for later use.

        Note: The `mask` keyword should be provided if geolocation may be valid
        where data points are invalid. This defaults to the `mask` attribute of
        the `data` numpy masked array passed to the `resample` method.
        """

        del kwargs

        source_geo_def = mask_source_lonlats(self.source_geo_def, mask)

        bil_hash = self.get_hash(source_geo_def=source_geo_def,
                                 radius_of_influence=radius_of_influence,
                                 mode="bilinear")

        filename = self._create_cache_filename(cache_dir, bil_hash)
        self._read_params_from_cache(cache_dir, bil_hash, filename)

        if self.cache is not None:
            LOG.debug("Loaded bilinear parameters")
            return self.cache
        else:
            LOG.debug("Computing bilinear parameters")

        bilinear_t, bilinear_s, input_idxs, idx_arr = get_bil_info(source_geo_def, self.target_geo_def,
                                                                   radius_of_influence, neighbours=32,
                                                                   nprocs=nprocs, masked=False)
        self.cache = {'bilinear_s': bilinear_s,
                      'bilinear_t': bilinear_t,
                      'input_idxs': input_idxs,
                      'idx_arr': idx_arr}

        self._update_caches(bil_hash, cache_dir, filename)

        return self.cache

    def compute(self, data, fill_value=None, **kwargs):
        """Resample the given data using bilinear interpolation"""
        del kwargs

        target_shape = self.target_geo_def.shape
        if data.ndim == 3:
            output_shape = list(target_shape)
            output_shape.append(data.shape[-1])
            res = np.zeros(output_shape, dtype=data.dtype)
            for i in range(data.shape[-1]):
                res[:, :, i] = get_sample_from_bil_info(data[:, :, i].ravel(),
                                                        self.cache[
                                                            'bilinear_t'],
                                                        self.cache[
                                                            'bilinear_s'],
                                                        self.cache[
                                                            'input_idxs'],
                                                        self.cache['idx_arr'],
                                                        output_shape=target_shape)

        else:
            res = get_sample_from_bil_info(data.ravel(),
                                           self.cache['bilinear_t'],
                                           self.cache['bilinear_s'],
                                           self.cache['input_idxs'],
                                           self.cache['idx_arr'],
                                           output_shape=target_shape)
        res = np.ma.masked_invalid(res)

        return res


class NativeResampler(BaseResampler):

    """Expand or reduce input datasets to be the same shape."""

    def resample(self, data, cache_dir=False, mask_area=False, **kwargs):
        # use 'mask_area' with a default of False. It wouldn't do anything.
        return super(NativeResampler, self).resample(data,
                                                     cache_dir=cache_dir,
                                                     mask_area=mask_area,
                                                     **kwargs)

    @staticmethod
    def aggregate(d, y_size, x_size):
        """Average every 4 elements (2x2) in a 2D array"""
        def _mean(data):
            rows, cols = data.shape
            new_shape = (int(rows / y_size), y_size,
                         int(cols / x_size), x_size)
            data_mean = np.ma.mean(data.reshape(new_shape), axis=(1, 3))
            return data_mean

        if d.ndim != 2:
            # we can't guarantee what blocks we are getting and how
            # it should be reshaped to do the averaging.
            raise ValueError("Can't aggregrate (reduce) data arrays with "
                             "more than 2 dimensions.")
        if not (x_size.is_integer() and y_size.is_integer()):
            raise ValueError("Aggregation factors are not integers")
        for agg_size, chunks in zip([y_size, x_size], d.chunks):
            for chunk_size in chunks:
                if chunk_size % agg_size != 0:
                    raise ValueError("Aggregation requires arrays with "
                                     "shapes and chunks divisible by the "
                                     "factor")

        new_chunks = (tuple(int(x / y_size) for x in d.chunks[0]),
                      tuple(int(x / x_size) for x in d.chunks[1]))
        return da.core.map_blocks(_mean, d, dtype=d.dtype, chunks=new_chunks)

    @classmethod
    def expand_reduce(cls, d_arr, repeats):
        if not isinstance(d_arr, da.Array):
            d_arr = da.from_array(d_arr, chunks=CHUNK_SIZE)
        if all(x == 1 for x in repeats.values()):
            return d_arr
        elif all(x >= 1 for x in repeats.values()):
            # rechunk so new chunks are the same size as old chunks
            c_size = max(x[0] for x in d_arr.chunks)

            def _calc_chunks(c, c_size):
                whole_chunks = [c_size] * int(sum(c) // c_size)
                remaining = sum(c) - sum(whole_chunks)
                if remaining:
                    whole_chunks += [remaining]
                return tuple(whole_chunks)
            new_chunks = [_calc_chunks(x, int(c_size // repeats[axis]))
                          for axis, x in enumerate(d_arr.chunks)]
            d_arr = d_arr.rechunk(new_chunks)

            for axis, factor in repeats.items():
                if not factor.is_integer():
                    raise ValueError("Expand factor must be a whole number")
                d_arr = da.repeat(d_arr, int(factor), axis=axis)
            return d_arr
        elif all(x <= 1 for x in repeats.values()):
            # reduce
            y_size = 1. / repeats[0]
            x_size = 1. / repeats[1]
            return cls.aggregate(d_arr, y_size, x_size)
        else:
            raise ValueError("Must either expand or reduce in both "
                             "directions")

    def compute(self, data, expand=True, **kwargs):
        if isinstance(self.target_geo_def, (list, tuple)):
            # find the highest/lowest area among the provided
            test_func = max if expand else min
            target_geo_def = test_func(self.target_geo_def,
                                       key=lambda x: x.shape)
        else:
            target_geo_def = self.target_geo_def

        # convert xarray backed with numpy array to dask array
        if 'x' not in data.dims or 'y' not in data.dims:
            if data.ndim not in [2, 3]:
                raise ValueError("Can only handle 2D or 3D arrays without "
                                 "dimensions.")
            # assume rows is the second to last axis
            y_axis = data.ndim - 2
            x_axis = data.ndim - 1
        else:
            y_axis = data.dims.index('y')
            x_axis = data.dims.index('x')

        out_shape = target_geo_def.shape
        in_shape = data.shape
        y_repeats = out_shape[0] / float(in_shape[y_axis])
        x_repeats = out_shape[1] / float(in_shape[x_axis])
        repeats = {
            y_axis: y_repeats,
            x_axis: x_repeats,
        }

        d_arr = self.expand_reduce(data.data, repeats)

        coords = {}
        # Update coords if we can
        if 'y' in data.coords or 'x' in data.coords and \
                isinstance(target_geo_def, AreaDefinition):
            coord_chunks = (d_arr.chunks[y_axis], d_arr.chunks[x_axis])
            x_coord, y_coord = target_geo_def.get_proj_vectors_dask(
                chunks=coord_chunks)
            if 'y' in data.coords:
                coords['y'] = y_coord
            if 'x' in data.coords:
                coords['x'] = x_coord
        for dim in data.dims:
            if dim not in ['y', 'x'] and dim in data.coords:
                coords[dim] = data.coords[dim]

        return xr.DataArray(d_arr,
                            dims=data.dims,
                            coords=coords or None)


RESAMPLERS = {"kd_tree": KDTreeResampler,
              "nearest": KDTreeResampler,
              "ewa": EWAResampler,
              "bilinear": BilinearResampler,
              "native": NativeResampler,
              }


def prepare_resampler(source_area, destination_area, resampler=None):
    """Instanciate and return a resampler."""
    if resampler is None:
        LOG.info("Using default KDTree resampler")
        resampler = 'kd_tree'

    if isinstance(resampler, BaseResampler):
        raise ValueError("Trying to create a resampler when one already "
                         "exists.")
    elif isinstance(resampler, str):
        resampler_class = RESAMPLERS[resampler]
    else:
        resampler_class = resampler

    return resampler_class(source_area, destination_area)


def resample(source_area, data, destination_area,
             resampler=None, **kwargs):
    """Do the resampling."""
    if 'resampler_class' in kwargs:
        import warnings
        warnings.warn("'resampler_class' is deprecated, use 'resampler'",
                      DeprecationWarning)
        resampler = kwargs.pop('resampler_class')

    if not isinstance(resampler, BaseResampler):
        resampler_instance = prepare_resampler(source_area,
                                               destination_area,
                                               resampler)
    else:
        resampler_instance = resampler

    if isinstance(data, list):
        res = [resampler_instance.resample(ds, **kwargs) for ds in data]
    else:
        res = resampler_instance.resample(data, **kwargs)

    return res


def resample_dataset(dataset, destination_area, **kwargs):
    """Resample the current projectable and return the resampled one.

    Args:
        destination_area: The destination onto which to project the data,
          either a full blown area definition or a string corresponding to
          the name of the area as defined in the area file.
        **kwargs: The extra parameters to pass to the resampling functions.

    Returns:
        A resampled projectable, with updated .attrs["area"] field.

    """
    # call the projection stuff here
    try:
        source_area = dataset.attrs["area"]
    except KeyError:
        LOG.info("Cannot reproject dataset %s, missing area info",
                 dataset.attrs['name'])

        return dataset

    new_data = resample(source_area, dataset, destination_area, **kwargs)
    new_data.attrs = dataset.attrs.copy()
    new_data.attrs['area'] = destination_area

    return new_data


def mask_source_lonlats(source_def, mask):
    """Mask source longitudes and latitudes to match data mask."""
    source_geo_def = source_def

    # the data may have additional masked pixels
    # let's compare them to see if we can use the same area
    # assume lons and lats mask are the same
    if mask is not None and mask is not False and isinstance(source_geo_def, SwathDefinition):
        if np.issubsctype(mask.dtype, np.bool):
            # copy the source area and use it for the rest of the calculations
            LOG.debug("Copying source area to mask invalid dataset points")
            if mask.ndim != source_geo_def.lons.ndim:
                raise ValueError("Can't mask area, mask has different number "
                                 "of dimensions.")

            return SwathDefinition(source_geo_def.lons.where(~mask),
                                   source_geo_def.lats.where(~mask))
        else:
            return SwathDefinition(source_geo_def.lons.where(~xu.isnan(mask)),
                                   source_geo_def.lats.where(~xu.isnan(mask)))

    return source_geo_def
