#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Resampling in Satpy.

Satpy provides multiple resampling algorithms for resampling geolocated
data to uniform projected grids. The easiest way to perform resampling in
Satpy is through the :class:`~satpy.scene.Scene` object's
:meth:`~satpy.scene.Scene.resample` method. Additional utility functions are
also available to assist in resampling data. Below is more information on
resampling with Satpy as well as links to the relevant API documentation for
available keyword arguments.

Resampling algorithms
---------------------

.. csv-table:: Available Resampling Algorithms
    :header-rows: 1
    :align: center

    "Resampler", "Description", "Related"
    "nearest", "Nearest Neighbor", :class:`~satpy.resample.KDTreeResampler`
    "ewa", "Elliptical Weighted Averaging", :class:`~pyresample.ewa.DaskEWAResampler`
    "ewa_legacy", "Elliptical Weighted Averaging (Legacy)", :class:`~pyresample.ewa.LegacyDaskEWAResampler`
    "native", "Native", :class:`~satpy.resample.NativeResampler`
    "bilinear", "Bilinear", :class:`~satpy.resample.BilinearResampler`
    "bucket_avg", "Average Bucket Resampling", :class:`~satpy.resample.BucketAvg`
    "bucket_sum", "Sum Bucket Resampling", :class:`~satpy.resample.BucketSum`
    "bucket_count", "Count Bucket Resampling", :class:`~satpy.resample.BucketCount`
    "bucket_fraction", "Fraction Bucket Resampling", :class:`~satpy.resample.BucketFraction`
    "gradient_search", "Gradient Search Resampling", :meth:`~pyresample.gradient.create_gradient_search_resampler`

The resampling algorithm used can be specified with the ``resampler`` keyword
argument and defaults to ``nearest``:

.. code-block:: python

    >>> scn = Scene(...)
    >>> euro_scn = scn.resample('euro4', resampler='nearest')

.. warning::

    Some resampling algorithms expect certain forms of data. For example, the
    EWA resampling expects polar-orbiting swath data and prefers if the data
    can be broken in to "scan lines". See the API documentation for a specific
    algorithm for more information.

Resampling for comparison and composites
----------------------------------------

While all the resamplers can be used to put datasets of different resolutions
on to a common area, the 'native' resampler is designed to match datasets to
one resolution in the dataset's original projection. This is extremely useful
when generating composites between bands of different resolutions.

.. code-block:: python

    >>> new_scn = scn.resample(resampler='native')

By default this resamples to the
:meth:`highest resolution area <satpy.scene.Scene.finest_area>` (smallest footprint per
pixel) shared between the loaded datasets. You can easily specify the lowest
resolution area:

.. code-block:: python

    >>> new_scn = scn.resample(scn.coarsest_area(), resampler='native')

Providing an area that is neither the minimum or maximum resolution area
may work, but behavior is currently undefined.

Caching for geostationary data
------------------------------

Satpy will do its best to reuse calculations performed to resample datasets,
but it can only do this for the current processing and will lose this
information when the process/script ends. Some resampling algorithms, like
``nearest`` and ``bilinear``, can benefit by caching intermediate data on disk in the directory
specified by `cache_dir` and using it next time. This is most beneficial with
geostationary satellite data where the locations of the source data and the
target pixels don't change over time.

    >>> new_scn = scn.resample('euro4', cache_dir='/path/to/cache_dir')

See the documentation for specific algorithms to see availability and
limitations of caching for that algorithm.

Create custom area definition
-----------------------------

See :class:`pyresample.geometry.AreaDefinition` for information on creating
areas that can be passed to the resample method::

    >>> from pyresample.geometry import AreaDefinition
    >>> my_area = AreaDefinition(...)
    >>> local_scene = scn.resample(my_area)

Resize area definition in pixels
--------------------------------

Sometimes you may want to create a small image with fixed size in pixels.
For example, to create an image of (y, x) pixels :

    >>> small_scn = scn.resample(scn.finest_area().copy(height=y, width=x), resampler="nearest")


.. warning::

    Be aware that resizing with native resampling (``resampler="native"``) only
    works if the new size is an integer factor of the original input size. For example,
    multiplying the size by 2 or dividing the size by 2. Multiplying by 1.5 would
    not be allowed.


Create dynamic area definition
------------------------------

See :class:`pyresample.geometry.DynamicAreaDefinition` for more information.

Examples coming soon...

Store area definitions
----------------------

Area definitions can be saved to a custom YAML file (see
`pyresample's writing to disk <http://pyresample.readthedocs.io/en/stable/geometry_utils.html#writing-to-disk>`_)
and loaded using pyresample's utility methods
(`pyresample's loading from disk <http://pyresample.readthedocs.io/en/stable/geometry_utils.html#loading-from-disk>`_)::

    >>> from pyresample import load_area
    >>> my_area = load_area('my_areas.yaml', 'my_area')

Or using :func:`satpy.resample.get_area_def`, which will search through all
``areas.yaml`` files in your ``SATPY_CONFIG_PATH``::

    >>> from satpy.resample import get_area_def
    >>> area_eurol = get_area_def("eurol")

For examples of area definitions, see the file ``etc/areas.yaml`` that is
included with Satpy and where all the area definitions shipped with Satpy are
defined. The section below gives an overview of these area definitions.

Area definitions included in Satpy
----------------------------------

.. include:: /area_def_list.rst

"""
import hashlib
import json
import os
import warnings
from logging import getLogger
from math import lcm  # type: ignore
from weakref import WeakValueDictionary

import dask.array as da
import numpy as np
import xarray as xr
import zarr
from pyresample.ewa import DaskEWAResampler, LegacyDaskEWAResampler
from pyresample.geometry import SwathDefinition
from pyresample.gradient import create_gradient_search_resampler
from pyresample.resampler import BaseResampler as PRBaseResampler

from satpy._config import config_search_paths, get_config_path
from satpy.utils import PerformanceWarning, get_legacy_chunk_size

LOG = getLogger(__name__)

CHUNK_SIZE = get_legacy_chunk_size()
CACHE_SIZE = 10
NN_COORDINATES = {"valid_input_index": ("y1", "x1"),
                  "valid_output_index": ("y2", "x2"),
                  "index_array": ("y2", "x2", "z2")}
BIL_COORDINATES = {"bilinear_s": ("x1", ),
                   "bilinear_t": ("x1", ),
                   "slices_x": ("x1", "n"),
                   "slices_y": ("x1", "n"),
                   "mask_slices": ("x1", "n"),
                   "out_coords_x": ("x2", ),
                   "out_coords_y": ("y2", )}

resamplers_cache: "WeakValueDictionary[tuple, object]" = WeakValueDictionary()


def hash_dict(the_dict, the_hash=None):
    """Calculate a hash for a dictionary."""
    if the_hash is None:
        the_hash = hashlib.sha1()  # nosec
    the_hash.update(json.dumps(the_dict, sort_keys=True).encode("utf-8"))
    return the_hash


def get_area_file():
    """Find area file(s) to use.

    The files are to be named `areas.yaml` or `areas.def`.
    """
    paths = config_search_paths("areas.yaml")
    if paths:
        return paths
    else:
        return get_config_path("areas.def")


def get_area_def(area_name):
    """Get the definition of *area_name* from file.

    The file is defined to use is to be placed in the $SATPY_CONFIG_PATH
    directory, and its name is defined in satpy's configuration file.
    """
    try:
        from pyresample import parse_area_file
    except ImportError:
        from pyresample.utils import parse_area_file
    return parse_area_file(get_area_file(), area_name)[0]


def add_xy_coords(data_arr, area, crs=None):
    """Assign x/y coordinates to DataArray from provided area.

    If 'x' and 'y' coordinates already exist then they will not be added.

    Args:
        data_arr (xarray.DataArray): data object to add x/y coordinates to
        area (pyresample.geometry.AreaDefinition): area providing the
            coordinate data.
        crs (pyproj.crs.CRS or None): CRS providing additional information
            about the area's coordinate reference system if available.
            Requires pyproj 2.0+.

    Returns (xarray.DataArray): Updated DataArray object

    """
    if "x" in data_arr.coords and "y" in data_arr.coords:
        # x/y coords already provided
        return data_arr
    if "x" not in data_arr.dims or "y" not in data_arr.dims:
        # no defined x and y dimensions
        return data_arr
    if not hasattr(area, "get_proj_vectors"):
        return data_arr
    x, y = area.get_proj_vectors()

    # convert to DataArrays
    y_attrs = {}
    x_attrs = {}
    if crs is not None:
        units = crs.axis_info[0].unit_name
        # fix udunits/CF standard units
        units = units.replace("metre", "meter")
        if units == "degree":
            y_attrs["units"] = "degrees_north"
            x_attrs["units"] = "degrees_east"
        else:
            y_attrs["units"] = units
            x_attrs["units"] = units
    y = xr.DataArray(y, dims=("y",), attrs=y_attrs)
    x = xr.DataArray(x, dims=("x",), attrs=x_attrs)
    return data_arr.assign_coords(y=y, x=x)


def add_crs_xy_coords(data_arr, area):
    """Add :class:`pyproj.crs.CRS` and x/y or lons/lats to coordinates.

    For SwathDefinition or GridDefinition areas this will add a
    `crs` coordinate and coordinates for the 2D arrays of `lons` and `lats`.

    For AreaDefinition areas this will add a `crs` coordinate and the
    1-dimensional `x` and `y` coordinate variables.

    Args:
        data_arr (xarray.DataArray): DataArray to add the 'crs'
            coordinate.
        area (pyresample.geometry.AreaDefinition): Area to get CRS
            information from.

    """
    # add CRS object if pyproj 2.0+
    try:
        from pyproj import CRS
    except ImportError:
        LOG.debug("Could not add 'crs' coordinate with pyproj<2.0")
        crs = None
    else:
        # default lat/lon projection
        latlon_proj = "+proj=latlong +datum=WGS84 +ellps=WGS84"
        # otherwise get it from the area definition
        if hasattr(area, "crs"):
            crs = area.crs
        else:
            proj_str = getattr(area, "proj_str", latlon_proj)
            crs = CRS.from_string(proj_str)
        data_arr = data_arr.assign_coords(crs=crs)

    # Add x/y coordinates if possible
    if isinstance(area, SwathDefinition):
        # add lon/lat arrays for swath definitions
        # SwathDefinitions created by Satpy should be assigning DataArray
        # objects as the lons/lats attributes so use those directly to
        # maintain original .attrs metadata (instead of converting to dask
        # array).
        lons = area.lons
        lats = area.lats
        lons.attrs.setdefault("standard_name", "longitude")
        lons.attrs.setdefault("long_name", "longitude")
        lons.attrs.setdefault("units", "degrees_east")
        lats.attrs.setdefault("standard_name", "latitude")
        lats.attrs.setdefault("long_name", "latitude")
        lats.attrs.setdefault("units", "degrees_north")
        # See https://github.com/pydata/xarray/issues/3068
        # data_arr = data_arr.assign_coords(longitude=lons, latitude=lats)
    else:
        # Gridded data (AreaDefinition/StackedAreaDefinition)
        data_arr = add_xy_coords(data_arr, area, crs=crs)
    return data_arr


def update_resampled_coords(old_data, new_data, new_area):
    """Add coordinate information to newly resampled DataArray.

    Args:
        old_data (xarray.DataArray): Old data before resampling.
        new_data (xarray.DataArray): New data after resampling.
        new_area (pyresample.geometry.BaseDefinition): Area definition
            for the newly resampled data.

    """
    # copy over other non-x/y coordinates
    # this *MUST* happen before we set 'crs' below otherwise any 'crs'
    # coordinate in the coordinate variables we are copying will overwrite the
    # 'crs' coordinate we just assigned to the data
    ignore_coords = ("y", "x", "crs")
    new_coords = {}
    for cname, cval in old_data.coords.items():
        # we don't want coordinates that depended on the old x/y dimensions
        has_ignored_dims = any(dim in cval.dims for dim in ignore_coords)
        if cname in ignore_coords or has_ignored_dims:
            continue
        new_coords[cname] = cval
    new_data = new_data.assign_coords(**new_coords)

    # add crs, x, and y coordinates
    new_data = add_crs_xy_coords(new_data, new_area)
    return new_data


class KDTreeResampler(PRBaseResampler):
    """Resample using a KDTree-based nearest neighbor algorithm.

    This resampler implements on-disk caching when the `cache_dir` argument
    is provided to the `resample` method. This should provide significant
    performance improvements on consecutive resampling of geostationary data.
    It is not recommended to provide `cache_dir` when the `mask` keyword
    argument is provided to `precompute` which occurs by default for
    `SwathDefinition` source areas.

    Args:
        cache_dir (str): Long term storage directory for intermediate
                         results.
        mask (bool): Force resampled data's invalid pixel mask to be used
                     when searching for nearest neighbor pixels. By
                     default this is True for SwathDefinition source
                     areas and False for all other area definition types.
        radius_of_influence (float): Search radius cut off distance in meters
        epsilon (float): Allowed uncertainty in meters. Increasing uncertainty
                         reduces execution time.

    """

    def __init__(self, source_geo_def, target_geo_def):
        """Init KDTreeResampler."""
        super(KDTreeResampler, self).__init__(source_geo_def, target_geo_def)
        self.resampler = None
        self._index_caches = {}

    def precompute(self, mask=None, radius_of_influence=None, epsilon=0,
                   cache_dir=None, **kwargs):
        """Create a KDTree structure and store it for later use.

        Note: The `mask` keyword should be provided if geolocation may be valid
        where data points are invalid.

        """
        from pyresample.kd_tree import XArrayResamplerNN
        del kwargs
        if mask is not None and cache_dir is not None:
            LOG.warning("Mask and cache_dir both provided to nearest "
                        "resampler. Cached parameters are affected by "
                        "masked pixels. Will not cache results.")
            cache_dir = None

        if radius_of_influence is None and not hasattr(self.source_geo_def, "geocentric_resolution"):
            radius_of_influence = self._adjust_radius_of_influence(radius_of_influence)

        kwargs = dict(source_geo_def=self.source_geo_def,
                      target_geo_def=self.target_geo_def,
                      radius_of_influence=radius_of_influence,
                      neighbours=1,
                      epsilon=epsilon)

        if self.resampler is None:
            # FIXME: We need to move all of this caching logic to pyresample
            self.resampler = XArrayResamplerNN(**kwargs)

        try:
            self.load_neighbour_info(cache_dir, mask=mask, **kwargs)
            LOG.debug("Read pre-computed kd-tree parameters")
        except IOError:
            LOG.debug("Computing kd-tree parameters")
            self.resampler.get_neighbour_info(mask=mask)
            self.save_neighbour_info(cache_dir, mask=mask, **kwargs)

    def _adjust_radius_of_influence(self, radius_of_influence):
        """Adjust radius of influence."""
        warnings.warn(
            "Upgrade 'pyresample' for a more accurate default 'radius_of_influence'.",
            stacklevel=3
        )
        try:
            radius_of_influence = self.source_geo_def.lons.resolution * 3
        except AttributeError:
            try:
                radius_of_influence = max(abs(self.source_geo_def.pixel_size_x),
                                          abs(self.source_geo_def.pixel_size_y)) * 3
            except AttributeError:
                radius_of_influence = 1000

        except TypeError:
            radius_of_influence = 10000
        return radius_of_influence

    def _apply_cached_index(self, val, idx_name, persist=False):
        """Reassign resampler index attributes."""
        if isinstance(val, np.ndarray):
            val = da.from_array(val, chunks=CHUNK_SIZE)
        elif persist and isinstance(val, da.Array):
            val = val.persist()
        setattr(self.resampler, idx_name, val)
        return val

    def load_neighbour_info(self, cache_dir, mask=None, **kwargs):
        """Read index arrays from either the in-memory or disk cache."""
        mask_name = getattr(mask, "name", None)
        cached = {}
        for idx_name in NN_COORDINATES:
            if mask_name in self._index_caches:
                cached[idx_name] = self._apply_cached_index(
                    self._index_caches[mask_name][idx_name], idx_name)
            elif cache_dir:
                try:
                    filename = self._create_cache_filename(
                        cache_dir, prefix="nn_lut-",
                        mask=mask_name, **kwargs)
                    fid = zarr.open(filename, "r")
                    cache = np.array(fid[idx_name])
                    if idx_name == "valid_input_index":
                        # valid input index array needs to be boolean
                        cache = cache.astype(bool)
                except ValueError:
                    raise IOError
                cache = self._apply_cached_index(cache, idx_name)
                cached[idx_name] = cache
            else:
                raise IOError
        self._index_caches[mask_name] = cached

    def save_neighbour_info(self, cache_dir, mask=None, **kwargs):
        """Cache resampler's index arrays if there is a cache dir."""
        if cache_dir:
            mask_name = getattr(mask, "name", None)
            cache = self._read_resampler_attrs()
            filename = self._create_cache_filename(
                cache_dir, prefix="nn_lut-", mask=mask_name, **kwargs)
            LOG.info("Saving kd_tree neighbour info to %s", filename)
            zarr_out = xr.Dataset()
            for idx_name, coord in NN_COORDINATES.items():
                # update the cache in place with persisted dask arrays
                cache[idx_name] = self._apply_cached_index(cache[idx_name],
                                                           idx_name,
                                                           persist=True)
                zarr_out[idx_name] = (coord, cache[idx_name])

            # Write indices to Zarr file
            zarr_out.to_zarr(filename)

            self._index_caches[mask_name] = cache
            # Delete the kdtree, it's not needed anymore
            self.resampler.delayed_kdtree = None

    def _read_resampler_attrs(self):
        """Read certain attributes from the resampler for caching."""
        return {attr_name: getattr(self.resampler, attr_name)
                for attr_name in NN_COORDINATES}

    def compute(self, data, weight_funcs=None, fill_value=np.nan,
                with_uncert=False, **kwargs):
        """Resample data."""
        del kwargs
        LOG.debug("Resampling %s", str(data.name))
        res = self.resampler.get_sample_from_neighbour_info(data, fill_value)
        return update_resampled_coords(data, res, self.target_geo_def)


class BilinearResampler(PRBaseResampler):
    """Resample using bilinear interpolation.

    This resampler implements on-disk caching when the `cache_dir` argument
    is provided to the `resample` method. This should provide significant
    performance improvements on consecutive resampling of geostationary data.

    Args:
        cache_dir (str): Long term storage directory for intermediate
                         results.
        radius_of_influence (float): Search radius cut off distance in meters
        epsilon (float): Allowed uncertainty in meters. Increasing uncertainty
                         reduces execution time.
        reduce_data (bool): Reduce the input data to (roughly) match the
                            target area.

    """

    def __init__(self, source_geo_def, target_geo_def):
        """Init BilinearResampler."""
        super(BilinearResampler, self).__init__(source_geo_def, target_geo_def)
        self.resampler = None

    def precompute(self, mask=None, radius_of_influence=50000, epsilon=0,
                   reduce_data=True, cache_dir=False, **kwargs):
        """Create bilinear coefficients and store them for later use."""
        try:
            from pyresample.bilinear import XArrayBilinearResampler
        except ImportError:
            from pyresample.bilinear import XArrayResamplerBilinear as XArrayBilinearResampler

        del kwargs
        del mask

        if self.resampler is None:
            kwargs = dict(source_geo_def=self.source_geo_def,
                          target_geo_def=self.target_geo_def,
                          radius_of_influence=radius_of_influence,
                          neighbours=32,
                          epsilon=epsilon)

            self.resampler = XArrayBilinearResampler(**kwargs)
            try:
                self.load_bil_info(cache_dir, **kwargs)
                LOG.debug("Loaded bilinear parameters")
            except IOError:
                LOG.debug("Computing bilinear parameters")
                self.resampler.get_bil_info()
                LOG.debug("Saving bilinear parameters.")
                self.save_bil_info(cache_dir, **kwargs)

    def load_bil_info(self, cache_dir, **kwargs):
        """Load bilinear resampling info from cache directory."""
        if cache_dir:
            filename = self._create_cache_filename(cache_dir,
                                                   prefix="bil_lut-",
                                                   **kwargs)
            try:
                self.resampler.load_resampling_info(filename)
            except AttributeError:
                warnings.warn(
                    "Bilinear resampler can't handle caching, "
                    "please upgrade Pyresample to 0.17.0 or newer.",
                    stacklevel=2
                )
                raise IOError
        else:
            raise IOError

    def save_bil_info(self, cache_dir, **kwargs):
        """Save bilinear resampling info to cache directory."""
        if cache_dir:
            filename = self._create_cache_filename(cache_dir,
                                                   prefix="bil_lut-",
                                                   **kwargs)
            # There are some old caches, move them out of the way
            if os.path.exists(filename):
                _move_existing_caches(cache_dir, filename)
            LOG.info("Saving BIL neighbour info to %s", filename)
            try:
                self.resampler.save_resampling_info(filename)
            except AttributeError:
                warnings.warn(
                    "Bilinear resampler can't handle caching, "
                    "please upgrade Pyresample to 0.17.0 or newer.",
                    stacklevel=2
                )

    def compute(self, data, fill_value=None, **kwargs):
        """Resample the given data using bilinear interpolation."""
        del kwargs

        if fill_value is None:
            fill_value = data.attrs.get("_FillValue")
        target_shape = self.target_geo_def.shape

        res = self.resampler.get_sample_from_bil_info(data,
                                                      fill_value=fill_value,
                                                      output_shape=target_shape)

        return update_resampled_coords(data, res, self.target_geo_def)


def _move_existing_caches(cache_dir, filename):
    """Move existing cache files out of the way."""
    import os
    import shutil
    old_cache_dir = os.path.join(cache_dir, "moved_by_satpy")
    try:
        os.makedirs(old_cache_dir)
    except FileExistsError:
        pass
    try:
        shutil.move(filename, old_cache_dir)
    except shutil.Error:
        os.remove(os.path.join(old_cache_dir,
                               os.path.basename(filename)))
        shutil.move(filename, old_cache_dir)
    LOG.warning("Old cache file was moved to %s", old_cache_dir)


def _mean(data, y_size, x_size):
    rows, cols = data.shape
    new_shape = (int(rows / y_size), int(y_size),
                 int(cols / x_size), int(x_size))
    data_mean = np.nanmean(data.reshape(new_shape), axis=(1, 3))
    return data_mean


def _repeat_by_factor(data, block_info=None):
    if block_info is None:
        return data
    out_shape = block_info[None]["chunk-shape"]
    out_data = data
    for axis, axis_size in enumerate(out_shape):
        in_size = data.shape[axis]
        out_data = np.repeat(out_data, int(axis_size / in_size), axis=axis)
    return out_data


class NativeResampler(PRBaseResampler):
    """Expand or reduce input datasets to be the same shape.

    If data is higher resolution (more pixels) than the destination area
    then data is averaged to match the destination resolution.

    If data is lower resolution (less pixels) than the destination area
    then data is repeated to match the destination resolution.

    This resampler does not perform any caching or masking due to the
    simplicity of the operations.

    """

    def resample(self, data, cache_dir=None, mask_area=False, **kwargs):
        """Run NativeResampler."""
        # use 'mask_area' with a default of False. It wouldn't do anything.
        return super(NativeResampler, self).resample(data,
                                                     cache_dir=cache_dir,
                                                     mask_area=mask_area,
                                                     **kwargs)

    @classmethod
    def _expand_reduce(cls, d_arr, repeats):
        """Expand reduce."""
        if not isinstance(d_arr, da.Array):
            d_arr = da.from_array(d_arr, chunks=CHUNK_SIZE)
        if all(x == 1 for x in repeats.values()):
            return d_arr
        if all(x >= 1 for x in repeats.values()):
            return _replicate(d_arr, repeats)
        if all(x <= 1 for x in repeats.values()):
            # reduce
            y_size = 1. / repeats[0]
            x_size = 1. / repeats[1]
            return _aggregate(d_arr, y_size, x_size)
        raise ValueError("Must either expand or reduce in both "
                         "directions")

    def compute(self, data, expand=True, **kwargs):
        """Resample data with NativeResampler."""
        if isinstance(self.target_geo_def, (list, tuple)):
            # find the highest/lowest area among the provided
            test_func = max if expand else min
            target_geo_def = test_func(self.target_geo_def,
                                       key=lambda x: x.shape)
        else:
            target_geo_def = self.target_geo_def

        # convert xarray backed with numpy array to dask array
        if "x" not in data.dims or "y" not in data.dims:
            if data.ndim not in [2, 3]:
                raise ValueError("Can only handle 2D or 3D arrays without dimensions.")
            # assume rows is the second to last axis
            y_axis = data.ndim - 2
            x_axis = data.ndim - 1
        else:
            y_axis = data.dims.index("y")
            x_axis = data.dims.index("x")

        out_shape = target_geo_def.shape
        in_shape = data.shape
        y_repeats = out_shape[0] / float(in_shape[y_axis])
        x_repeats = out_shape[1] / float(in_shape[x_axis])
        repeats = {axis_idx: 1. for axis_idx in range(data.ndim) if axis_idx not in [y_axis, x_axis]}
        repeats[y_axis] = y_repeats
        repeats[x_axis] = x_repeats

        d_arr = self._expand_reduce(data.data, repeats)
        new_data = xr.DataArray(d_arr, dims=data.dims)
        return update_resampled_coords(data, new_data, target_geo_def)


def _aggregate(d, y_size, x_size):
    """Average every 4 elements (2x2) in a 2D array."""
    if d.ndim != 2:
        # we can't guarantee what blocks we are getting and how
        # it should be reshaped to do the averaging.
        raise ValueError("Can't aggregrate (reduce) data arrays with "
                         "more than 2 dimensions.")
    if not (x_size.is_integer() and y_size.is_integer()):
        raise ValueError("Aggregation factors are not integers")
    y_size = int(y_size)
    x_size = int(x_size)
    d = _rechunk_if_nonfactor_chunks(d, y_size, x_size)
    new_chunks = (tuple(int(x / y_size) for x in d.chunks[0]),
                  tuple(int(x / x_size) for x in d.chunks[1]))
    return da.core.map_blocks(_mean, d, y_size, x_size,
                              meta=np.array((), dtype=d.dtype),
                              dtype=d.dtype, chunks=new_chunks)


def _rechunk_if_nonfactor_chunks(dask_arr, y_size, x_size):
    need_rechunk = False
    new_chunks = list(dask_arr.chunks)
    for dim_idx, agg_size in enumerate([y_size, x_size]):
        if dask_arr.shape[dim_idx] % agg_size != 0:
            raise ValueError("Aggregation requires arrays with shapes divisible by the factor.")
        for chunk_size in dask_arr.chunks[dim_idx]:
            if chunk_size % agg_size != 0:
                need_rechunk = True
                new_dim_chunk = lcm(chunk_size, agg_size)
                new_chunks[dim_idx] = new_dim_chunk
    if need_rechunk:
        warnings.warn(
            "Array chunk size is not divisible by aggregation factor. "
            "Re-chunking to continue native resampling.",
            PerformanceWarning,
            stacklevel=5
        )
        dask_arr = dask_arr.rechunk(tuple(new_chunks))
    return dask_arr


def _replicate(d_arr, repeats):
    """Repeat data pixels by the per-axis factors specified."""
    repeated_chunks = _get_replicated_chunk_sizes(d_arr, repeats)
    d_arr = d_arr.map_blocks(_repeat_by_factor,
                             meta=np.array((), dtype=d_arr.dtype),
                             dtype=d_arr.dtype,
                             chunks=repeated_chunks)
    return d_arr


def _get_replicated_chunk_sizes(d_arr, repeats):
    repeated_chunks = []
    for axis, axis_chunks in enumerate(d_arr.chunks):
        factor = repeats[axis]
        if not factor.is_integer():
            raise ValueError("Expand factor must be a whole number")
        repeated_chunks.append(tuple(x * int(factor) for x in axis_chunks))
    return tuple(repeated_chunks)


class BucketResamplerBase(PRBaseResampler):
    """Base class for bucket resampling which implements averaging."""

    def __init__(self, source_geo_def, target_geo_def):
        """Initialize bucket resampler."""
        super(BucketResamplerBase, self).__init__(source_geo_def, target_geo_def)
        self.resampler = None

    def precompute(self, **kwargs):
        """Create X and Y indices and store them for later use."""
        from pyresample import bucket

        LOG.debug("Initializing bucket resampler.")
        source_lons, source_lats = self.source_geo_def.get_lonlats(
            chunks=CHUNK_SIZE)
        self.resampler = bucket.BucketResampler(self.target_geo_def,
                                                source_lons,
                                                source_lats)

    def compute(self, data, **kwargs):
        """Call the resampling."""
        raise NotImplementedError("Use the sub-classes")

    def resample(self, data, **kwargs):  # noqa: D417
        """Resample `data` by calling `precompute` and `compute` methods.

        Args:
            data (xarray.DataArray): Data to be resampled

        Returns (xarray.DataArray): Data resampled to the target area

        """
        self.precompute(**kwargs)
        attrs = data.attrs.copy()
        data_arr = data.data
        if data.ndim == 3 and data.dims[0] == "bands":
            dims = ("bands", "y", "x")
        # Both one and two dimensional input data results in 2D output
        elif data.ndim in (1, 2):
            dims = ("y", "x")
        else:
            dims = data.dims
        LOG.debug("Resampling %s", str(data.attrs.get("_satpy_id", "unknown")))
        result = self.compute(data_arr, **kwargs)
        coords = {}
        if "bands" in data.coords:
            coords["bands"] = data.coords["bands"]
        # Fractions are returned in a dict
        elif isinstance(result, dict):
            coords["categories"] = sorted(result.keys())
            dims = ("categories", "y", "x")
            new_result = []
            for cat in coords["categories"]:
                new_result.append(result[cat])
            result = da.stack(new_result)
        if result.ndim > len(dims):
            result = da.squeeze(result)

        # Adjust some attributes
        if "BucketFraction" in str(self):
            attrs["units"] = ""
            attrs["calibration"] = ""
            attrs["standard_name"] = "area_fraction"
        elif "BucketCount" in str(self):
            attrs["units"] = ""
            attrs["calibration"] = ""
            attrs["standard_name"] = "number_of_observations"

        result = xr.DataArray(result, dims=dims, coords=coords,
                              attrs=attrs)

        return update_resampled_coords(data, result, self.target_geo_def)


class BucketAvg(BucketResamplerBase):
    """Class for averaging bucket resampling.

    Bucket resampling calculates the average of all the values that
    are closest to each bin and inside the target area.

    Parameters
    ----------
    fill_value : float (default: np.nan)
        Fill value to mark missing/invalid values in the input data,
        as well as in the binned and averaged output data.
    skipna : boolean (default: True)
        If True, skips missing values (as marked by NaN or `fill_value`) for the average calculation
        (similarly to Numpy's `nanmean`). Buckets containing only missing values are set to fill_value.
        If False, sets the bucket to fill_value if one or more missing values are present in the bucket
        (similarly to Numpy's `mean`).
        In both cases, empty buckets are set to `fill_value`.

    """

    def compute(self, data, fill_value=np.nan, skipna=True, **kwargs):  # noqa: D417
        """Call the resampling.

        Args:
            data (numpy.Array, dask.Array): Data to be resampled
            fill_value (numpy.nan, int): fill_value. Defaults to numpy.nan
            skipna (boolean): Skip NA's. Default `True`

        Returns:
            dask.Array
        """
        results = []
        if data.ndim == 3:
            for i in range(data.shape[0]):
                res = self.resampler.get_average(data[i, :, :],
                                                 fill_value=fill_value,
                                                 skipna=skipna,
                                                 **kwargs)
                results.append(res)
        else:
            res = self.resampler.get_average(data, fill_value=fill_value, skipna=skipna,
                                             **kwargs)
            results.append(res)

        return da.stack(results)


class BucketSum(BucketResamplerBase):
    """Class for bucket resampling which implements accumulation (sum).

    This resampler calculates the cumulative sum of all the values
    that are closest to each bin and inside the target area.

    Parameters
    ----------
    fill_value : float (default: np.nan)
        Fill value for missing data
    skipna : boolean (default: True)
        If True, skips NaN values for the sum calculation
        (similarly to Numpy's `nansum`). Buckets containing only NaN are set to zero.
        If False, sets the bucket to NaN if one or more NaN values are present in the bucket
        (similarly to Numpy's `sum`).
        In both cases, empty buckets are set to 0.

    """

    def compute(self, data, skipna=True, **kwargs):
        """Call the resampling."""
        results = []
        if data.ndim == 3:
            for i in range(data.shape[0]):
                res = self.resampler.get_sum(data[i, :, :], skipna=skipna,
                                             **kwargs)
                results.append(res)
        else:
            res = self.resampler.get_sum(data, skipna=skipna, **kwargs)
            results.append(res)

        return da.stack(results)


class BucketCount(BucketResamplerBase):
    """Class for bucket resampling which implements hit-counting.

    This resampler calculates the number of occurences of the input
    data closest to each bin and inside the target area.

    """

    def compute(self, data, **kwargs):
        """Call the resampling."""
        results = []
        if data.ndim == 3:
            for _i in range(data.shape[0]):
                res = self.resampler.get_count()
                results.append(res)
        else:
            res = self.resampler.get_count()
            results.append(res)

        return da.stack(results)


class BucketFraction(BucketResamplerBase):
    """Class for bucket resampling to compute category fractions.

    This resampler calculates the fraction of occurences of the input
    data per category.

    """

    def compute(self, data, fill_value=np.nan, categories=None, **kwargs):
        """Call the resampling."""
        if data.ndim > 2:
            raise ValueError("BucketFraction not implemented for 3D datasets")

        result = self.resampler.get_fractions(data, categories=categories,
                                              fill_value=fill_value)

        return result


# TODO: move this to pyresample.resampler
RESAMPLERS = {"kd_tree": KDTreeResampler,
              "nearest": KDTreeResampler,
              "bilinear": BilinearResampler,
              "native": NativeResampler,
              "gradient_search": create_gradient_search_resampler,
              "bucket_avg": BucketAvg,
              "bucket_sum": BucketSum,
              "bucket_count": BucketCount,
              "bucket_fraction": BucketFraction,
              "ewa": DaskEWAResampler,
              "ewa_legacy": LegacyDaskEWAResampler,
              }


# TODO: move this to pyresample
def prepare_resampler(source_area, destination_area, resampler=None, **resample_kwargs):
    """Instantiate and return a resampler."""
    if resampler is None:
        LOG.info("Using default KDTree resampler")
        resampler = "kd_tree"

    if isinstance(resampler, PRBaseResampler):
        raise ValueError("Trying to create a resampler when one already "
                         "exists.")
    if isinstance(resampler, str):
        resampler_class = RESAMPLERS.get(resampler, None)
        if resampler_class is None:
            if resampler == "gradient_search":
                warnings.warn(
                    "Gradient search resampler not available. Maybe missing `shapely`?",
                    stacklevel=2
                )
            raise KeyError("Resampler '%s' not available" % resampler)
    else:
        resampler_class = resampler

    key = (resampler_class,
           source_area, destination_area,
           hash_dict(resample_kwargs).hexdigest())
    try:
        resampler_instance = resamplers_cache[key]
    except KeyError:
        resampler_instance = resampler_class(source_area, destination_area)
        resamplers_cache[key] = resampler_instance
    return key, resampler_instance


# TODO: move this to pyresample
def resample(source_area, data, destination_area,
             resampler=None, **kwargs):
    """Do the resampling."""
    if not isinstance(resampler, PRBaseResampler):
        # we don't use the first argument (cache key)
        _, resampler_instance = prepare_resampler(source_area,
                                                  destination_area,
                                                  resampler)
    else:
        resampler_instance = resampler

    if isinstance(data, list):
        res = [resampler_instance.resample(ds, **kwargs) for ds in data]
    else:
        res = resampler_instance.resample(data, **kwargs)

    return res


def get_fill_value(dataset):
    """Get the fill value of the *dataset*, defaulting to np.nan."""
    if np.issubdtype(dataset.dtype, np.integer):
        return dataset.attrs.get("_FillValue", np.nan)
    return np.nan


def resample_dataset(dataset, destination_area, **kwargs):
    """Resample *dataset* and return the resampled version.

    Args:
        dataset (xarray.DataArray): Data to be resampled.
        destination_area: The destination onto which to project the data,
          either a full blown area definition or a string corresponding to
          the name of the area as defined in the area file.
        **kwargs: The extra parameters to pass to the resampler objects.

    Returns:
        A resampled DataArray with updated ``.attrs["area"]`` field. The dtype
        of the array is preserved.

    """
    # call the projection stuff here
    try:
        source_area = dataset.attrs["area"]
    except KeyError:
        LOG.info("Cannot reproject dataset %s, missing area info",
                 dataset.attrs["name"])

        return dataset

    fill_value = kwargs.pop("fill_value", get_fill_value(dataset))
    new_data = resample(source_area, dataset, destination_area, fill_value=fill_value, **kwargs)
    new_attrs = new_data.attrs
    new_data.attrs = dataset.attrs.copy()
    new_data.attrs.update(new_attrs)
    new_data.attrs.update(area=destination_area)

    return new_data
