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
"""Satpy resampling module.

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
    "ewa", "Elliptical Weighted Averaging", :class:`~satpy.resample.EWAResampler`
    "native", "Native", :class:`~satpy.resample.NativeResampler`
    "bilinear", "Bilinear", :class:`~satpy.resample.BilinearResampler`
    "bucket_avg", "Average Bucket Resampling", :class:`~satpy.resample.BucketAvg`
    "bucket_sum", "Sum Bucket Resampling", :class:`~satpy.resample.BucketSum`
    "bucket_count", "Count Bucket Resampling", :class:`~satpy.resample.BucketCount`
    "bucket_fraction", "Fraction Bucket Resampling", :class:`~satpy.resample.BucketFraction`
    "gradient_search", "Gradient Search Resampling", :class:`~pyresample.gradient.GradientSearchResampler`

The resampling algorithm used can be specified with the ``resampler`` keyword
argument and defaults to ``nearest``:

.. code-block:: python

    >>> scn = Scene(...)
    >>> euro_scn = global_scene.resample('euro4', resampler='nearest')

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
:meth:`highest resolution area <satpy.scene.Scene.max_area>` (smallest footprint per
pixel) shared between the loaded datasets. You can easily specify the lower
resolution area:

.. code-block:: python

    >>> new_scn = scn.resample(scn.min_area(), resampler='native')

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
    >>> local_scene = global_scene.resample(my_area)

Create dynamic area definition
------------------------------

See :class:`pyresample.geometry.DynamicAreaDefinition` for more information.

Examples coming soon...

Store area definitions
----------------------

Area definitions can be added to a custom YAML file (see
`pyresample's documentation <http://pyresample.readthedocs.io/en/stable/geo_def.html#pyresample-utils>`_
for more information)
and loaded using pyresample's utility methods::

    >>> from pyresample.utils import parse_area_file
    >>> my_area = parse_area_file('my_areas.yaml', 'my_area')[0]

Examples coming soon...

"""
import hashlib
import json
import os
from logging import getLogger
from weakref import WeakValueDictionary
import warnings
import numpy as np
import xarray as xr
import dask
import dask.array as da
import zarr

from pyresample.ewa import fornav, ll2cr
from pyresample.geometry import SwathDefinition
try:
    from pyresample.resampler import BaseResampler as PRBaseResampler
    from pyresample.gradient import GradientSearchResampler
except ImportError:
    warnings.warn('Gradient search resampler not available, upgrade Pyresample.')
    PRBaseResampler = None
    GradientSearchResampler = None

from satpy import CHUNK_SIZE
from satpy.config import config_search_paths, get_config_path


LOG = getLogger(__name__)

CACHE_SIZE = 10
NN_COORDINATES = {'valid_input_index': ('y1', 'x1'),
                  'valid_output_index': ('y2', 'x2'),
                  'index_array': ('y2', 'x2', 'z2')}
BIL_COORDINATES = {'bilinear_s': ('x1', ),
                   'bilinear_t': ('x1', ),
                   'slices_x': ('x1', 'n'),
                   'slices_y': ('x1', 'n'),
                   'mask_slices': ('x1', 'n'),
                   'out_coords_x': ('x2', ),
                   'out_coords_y': ('y2', )}

resamplers_cache = WeakValueDictionary()


def hash_dict(the_dict, the_hash=None):
    """Calculate a hash for a dictionary."""
    if the_hash is None:
        the_hash = hashlib.sha1()
    the_hash.update(json.dumps(the_dict, sort_keys=True).encode('utf-8'))
    return the_hash


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
    if 'x' in data_arr.coords and 'y' in data_arr.coords:
        # x/y coords already provided
        return data_arr
    elif 'x' not in data_arr.dims or 'y' not in data_arr.dims:
        # no defined x and y dimensions
        return data_arr

    if hasattr(area, 'get_proj_vectors'):
        x, y = area.get_proj_vectors()
    else:
        return data_arr

    # convert to DataArrays
    y_attrs = {}
    x_attrs = {}
    if crs is not None:
        units = crs.axis_info[0].unit_name
        # fix udunits/CF standard units
        units = units.replace('metre', 'meter')
        if units == 'degree':
            y_attrs['units'] = 'degrees_north'
            x_attrs['units'] = 'degrees_east'
        else:
            y_attrs['units'] = units
            x_attrs['units'] = units
    y = xr.DataArray(y, dims=('y',), attrs=y_attrs)
    x = xr.DataArray(x, dims=('x',), attrs=x_attrs)
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
        if hasattr(area, 'crs'):
            crs = area.crs
        else:
            proj_str = getattr(area, 'proj_str', latlon_proj)
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
        lons.attrs.setdefault('standard_name', 'longitude')
        lons.attrs.setdefault('long_name', 'longitude')
        lons.attrs.setdefault('units', 'degrees_east')
        lats.attrs.setdefault('standard_name', 'latitude')
        lats.attrs.setdefault('long_name', 'latitude')
        lats.attrs.setdefault('units', 'degrees_north')
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
    ignore_coords = ('y', 'x', 'crs')
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


class BaseResampler(object):
    """Base abstract resampler class."""

    def __init__(self, source_geo_def, target_geo_def):
        """Initialize resampler with geolocation information.

        Args:
            source_geo_def (SwathDefinition, AreaDefinition):
                Geolocation definition for the data to be resampled
            target_geo_def (CoordinateDefinition, AreaDefinition):
                Geolocation definition for the area to resample data to.

        """
        self.source_geo_def = source_geo_def
        self.target_geo_def = target_geo_def

    def get_hash(self, source_geo_def=None, target_geo_def=None, **kwargs):
        """Get hash for the current resample with the given *kwargs*."""
        if source_geo_def is None:
            source_geo_def = self.source_geo_def
        if target_geo_def is None:
            target_geo_def = self.target_geo_def
        the_hash = source_geo_def.update_hash()
        target_geo_def.update_hash(the_hash)
        hash_dict(kwargs, the_hash)
        return the_hash.hexdigest()

    def precompute(self, **kwargs):
        """Do the precomputation.

        This is an optional step if the subclass wants to implement more
        complex features like caching or can share some calculations
        between multiple datasets to be processed.

        """
        return None

    def compute(self, data, **kwargs):
        """Do the actual resampling.

        This must be implemented by subclasses.

        """
        raise NotImplementedError

    def resample(self, data, cache_dir=None, mask_area=None, **kwargs):
        """Resample `data` by calling `precompute` and `compute` methods.

        Only certain resampling classes may use `cache_dir` and the `mask`
        provided when `mask_area` is True. The return value of calling the
        `precompute` method is passed as the `cache_id` keyword argument
        of the `compute` method, but may not be used directly for caching. It
        is up to the individual resampler subclasses to determine how this
        is used.

        Args:
            data (xarray.DataArray): Data to be resampled
            cache_dir (str): directory to cache precomputed results
                             (default False, optional)
            mask_area (bool): Mask geolocation data where data values are
                              invalid. This should be used when data values
                              may affect what neighbors are considered valid.

        Returns (xarray.DataArray): Data resampled to the target area

        """
        # default is to mask areas for SwathDefinitions
        if mask_area is None and isinstance(
                self.source_geo_def, SwathDefinition):
            mask_area = True

        if mask_area:
            if isinstance(self.source_geo_def, SwathDefinition):
                geo_dims = self.source_geo_def.lons.dims
            else:
                geo_dims = ('y', 'x')
            flat_dims = [dim for dim in data.dims if dim not in geo_dims]
            if np.issubdtype(data.dtype, np.integer):
                kwargs['mask'] = data == data.attrs.get('_FillValue', np.iinfo(data.dtype.type).max)
            else:
                kwargs['mask'] = data.isnull()
            kwargs['mask'] = kwargs['mask'].all(dim=flat_dims)

        cache_id = self.precompute(cache_dir=cache_dir, **kwargs)
        return self.compute(data, cache_id=cache_id, **kwargs)

    def _create_cache_filename(self, cache_dir=None, prefix='',
                               fmt='.zarr', **kwargs):
        """Create filename for the cached resampling parameters."""
        cache_dir = cache_dir or '.'
        hash_str = self.get_hash(**kwargs)

        return os.path.join(cache_dir, prefix + hash_str + fmt)


class KDTreeResampler(BaseResampler):
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

        if radius_of_influence is None and not hasattr(self.source_geo_def, 'geocentric_resolution'):
            warnings.warn("Upgrade 'pyresample' for a more accurate default 'radius_of_influence'.")
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

    def _apply_cached_index(self, val, idx_name, persist=False):
        """Reassign resampler index attributes."""
        if isinstance(val, np.ndarray):
            val = da.from_array(val, chunks=CHUNK_SIZE)
        elif persist and isinstance(val, da.Array):
            val = val.persist()
        setattr(self.resampler, idx_name, val)
        return val

    def _check_numpy_cache(self, cache_dir, mask=None,
                           **kwargs):
        """Check if there's Numpy cache file and convert it to zarr."""
        fname_np = self._create_cache_filename(cache_dir,
                                               prefix='resample_lut-',
                                               mask=mask, fmt='.npz',
                                               **kwargs)
        fname_zarr = self._create_cache_filename(cache_dir, prefix='nn_lut-',
                                                 mask=mask, fmt='.zarr',
                                                 **kwargs)
        LOG.debug("Check if %s exists", fname_np)
        if os.path.exists(fname_np) and not os.path.exists(fname_zarr):
            import warnings
            warnings.warn("Using Numpy files as resampling cache is "
                          "deprecated.")
            LOG.warning("Converting resampling LUT from .npz to .zarr")
            zarr_out = xr.Dataset()
            with np.load(fname_np, 'r') as fid:
                for idx_name, coord in NN_COORDINATES.items():
                    zarr_out[idx_name] = (coord, fid[idx_name])

            # Write indices to Zarr file
            zarr_out.to_zarr(fname_zarr)
            LOG.debug("Resampling LUT saved to %s", fname_zarr)

    def load_neighbour_info(self, cache_dir, mask=None, **kwargs):
        """Read index arrays from either the in-memory or disk cache."""
        mask_name = getattr(mask, 'name', None)
        cached = {}
        self._check_numpy_cache(cache_dir, mask=mask_name, **kwargs)

        filename = self._create_cache_filename(cache_dir, prefix='nn_lut-',
                                               mask=mask_name, **kwargs)
        for idx_name in NN_COORDINATES.keys():
            if mask_name in self._index_caches:
                cached[idx_name] = self._apply_cached_index(
                    self._index_caches[mask_name][idx_name], idx_name)
            elif cache_dir:
                try:
                    fid = zarr.open(filename, 'r')
                    cache = np.array(fid[idx_name])
                    if idx_name == 'valid_input_index':
                        # valid input index array needs to be boolean
                        cache = cache.astype(np.bool)
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
            mask_name = getattr(mask, 'name', None)
            cache = self._read_resampler_attrs()
            filename = self._create_cache_filename(
                cache_dir, prefix='nn_lut-', mask=mask_name, **kwargs)
            LOG.info('Saving kd_tree neighbour info to %s', filename)
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
                for attr_name in NN_COORDINATES.keys()}

    def compute(self, data, weight_funcs=None, fill_value=np.nan,
                with_uncert=False, **kwargs):
        """Resample data."""
        del kwargs
        LOG.debug("Resampling %s", str(data.name))
        res = self.resampler.get_sample_from_neighbour_info(data, fill_value)
        return update_resampled_coords(data, res, self.target_geo_def)


class EWAResampler(BaseResampler):
    """Resample using an elliptical weighted averaging algorithm.

    This algorithm does **not** use caching or any externally provided data
    mask (unlike the 'nearest' resampler).

    This algorithm works under the assumption that the data is observed
    one scan line at a time. However, good results can still be achieved
    for non-scan based data provided `rows_per_scan` is set to the
    number of rows in the entire swath or by setting it to `None`.

    Args:
        rows_per_scan (int, None):
            Number of data rows for every observed scanline. If None then the
            entire swath is treated as one large scanline.
        weight_count (int):
            number of elements to create in the gaussian weight table.
            Default is 10000. Must be at least 2
        weight_min (float):
            the minimum value to store in the last position of the
            weight table. Default is 0.01, which, with a
            `weight_distance_max` of 1.0 produces a weight of 0.01
            at a grid cell distance of 1.0. Must be greater than 0.
        weight_distance_max (float):
            distance in grid cell units at which to
            apply a weight of `weight_min`. Default is
            1.0. Must be greater than 0.
        weight_delta_max (float):
            maximum distance in grid cells in each grid
            dimension over which to distribute a single swath cell.
            Default is 10.0.
        weight_sum_min (float):
            minimum weight sum value. Cells whose weight sums
            are less than `weight_sum_min` are set to the grid fill value.
            Default is EPSILON.
        maximum_weight_mode (bool):
            If False (default), a weighted average of
            all swath cells that map to a particular grid cell is used.
            If True, the swath cell having the maximum weight of all
            swath cells that map to a particular grid cell is used. This
            option should be used for coded/category data, i.e. snow cover.

    """

    def __init__(self, source_geo_def, target_geo_def):
        """Init EWAResampler."""
        super(EWAResampler, self).__init__(source_geo_def, target_geo_def)
        self.cache = {}

    def resample(self, *args, **kwargs):
        """Run precompute and compute methods.

        .. note::

            This sets the default of 'mask_area' to False since it is
            not needed in EWA resampling currently.

        """
        kwargs.setdefault('mask_area', False)
        return super(EWAResampler, self).resample(*args, **kwargs)

    def _call_ll2cr(self, lons, lats, target_geo_def, swath_usage=0):
        """Wrap ll2cr() for handling dask delayed calls better."""
        new_src = SwathDefinition(lons, lats)

        swath_points_in_grid, cols, rows = ll2cr(new_src, target_geo_def)
        # FIXME: How do we check swath usage/coverage if we only do this
        #        per-block
        # # Determine if enough of the input swath was used
        # grid_name = getattr(self.target_geo_def, "name", "N/A")
        # fraction_in = swath_points_in_grid / float(lons.size)
        # swath_used = fraction_in > swath_usage
        # if not swath_used:
        #     LOG.info("Data does not fit in grid %s because it only %f%% of "
        #              "the swath is used" %
        #              (grid_name, fraction_in * 100))
        #     raise RuntimeError("Data does not fit in grid %s" % (grid_name,))
        # else:
        #     LOG.debug("Data fits in grid %s and uses %f%% of the swath",
        #               grid_name, fraction_in * 100)

        return np.stack([cols, rows], axis=0)

    def precompute(self, cache_dir=None, swath_usage=0, **kwargs):
        """Generate row and column arrays and store it for later use."""
        if self.cache:
            # this resampler should be used for one SwathDefinition
            # no need to recompute ll2cr output again
            return None

        if kwargs.get('mask') is not None:
            LOG.warning("'mask' parameter has no affect during EWA "
                        "resampling")

        del kwargs
        source_geo_def = self.source_geo_def
        target_geo_def = self.target_geo_def

        if cache_dir:
            LOG.warning("'cache_dir' is not used by EWA resampling")

        # Satpy/PyResample don't support dynamic grids out of the box yet
        lons, lats = source_geo_def.get_lonlats()
        if isinstance(lons, xr.DataArray):
            # get dask arrays
            lons = lons.data
            lats = lats.data
        # we are remapping to a static unchanging grid/area with all of
        # its parameters specified
        chunks = (2,) + lons.chunks
        res = da.map_blocks(self._call_ll2cr, lons, lats,
                            target_geo_def, swath_usage,
                            dtype=lons.dtype, chunks=chunks, new_axis=[0])
        cols = res[0]
        rows = res[1]

        # save the dask arrays in the class instance cache
        # the on-disk cache will store the numpy arrays
        self.cache = {
            "rows": rows,
            "cols": cols,
        }

        return None

    def _call_fornav(self, cols, rows, target_geo_def, data,
                     grid_coverage=0, **kwargs):
        """Wrap fornav() to run as a dask delayed."""
        num_valid_points, res = fornav(cols, rows, target_geo_def,
                                       data, **kwargs)

        if isinstance(data, tuple):
            # convert 'res' from tuple of arrays to one array
            res = np.stack(res)
            num_valid_points = sum(num_valid_points)

        grid_covered_ratio = num_valid_points / float(res.size)
        grid_covered = grid_covered_ratio > grid_coverage
        if not grid_covered:
            msg = "EWA resampling only found %f%% of the grid covered " \
                  "(need %f%%)" % (grid_covered_ratio * 100,
                                   grid_coverage * 100)
            raise RuntimeError(msg)
        LOG.debug("EWA resampling found %f%% of the grid covered" %
                  (grid_covered_ratio * 100))

        return res

    def compute(self, data, cache_id=None, fill_value=0, weight_count=10000,
                weight_min=0.01, weight_distance_max=1.0,
                weight_delta_max=1.0, weight_sum_min=-1.0,
                maximum_weight_mode=False, grid_coverage=0, **kwargs):
        """Resample the data according to the precomputed X/Y coordinates."""
        rows = self.cache["rows"]
        cols = self.cache["cols"]

        # if the data is scan based then check its metadata or the passed
        # kwargs otherwise assume the entire input swath is one large
        # "scanline"
        rows_per_scan = kwargs.get('rows_per_scan',
                                   data.attrs.get("rows_per_scan",
                                                  data.shape[0]))

        if data.ndim == 3 and 'bands' in data.dims:
            data_in = tuple(data.sel(bands=band).data
                            for band in data['bands'])
        elif data.ndim == 2:
            data_in = data.data
        else:
            raise ValueError("Unsupported data shape for EWA resampling.")

        res = dask.delayed(self._call_fornav)(
            cols, rows, self.target_geo_def, data_in,
            grid_coverage=grid_coverage,
            rows_per_scan=rows_per_scan, weight_count=weight_count,
            weight_min=weight_min, weight_distance_max=weight_distance_max,
            weight_delta_max=weight_delta_max, weight_sum_min=weight_sum_min,
            maximum_weight_mode=maximum_weight_mode)
        if isinstance(data_in, tuple):
            new_shape = (len(data_in),) + self.target_geo_def.shape
        else:
            new_shape = self.target_geo_def.shape
        data_arr = da.from_delayed(res, new_shape, data.dtype)
        # from delayed creates one large chunk, break it up a bit if we can
        data_arr = data_arr.rechunk([CHUNK_SIZE] * data_arr.ndim)
        if data.ndim == 3 and data.dims[0] == 'bands':
            dims = ('bands', 'y', 'x')
        elif data.ndim == 2:
            dims = ('y', 'x')
        else:
            dims = data.dims

        res = xr.DataArray(data_arr, dims=dims, attrs=data.attrs.copy())
        return update_resampled_coords(data, res, self.target_geo_def)


class BilinearResampler(BaseResampler):
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
        from pyresample.bilinear.xarr import XArrayResamplerBilinear

        del kwargs
        del mask

        if self.resampler is None:
            kwargs = dict(source_geo_def=self.source_geo_def,
                          target_geo_def=self.target_geo_def,
                          radius_of_influence=radius_of_influence,
                          neighbours=32,
                          epsilon=epsilon)

            self.resampler = XArrayResamplerBilinear(**kwargs)
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
                                                   prefix='bil_lut-',
                                                   **kwargs)
            try:
                fid = zarr.open(filename, 'r')
                for val in BIL_COORDINATES.keys():
                    cache = np.array(fid[val])
                    setattr(self.resampler, val, cache)
            except ValueError:
                raise IOError
        else:
            raise IOError

    def save_bil_info(self, cache_dir, **kwargs):
        """Save bilinear resampling info to cache directory."""
        if cache_dir:
            filename = self._create_cache_filename(cache_dir,
                                                   prefix='bil_lut-',
                                                   **kwargs)
            # There are some old caches, move them out of the way
            if os.path.exists(filename):
                _move_existing_caches(cache_dir, filename)
            LOG.info('Saving BIL neighbour info to %s', filename)
            zarr_out = xr.Dataset()
            for idx_name, coord in BIL_COORDINATES.items():
                var = getattr(self.resampler, idx_name)
                if isinstance(var, np.ndarray):
                    var = da.from_array(var, chunks=CHUNK_SIZE)
                else:
                    var = var.rechunk(CHUNK_SIZE)
                zarr_out[idx_name] = (coord, var)
            zarr_out.to_zarr(filename)

    def compute(self, data, fill_value=None, **kwargs):
        """Resample the given data using bilinear interpolation."""
        del kwargs

        if fill_value is None:
            fill_value = data.attrs.get('_FillValue')
        target_shape = self.target_geo_def.shape

        res = self.resampler.get_sample_from_bil_info(data,
                                                      fill_value=fill_value,
                                                      output_shape=target_shape)

        return update_resampled_coords(data, res, self.target_geo_def)


def _move_existing_caches(cache_dir, filename):
    """Move existing cache files out of the way."""
    import os
    import shutil
    old_cache_dir = os.path.join(cache_dir, 'moved_by_satpy')
    try:
        os.mkdir(old_cache_dir)
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


class NativeResampler(BaseResampler):
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

    @staticmethod
    def aggregate(d, y_size, x_size):
        """Average every 4 elements (2x2) in a 2D array."""
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
        return da.core.map_blocks(_mean, d, y_size, x_size, dtype=d.dtype, chunks=new_chunks)

    @classmethod
    def expand_reduce(cls, d_arr, repeats):
        """Expand reduce."""
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
        """Resample data with NativeResampler."""
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
                raise ValueError("Can only handle 2D or 3D arrays without dimensions.")
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
        repeats = {axis_idx: 1. for axis_idx in range(data.ndim) if axis_idx not in [y_axis, x_axis]}
        repeats[y_axis] = y_repeats
        repeats[x_axis] = x_repeats

        d_arr = self.expand_reduce(data.data, repeats)
        new_data = xr.DataArray(d_arr, dims=data.dims)
        return update_resampled_coords(data, new_data, target_geo_def)


class BucketResamplerBase(BaseResampler):
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

    def resample(self, data, **kwargs):
        """Resample `data` by calling `precompute` and `compute` methods.

        Args:
            data (xarray.DataArray): Data to be resampled

        Returns (xarray.DataArray): Data resampled to the target area

        """
        self.precompute(**kwargs)
        attrs = data.attrs.copy()
        data_arr = data.data
        if data.ndim == 3 and data.dims[0] == 'bands':
            dims = ('bands', 'y', 'x')
        # Both one and two dimensional input data results in 2D output
        elif data.ndim in (1, 2):
            dims = ('y', 'x')
        else:
            dims = data.dims
        result = self.compute(data_arr, **kwargs)
        coords = {}
        if 'bands' in data.coords:
            coords['bands'] = data.coords['bands']
        # Fractions are returned in a dict
        elif isinstance(result, dict):
            coords['categories'] = sorted(result.keys())
            dims = ('categories', 'y', 'x')
            new_result = []
            for cat in coords['categories']:
                new_result.append(result[cat])
            result = da.stack(new_result)
        if result.ndim > len(dims):
            result = da.squeeze(result)

        # Adjust some attributes
        if "BucketFraction" in str(self):
            attrs['units'] = ''
            attrs['calibration'] = ''
            attrs['standard_name'] = 'area_fraction'
        elif "BucketCount" in str(self):
            attrs['units'] = ''
            attrs['calibration'] = ''
            attrs['standard_name'] = 'number_of_observations'

        result = xr.DataArray(result, dims=dims, coords=coords,
                              attrs=attrs)

        return result


class BucketAvg(BucketResamplerBase):
    """Class for averaging bucket resampling.

    Bucket resampling calculates the average of all the values that
    are closest to each bin and inside the target area.

    Parameters
    ----------
    fill_value : float (default: np.nan)
        Fill value for missing data
    mask_all_nans : boolean (default: False)
        Mask all locations with all-NaN values

    """

    def compute(self, data, fill_value=np.nan, mask_all_nan=False, **kwargs):
        """Call the resampling."""
        results = []
        if data.ndim == 3:
            for i in range(data.shape[0]):
                res = self.resampler.get_average(data[i, :, :],
                                                 fill_value=fill_value,
                                                 mask_all_nan=mask_all_nan)
                results.append(res)
        else:
            res = self.resampler.get_average(data, fill_value=fill_value,
                                             mask_all_nan=mask_all_nan)
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
    mask_all_nans : boolean (default: False)
        Mask all locations with all-NaN values

    """

    def compute(self, data, mask_all_nan=False, **kwargs):
        """Call the resampling."""
        LOG.debug("Resampling %s", str(data.name))
        results = []
        if data.ndim == 3:
            for i in range(data.shape[0]):
                res = self.resampler.get_sum(data[i, :, :],
                                             mask_all_nan=mask_all_nan)
                results.append(res)
        else:
            res = self.resampler.get_sum(data, mask_all_nan=mask_all_nan)
            results.append(res)

        return da.stack(results)


class BucketCount(BucketResamplerBase):
    """Class for bucket resampling which implements hit-counting.

    This resampler calculates the number of occurences of the input
    data closest to each bin and inside the target area.

    """

    def compute(self, data, **kwargs):
        """Call the resampling."""
        LOG.debug("Resampling %s", str(data.name))
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
        LOG.debug("Resampling %s", str(data.name))
        if data.ndim > 2:
            raise ValueError("BucketFraction not implemented for 3D datasets")

        result = self.resampler.get_fractions(data, categories=categories,
                                              fill_value=fill_value)

        return result


# TODO: move this to pyresample.resampler
RESAMPLERS = {"kd_tree": KDTreeResampler,
              "nearest": KDTreeResampler,
              "ewa": EWAResampler,
              "bilinear": BilinearResampler,
              "native": NativeResampler,
              "gradient_search": GradientSearchResampler,
              "bucket_avg": BucketAvg,
              "bucket_sum": BucketSum,
              "bucket_count": BucketCount,
              "bucket_fraction": BucketFraction,
              }


if PRBaseResampler is None:
    PRBaseResampler = BaseResampler


# TODO: move this to pyresample
def prepare_resampler(source_area, destination_area, resampler=None, **resample_kwargs):
    """Instantiate and return a resampler."""
    if resampler is None:
        LOG.info("Using default KDTree resampler")
        resampler = 'kd_tree'

    if isinstance(resampler, (BaseResampler, PRBaseResampler)):
        raise ValueError("Trying to create a resampler when one already "
                         "exists.")
    elif isinstance(resampler, str):
        resampler_class = RESAMPLERS.get(resampler, None)
        if resampler_class is None:
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
    if not isinstance(resampler, (BaseResampler, PRBaseResampler)):
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
        return dataset.attrs.get('_FillValue', np.nan)
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
                 dataset.attrs['name'])

        return dataset

    fill_value = kwargs.pop('fill_value', get_fill_value(dataset))
    new_data = resample(source_area, dataset, destination_area, fill_value=fill_value, **kwargs)
    new_attrs = new_data.attrs
    new_data.attrs = dataset.attrs.copy()
    new_data.attrs.update(new_attrs)
    new_data.attrs.update(area=destination_area)

    return new_data
