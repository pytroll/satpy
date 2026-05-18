"""Resamplers based on the kdtree algorightm."""

import os
import warnings
from logging import getLogger

import dask.array as da
import numpy as np
import xarray as xr
import zarr
from pyresample.resampler import BaseResampler as PRBaseResampler

from satpy.resample.base import _update_resampled_coords
from satpy.utils import get_legacy_chunk_size

LOG = getLogger(__name__)

CHUNK_SIZE = get_legacy_chunk_size()

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
                cache = self._load_neighbour_info_from_cache(
                    cache_dir, idx_name, mask_name, **kwargs)
                cache = self._apply_cached_index(cache, idx_name)
                cached[idx_name] = cache
            else:
                raise IOError
        self._index_caches[mask_name] = cached

    def _load_neighbour_info_from_cache(self, cache_dir, idx_name, mask_name, **kwargs):
        try:
            filename = self._create_cache_filename(
                cache_dir, prefix="nn_lut-",
                mask=mask_name, **kwargs)
            fid = zarr.open(filename, mode="r")
            cache = np.array(fid[idx_name])
            if idx_name == "valid_input_index":
                # valid input index array needs to be boolean
                cache = cache.astype(bool)
        except ValueError:
            raise IOError
        return cache

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
        return _update_resampled_coords(data, res, self.target_geo_def)


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

        return _update_resampled_coords(data, res, self.target_geo_def)


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


def get_resampler_classes():
    """Get resampler classes based on kdtree."""
    return {
        "kd_tree": KDTreeResampler,
        "nearest": KDTreeResampler,
        "bilinear": BilinearResampler
    }
